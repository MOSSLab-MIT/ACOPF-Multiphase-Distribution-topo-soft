# Correlated DER sampler, DER sigma handling in configs, and shift clamps.
# Run: julia --project=datagen_topo/FeederFlow.jl tests/test_profile_sampling.jl
using Test
using Random
using Statistics
using Distributions

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const DATAGEN_DIR = joinpath(ROOT_DIR, "datagen_topo")

include(joinpath(DATAGEN_DIR, "datagen_topo.jl"))
include(joinpath(DATAGEN_DIR, "src", "generation", "training.jl"))
include(joinpath(DATAGEN_DIR, "src", "generation", "testing.jl"))

const CONFIG_DIR = joinpath(DATAGEN_DIR, "gen_scripts", "configs")
const CASE_13 = joinpath(DATAGEN_DIR, "grid_data", "13_bus", "IEEE13Nodeckt.dss")
const LOAD_SIGMA = 0.3
const DER_SIGMA = 0.5

der_options(l, u, σ) = Dict{String, Any}("frac_l" => l, "frac_u" => u, "sigma" => σ)

# Monte Carlo draws of availability / rated, one row per sample.
function der_fractions(ds, n; first_seed=1)
    rows = [Random.rand(MersenneTwister(first_seed + k), ds) ./ ds.rated for k in 0:(n - 1)]
    return reduce(vcat, permutedims.(rows))
end

within_mc_error(values, expected) = abs(mean(values) - expected) <= 4 * std(values) / sqrt(length(values))

@testset "DERScaler" begin
    network = create_network_data(CASE_13)
    rated = copy(network.der_pavail_ref)
    D = network.n_der_unit
    @test D > 1
    @test all(rated .> 0)

    @testset "draws match the formula exactly" begin
        l, u, σ = 0.1, 0.9, DER_SIGMA
        ds = DERScaler(network, der_options(l, u, σ))
        for seed in 1:200
            replay = MersenneTwister(seed)
            α = rand(replay, Uniform(l, u))
            η = [rand(replay, Uniform(1 - σ, 1 + σ)) for _ in 1:D]
            @test Random.rand(MersenneTwister(seed), ds) == min.(α .* η, 1.0) .* rated
        end
    end

    @testset "sigma 0 gives one shared factor" begin
        l, u = 0.2, 0.7
        F = der_fractions(DERScaler(network, der_options(l, u, 0.0)), 500)
        @test all(maximum(F; dims=2) .- minimum(F; dims=2) .<= 1e-12)
        @test all(l .<= F[:, 1] .<= u)
        @test maximum(F[:, 1]) - minimum(F[:, 1]) > 0.9 * (u - l)
    end

    @testset "availability stays within rated" begin
        F = der_fractions(DERScaler(network, der_options(0.0, 1.0, DER_SIGMA)), 5000)
        @test all(0.0 .<= F)
        @test all(F .<= 1.0 + 1e-12)
        @test any(F .>= 1.0 - 1e-12)  # the cap is actually reached
    end

    @testset "moments match theory" begin
        # Unclipped: u(1+σ) < 1, so Corr(ε_i, ε_j) = Var(α) / (E[α²](1+σ²/3) - E[α]²) = 0.75.
        n = 40_000
        l, u, σ = 0.0, 0.5, DER_SIGMA
        F = der_fractions(DERScaler(network, der_options(l, u, σ)), n)
        @test within_mc_error(F[:, 1], (l + u) / 2)
        ρ = cor(F[:, 1], F[:, 2])
        @test abs(ρ - 0.75) <= 4 * (1 - 0.75^2) / sqrt(n)

        # Clipped at α = 1: E[min(η, 1)] = 1 - σ/4.
        F = der_fractions(DERScaler(network, der_options(1.0, 1.0, σ)), n)
        @test within_mc_error(F[:, 1], 1 - σ / 4)
    end

    @testset "rated is a snapshot" begin
        ds = DERScaler(network, der_options(1.0, 1.0, 0.0))
        _set_DERs!(network, 0.25 .* rated)
        try
            @test ds.rated == rated
            @test Random.rand(MersenneTwister(1), ds) == rated
        finally
            _set_DERs!(network, rated)
        end
    end

    @testset "option validation" begin
        @test_throws ErrorException DERScaler(network, der_options(0.0, 1.0, -0.1))
        @test_throws ErrorException DERScaler(network, der_options(0.0, 1.0, 1.1))
        @test_throws ErrorException DERScaler(network, der_options(0.0, 1.0, NaN))
        @test_throws ErrorException DERScaler(network, der_options(0.5, 0.2, 0.1))
        @test_throws ErrorException DERScaler(network, der_options(0.0, 1.1, 0.1))
        @test DERScaler(network).d.d_η == DERScaler(network, der_options(0.0, 1.0, DEFAULT_DER_SIGMA)).d.d_η
    end

    @testset "feeder without DER units" begin
        @test Random.rand(MersenneTwister(1), DERScaler(nothing, Float64[])) == Float64[]
    end
end

@testset "regime handling keeps DER sigma" begin
    regimes = [
        Dict{String, Any}(
            "name" => "base",
            "weight" => 1.0,
            "load" => Dict{String, Any}("l" => 0.2, "u" => 1.5),
            "der" => Dict{String, Any}("frac_l" => 0.0, "frac_u" => 1.0, "sigma" => 0.25),
        ),
    ]
    validate_regimes!(regimes, Dict("noise_type" => "ScaledUniform", "sigma" => LOAD_SIGMA))
    @test regimes[1]["der"]["sigma"] == 0.25
    apply_regime_cli_overrides!(regimes, Dict("der_frac_u" => "0.8"))
    @test regimes[1]["der"]["sigma"] == 0.25
    @test regimes[1]["der"]["frac_u"] == 0.8
    @test regimes[1]["load"]["sigma"] == LOAD_SIGMA

    missing_sigma = [
        Dict{String, Any}("name" => "base", "weight" => 1.0, "load" => Dict{String, Any}("l" => 0.2, "u" => 1.5)),
    ]
    validate_regimes!(missing_sigma, Dict("noise_type" => "ScaledUniform", "sigma" => LOAD_SIGMA))
    @test missing_sigma[1]["der"]["sigma"] == DEFAULT_DER_SIGMA

    bad_sigma = [
        Dict{String, Any}(
            "name" => "base",
            "weight" => 1.0,
            "load" => Dict{String, Any}("l" => 0.2, "u" => 1.5),
            "der" => Dict{String, Any}("sigma" => 2.0),
        ),
    ]
    @test_throws ErrorException validate_regimes!(bad_sigma, Dict("noise_type" => "ScaledUniform", "sigma" => LOAD_SIGMA))
end

@testset "shift clamps" begin
    shift_cfg(mu, sigma; noise_type="gaussian") = Dict{String, Any}(
        "enabled" => true, "noise_type" => noise_type,
        "mu_rel" => mu, "sigma_rel" => sigma, "der_mu" => mu, "der_sigma" => sigma,
        "scale_factor" => 1.0, "random_sign" => true,
    )
    pd_ref = [0.2, 0.0, 0.05, 0.1]
    qd_ref = [0.1, 0.0, -0.04, 0.03]
    der_rated = [0.02, 0.005, 0.01]
    cap = shift_load_cap_ratio(Dict("noise_type" => "ScaledUniform", "u" => 1.5, "sigma" => LOAD_SIGMA))
    @test cap ≈ 1.95
    @test_throws ErrorException shift_load_cap_ratio(Dict("noise_type" => "ScaledLogNormal", "u" => 1.5, "sigma" => 0.1))
    refs = (; pd_ref, qd_ref, der_rated, load_cap_ratio=cap)

    @testset "zero shift leaves extreme base samples unchanged" begin
        pd = cap .* pd_ref
        qd = cap .* qd_ref
        pder = copy(der_rated)
        apply_shift!(pd, qd, pder, 1, shift_cfg(0.0, 0.0); refs...)
        @test pd == cap .* pd_ref
        @test qd == cap .* qd_ref
        @test pder == der_rated
    end

    @testset "disabled shift is a no-op" begin
        pd, qd, pder = 9 .* pd_ref, 9 .* qd_ref, 9 .* der_rated
        apply_shift!(pd, qd, pder, 1, Dict{String, Any}("enabled" => false); refs...)
        @test pd == 9 .* pd_ref && qd == 9 .* qd_ref && pder == 9 .* der_rated
    end

    @testset "large shifts respect the caps" begin
        hit_upper = false
        hit_lower = false
        for noise_type in ("gaussian", "additive"), seed in 1:500
            pd, qd, pder = copy(pd_ref), copy(qd_ref), copy(der_rated)
            apply_shift!(pd, qd, pder, seed, shift_cfg(1.0, 1.0; noise_type); refs...)
            @test all(0.0 .<= pd .<= cap .* pd_ref)
            @test all(0.0 .<= pder .<= der_rated)
            @test pd[2] == 0.0 && qd[2] == 0.0  # zero references stay zero
            @test all(0.0 .<= qd[[1, 4]] .<= cap .* qd_ref[[1, 4]])
            @test cap * qd_ref[3] <= qd[3] <= 0.0  # negative reference keeps its sign
            hit_upper |= pd[1] == cap * pd_ref[1]
            hit_lower |= pd[1] == 0.0
        end
        @test hit_upper && hit_lower  # both bounds are exercised
    end

    @testset "values inside the caps are untouched" begin
        # mu = sigma = 0.01 keeps every value strictly inside the caps unless |Z| is enormous.
        pd, qd, pder = copy(pd_ref), copy(qd_ref), 0.5 .* der_rated
        apply_shift!(pd, qd, pder, 7, shift_cfg(0.01, 0.01); refs...)
        rng = MersenneTwister(7 + 1000)
        @test pd == pd_ref .* (1 .+ _relative_shift_noise(rng, 4, 0.01, 0.01, true))
        @test qd == qd_ref .* (1 .+ _relative_shift_noise(rng, 4, 0.01, 0.01, true))
        @test pder == 0.5 .* der_rated .* (1 .+ _relative_shift_noise(rng, 3, 0.01, 0.01, true))
    end
end

@testset "configs and code defaults use the agreed sigmas" begin
    paths = filter(p -> endswith(p, ".json"), readdir(CONFIG_DIR; join=true))
    @test length(paths) == 15
    for path in paths
        config = load_json_config(path)
        if haskey(config, "training_regimes")
            @test config["regime_load_defaults"]["sigma"] == LOAD_SIGMA
            for regime in config["training_regimes"]
                @test !haskey(regime["load"], "sigma")
                @test regime["der"]["sigma"] == DER_SIGMA
            end
        else
            @test config["load"]["sigma"] == LOAD_SIGMA
            @test config["der"]["sigma"] == DER_SIGMA
        end
    end
    @test REGIME_LOAD_DEFAULTS["sigma"] == LOAD_SIGMA
    @test all(r["der"]["sigma"] == DER_SIGMA for r in TRAINING_REGIMES)
    @test DEFAULT_TESTING_CONFIG["load"]["sigma"] == LOAD_SIGMA
    @test DEFAULT_TESTING_CONFIG["der"]["sigma"] == DER_SIGMA
    @test DEFAULT_DER_SIGMA == DER_SIGMA
end
