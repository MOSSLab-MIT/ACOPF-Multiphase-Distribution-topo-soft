# Substation / DER price lives in the JSON configs (`cost` block) and reaches the OPF objective.
# Run: julia --project=datagen_topo/FeederFlow.jl tests/test_config_cost.jl
using Test
using Random
using Ipopt
using JuMP

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const DATAGEN_DIR = joinpath(ROOT_DIR, "datagen_topo")

include(joinpath(DATAGEN_DIR, "datagen_topo.jl"))
include(joinpath(DATAGEN_DIR, "src", "generation", "training.jl"))
include(joinpath(DATAGEN_DIR, "src", "generation", "testing.jl"))

const CONFIG_DIR = joinpath(DATAGEN_DIR, "gen_scripts", "configs")
const CASE_123 = joinpath(DATAGEN_DIR, "grid_data", "123_bus", "IEEE123Master.dss")

@testset "code defaults equal the FeederFlow constants" begin
    for default in (DEFAULT_TRAINING_CONFIG, DEFAULT_TESTING_CONFIG)
        cost = require_cost_block(default)
        @test cost["sub_cost"] == FeederFlow.DEFAULT_SOURCE_COST_COEFF
        @test cost["der_cost"] == FeederFlow.DEFAULT_PV_COST_COEFF
    end
end

@testset "all configs carry one shared, valid cost block" begin
    paths = filter(p -> endswith(p, ".json"), readdir(CONFIG_DIR; join=true))
    @test !isempty(paths)
    costs = [require_cost_block(load_json_config(p)) for p in paths]
    for (p, c) in zip(paths, costs)
        @test c == costs[1] || error("cost block in $p differs from $(paths[1])")
    end
end

@testset "require_cost_block validation" begin
    ok = Dict("cost" => Dict("sub_cost" => [1.0, 100.0, 0.0], "der_cost" => [0.1, 5]))
    c = require_cost_block(ok)
    @test c["sub_cost"] == [1.0, 100.0, 0.0]
    @test c["der_cost"] == [0.1, 5.0]
    @test c["der_cost"] isa Vector{Float64}
    bad(sub, der) = Dict("cost" => Dict("sub_cost" => sub, "der_cost" => der))
    @test_throws ErrorException require_cost_block(Dict("obj" => Dict()))
    @test_throws ErrorException require_cost_block(bad([1.0], [0.1, 5.0]))               # length 1
    @test_throws ErrorException require_cost_block(bad([1.0, 2.0, 0.0, 0.0], [0.1, 5.0])) # length 4
    @test_throws ErrorException require_cost_block(bad([1.0, 100.0, 3.0], [0.1, 5.0]))   # constant ignored
    @test_throws ErrorException require_cost_block(bad([-1.0, 100.0], [0.1, 5.0]))       # nonconvex
    @test_throws ErrorException require_cost_block(bad([1.0, NaN], [0.1, 5.0]))
    @test_throws ErrorException require_cost_block(bad(1.0, [0.1, 5.0]))                 # scalar
    @test_throws ErrorException require_cost_block(bad([1.0, "a"], [0.1, 5.0]))
    @test_throws ErrorException require_cost_block(Dict("cost" => Dict("sub_cost" => [1.0, 1.0])))
    @test_throws ErrorException require_cost_block(
        Dict("cost" => Dict("sub_cost" => [1.0, 1.0], "der_cost" => [1.0, 1.0], "curtail" => [1.0])))
end

@testset "--sub_cost / --der_cost CLI flags override the config" begin
    _, flags = parse_cli_tokens(["grid.dss", "--config", "x.json", "--sub_cost", "2,200,0", "--der_cost", "0.2, 10"])
    config = deep_merge_config(DEFAULT_TESTING_CONFIG, Dict("cost" => Dict("sub_cost" => [3.0, 4.0, 0.0])))
    apply_cost_cli_flags!(config, flags)
    c = require_cost_block(config)
    @test c["sub_cost"] == [2.0, 200.0, 0.0]
    @test c["der_cost"] == [0.2, 10.0]
    # Only the flag that is given is overridden; the defaults are not mutated.
    config = deepcopy(DEFAULT_TRAINING_CONFIG)
    apply_cost_cli_flags!(config, Dict("der_cost" => "0.2,10,0"))
    c = require_cost_block(config)
    @test c["sub_cost"] == FeederFlow.DEFAULT_SOURCE_COST_COEFF
    @test c["der_cost"] == [0.2, 10.0, 0.0]
    @test DEFAULT_TRAINING_CONFIG["cost"]["der_cost"] == FeederFlow.DEFAULT_PV_COST_COEFF
    # No flags: config untouched.
    config = deepcopy(DEFAULT_TESTING_CONFIG)
    apply_cost_cli_flags!(config, Dict{String, String}())
    @test require_cost_block(config) == require_cost_block(DEFAULT_TESTING_CONFIG)
    @test_throws ErrorException apply_cost_cli_flags!(deepcopy(DEFAULT_TESTING_CONFIG), Dict("sub_cost" => "2,abc"))
    @test_throws ErrorException require_cost_block(apply_cost_cli_flags!(deepcopy(DEFAULT_TESTING_CONFIG),
                                                                        Dict("sub_cost" => "2,200,5")))
end

@testset "configured price reaches the OPF objective (123-bus)" begin
    network = create_network_data(CASE_123)
    n_der = length(network.der_cost)
    @test n_der > 0
    cost = require_cost_block(Dict("cost" => Dict("sub_cost" => [2.0, 37.0, 0.0], "der_cost" => [0.3, 11.0])))
    apply_cost_config!(network, cost)
    @test network.sub_cost == [2.0, 37.0, 0.0]
    @test length(network.der_cost) == n_der
    @test all(c == [0.3, 11.0] for c in network.der_cost)
    network.der_cost[1][1] = 99.0               # entries must not alias each other or the config
    @test network.der_cost[2] == [0.3, 11.0] && cost["der_cost"] == [0.3, 11.0]
    network.der_cost[1][1] = 0.3

    ls = LoadScaler(network, Dict("noise_type" => "ScaledUniform", "l" => 0.6, "u" => 0.61, "sigma" => 1e-3))
    ds = DERScaler(network)
    os = ObjScaler(network, Dict("wmin" => [1.0, 1.0], "wmax" => [1.0, 1.0]))
    rng = MersenneTwister(7)
    pd, qd = Random.rand(rng, ls)
    _set_loads!(network, pd, qd)
    _set_DERs!(network, Random.rand(rng, ds))
    _set_obj_weights!(network, Random.rand(rng, os))

    cache = build_opf_cache(network, Ipopt.Optimizer; warm_start=compute_warm_start(network))
    res = solve_cached_opf!(cache, network)
    @test is_opf_feasible(res["meta"])
    pder = collect(res["primal"]["pder"]); psub = res["primal"]["psub"]
    manual = sum(11.0 .* pder .+ 0.3 .* pder .^ 2) + 37.0 * psub + 2.0 * psub^2
    @test value(cache.model[:obj_cost]) ≈ manual rtol = 1e-10 atol = 1e-12
end