using Test
using Random
using LinearAlgebra
using Ipopt

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const DATAGEN_DIR = joinpath(ROOT_DIR, "datagen_topo")

include(joinpath(DATAGEN_DIR, "datagen_topo.jl"))

const CASE_PATHS = Dict(
    "IEEE13Nodeckt" => joinpath(DATAGEN_DIR, "grid_data", "13_bus", "IEEE13Nodeckt.dss"),
    "IEEE123Master" => joinpath(DATAGEN_DIR, "grid_data", "123_bus", "IEEE123Master.dss"),
)

const PARITY_RTOL = 1e-6
const PARITY_ATOL = 1e-8

function _sample_profile!(network, ls, ds, os, seed::Int)
    rng = MersenneTwister(seed)
    pd, qd = Random.rand(rng, ls)
    pder_avai = Random.rand(rng, ds)
    obj_weights = Random.rand(rng, os)
    _set_loads!(network, pd, qd)
    _set_DERs!(network, pder_avai)
    _set_obj_weights!(network, obj_weights)
    return pd, qd, pder_avai, obj_weights
end

function _primal_vectors(res)
    p = res["primal"]
    return (
        vr = collect(p["vr"]),
        vi = collect(p["vi"]),
        pder = collect(p["pder"]),
        qder = collect(p["qder"]),
        psub = p["psub"],
        qsub = p["qsub"],
        psub_phase = collect(p["psub_phase"]),
        qsub_phase = collect(p["qsub_phase"]),
    )
end

function _compare_primals(a, b)
    @test a.vr ≈ b.vr rtol = PARITY_RTOL atol = PARITY_ATOL
    @test a.vi ≈ b.vi rtol = PARITY_RTOL atol = PARITY_ATOL
    @test a.pder ≈ b.pder rtol = PARITY_RTOL atol = PARITY_ATOL
    @test a.qder ≈ b.qder rtol = PARITY_RTOL atol = PARITY_ATOL
    @test a.psub ≈ b.psub rtol = PARITY_RTOL atol = PARITY_ATOL
    @test a.qsub ≈ b.qsub rtol = PARITY_RTOL atol = PARITY_ATOL
    @test a.psub_phase ≈ b.psub_phase rtol = PARITY_RTOL atol = PARITY_ATOL
    @test a.qsub_phase ≈ b.qsub_phase rtol = PARITY_RTOL atol = PARITY_ATOL
end

@testset "OPF model reuse parity" begin
    load_cfg = Dict(
        "noise_type" => "ScaledUniform",
        "l" => 0.60,
        "u" => 0.61,
        "sigma" => 1e-3,
    )
    obj_cfg = Dict(
        "wmin" => [1.0, 1.0],
        "wmax" => [1.0, 1.0],
    )
    seeds = [11, 17, 23, 29, 31]

    for (case_name, case_path) in CASE_PATHS
        @testset "Primal parity: $case_name" begin
            network = create_network_data(case_path)
            ls = LoadScaler(network, load_cfg)
            ds = DERScaler(network)
            os = ObjScaler(network, obj_cfg)

            warm_start = compute_warm_start(network)
            cache = build_opf_cache(network, Ipopt.Optimizer; warm_start=warm_start)

            for seed in seeds
                rebuild_data = create_network_data(case_path)
                _sample_profile!(rebuild_data, ls, ds, os, seed)
                res_rebuild = build_and_solve_opf(
                    rebuild_data,
                    Ipopt.Optimizer;
                    warm_start=warm_start,
                )

                _sample_profile!(network, ls, ds, os, seed)
                res_cached = solve_cached_opf!(cache, network)

                @test res_rebuild["meta"]["termination_status"] == res_cached["meta"]["termination_status"]
                @test res_rebuild["meta"]["primal_status"] == res_cached["meta"]["primal_status"]
                _compare_primals(_primal_vectors(res_rebuild), _primal_vectors(res_cached))
            end
        end
    end
end

@testset "OPF model reuse topology guard" begin
    data = create_network_data(CASE_PATHS["IEEE13Nodeckt"])
    if isempty(data.switch_idx)
        @test_skip "No switch lines on IEEE13 for topology guard test."
    else
        cache = build_opf_cache(data, Ipopt.Optimizer; warm_start=compute_warm_start(data))
        toggled = copy(data.branch_status)
        toggled[data.switch_idx[1]] = !toggled[data.switch_idx[1]]
        update_branch_status!(data, toggled)
        @test_throws ArgumentError sync_params!(cache, data)
    end
end

@testset "OPF model reuse build_time smoke (123-bus)" begin
    case_path = CASE_PATHS["IEEE123Master"]
    network = create_network_data(case_path)
    load_cfg = Dict("noise_type" => "ScaledUniform", "l" => 0.60, "u" => 0.61, "sigma" => 1e-3)
    obj_cfg = Dict("wmin" => [1.0, 1.0], "wmax" => [1.0, 1.0])
    ls = LoadScaler(network, load_cfg)
    ds = DERScaler(network)
    os = ObjScaler(network, obj_cfg)

    warm_start = compute_warm_start(network)
    cache = build_opf_cache(network, Ipopt.Optimizer; warm_start=warm_start)
    @test cache.build_time > 0.0

    _sample_profile!(network, ls, ds, os, 101)
    res1 = solve_cached_opf!(cache, network)
    @test res1["meta"]["build_time"] > 0.0

    _sample_profile!(network, ls, ds, os, 102)
    res2 = solve_cached_opf!(cache, network)
    @test res2["meta"]["build_time"] == 0.0
    @test res2["meta"]["update_time"] < 1.0
    @test res2["meta"]["solve_time"] > 0.0
end
