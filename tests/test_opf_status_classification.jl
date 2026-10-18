using Test
using Random
using Ipopt

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const DATAGEN_DIR = joinpath(ROOT_DIR, "datagen_topo")

include(joinpath(DATAGEN_DIR, "datagen_topo.jl"))

@testset "OPF status classification" begin
    @test is_opf_feasible(Dict("termination_status" => "LOCALLY_SOLVED"))
    @test is_opf_feasible(Dict("termination_status" => "ALMOST_LOCALLY_SOLVED"))
    @test !is_opf_feasible(Dict("termination_status" => "LOCALLY_INFEASIBLE"))

    @test is_opf_infeasible(Dict("termination_status" => "LOCALLY_INFEASIBLE"))
    @test is_opf_infeasible(Dict("termination_status" => "INFEASIBLE"))
    @test is_opf_infeasible(Dict("termination_status" => "INFEASIBLE_OR_UNBOUNDED"))
    @test !is_opf_infeasible(Dict("termination_status" => "LOCALLY_SOLVED"))

    # Ipopt infeasible solves use termination LOCALLY_INFEASIBLE and primal INFEASIBLE_POINT.
    ipopt_infeasible_meta = Dict(
        "termination_status" => "LOCALLY_INFEASIBLE",
        "primal_status" => "INFEASIBLE_POINT",
    )
    @test is_opf_infeasible(ipopt_infeasible_meta)
    @test ipopt_infeasible_meta["primal_status"] != "LOCALLY_INFEASIBLE"
end

@testset "OPF infeasible samples are detectable under high-load low-DER regime" begin
    case_path = joinpath(DATAGEN_DIR, "grid_data", "13_bus", "IEEE13Nodeckt.dss")
    network = create_network_data(case_path)
    ls = LoadScaler(network, Dict(
        "noise_type" => "ScaledUniform",
        "l" => 0.7,
        "u" => 1.0,
        "sigma" => 0.10,
    ))
    ds = DERScaler(network, Dict("frac_l" => 0.0, "frac_u" => 0.3))
    os = ObjScaler(network, Dict("wmin" => [0, 0], "wmax" => [1, 10]))

    warm_start = compute_warm_start(network)
    cache = build_opf_cache(network, Ipopt.Optimizer; warm_start=warm_start)

    found_infeasible = false
    for seed in 1461:1475
        rng = MersenneTwister(seed)
        pd, qd = Random.rand(rng, ls)
        pder_avai = Random.rand(rng, ds)
        obj_weights = Random.rand(rng, os)
        _set_loads!(network, pd, qd)
        _set_DERs!(network, pder_avai)
        _set_obj_weights!(network, obj_weights)

        res = solve_cached_opf!(cache, network)
        if is_opf_infeasible(res["meta"])
            found_infeasible = true
            @test res["meta"]["termination_status"] == "LOCALLY_INFEASIBLE"
            @test res["meta"]["primal_status"] == "INFEASIBLE_POINT"
            break
        end
    end

    @test found_infeasible
end
