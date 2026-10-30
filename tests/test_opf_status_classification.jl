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

@testset "Soft OPF: solver infeasibility still detected when hard limits cannot be met" begin
    # All network limits (V / I / θ / substation) are soft, so infeasibility now comes only from power balance
    # (loadability) or from a limit made hard. Restore the hard per-phase substation limit |S_sub,k| <= smax_k
    # by fixing s_sub = 0: with no DER the problem is then provably infeasible once total active load exceeds
    # Σ_k smax_k, since Σ_k P_k <= Σ_k |S_k| <= Σ_k smax_k and losses only add to demand.
    case_path = joinpath(DATAGEN_DIR, "grid_data", "13_bus", "IEEE13Nodeckt.dss")
    network = create_network_data(case_path)
    ls = LoadScaler(network, Dict(
        "noise_type" => "ScaledUniform",
        "l" => 0.7,
        "u" => 1.0,
        "sigma" => 0.10,
    ))
    os = ObjScaler(network, Dict("wmin" => [0, 0], "wmax" => [1, 10]))

    warm_start = compute_warm_start(network)
    cache = build_opf_cache(network, Ipopt.Optimizer; warm_start=warm_start)

    rng = MersenneTwister(1461)
    pd, qd = Random.rand(rng, ls)
    _set_DERs!(network, zeros(network.n_der_unit))
    _set_obj_weights!(network, Random.rand(rng, os))

    # Feasible reference: the sampled load is served.
    _set_loads!(network, pd, qd)
    res = solve_cached_opf!(cache, network)
    @test is_opf_feasible(res["meta"])

    # Scale load so total P demand is twice the substation capability, with the substation limit hard.
    k = 2.0 * sum(network.sub_smax_phase) / sum(pd)
    @test k > 1.0
    _set_loads!(network, k .* pd, k .* qd)
    fix.(cache.model[:s_sub], 0.0; force=true)
    res = solve_cached_opf!(cache, network)
    @test is_opf_infeasible(res["meta"])
    @test res["meta"]["termination_status"] == "LOCALLY_INFEASIBLE"
    @test res["meta"]["primal_status"] == "INFEASIBLE_POINT"
end
