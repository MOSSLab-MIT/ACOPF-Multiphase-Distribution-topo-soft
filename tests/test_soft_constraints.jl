# Soft V / I / θ / substation limits in the AC-OPF: penalty-weight sampling, exact-penalty recovery, KKT,
# complementarity.
# Run: julia --project=datagen_topo/FeederFlow.jl tests/test_soft_constraints.jl
using Test
using Random
using Statistics
using Ipopt
using JuMP

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const DATAGEN_DIR = joinpath(ROOT_DIR, "datagen_topo")

include(joinpath(DATAGEN_DIR, "datagen_topo.jl"))

const CASE_13 = joinpath(DATAGEN_DIR, "grid_data", "13_bus", "IEEE13Nodeckt.dss")

# Slack variables paired with the soft constraints they relax (row-aligned containers).
const SOFT_PAIRS = (
    (:s_vmin, :c_vmin),
    (:s_vmax, :c_vmax),
    (:s_imax, :c_imax),
    (:s_ang_pos, :c_ang_pos),
    (:s_ang_neg, :c_ang_neg),
    (:s_sub, :c_sub),
)
# Penalty weight index (into obj_weights) of each slack family.
const SLACK_RHO_INDEX = Dict(
    :s_vmin => 3, :s_vmax => 3, :s_imax => 4, :s_ang_pos => 5, :s_ang_neg => 5, :s_sub => 6,
)

# Ipopt contract used by model.jl: primal constraint violation <= constr_viol_tol.
const CONSTR_VIOL_TOL = 1e-6
# Optimal objectives / duals agree to Ipopt's overall tol (1e-8) with margin for conditioning.
const OBJ_RTOL = 1e-6
const DUAL_RTOL = 1e-6
# Slacks above this are "clearly active" (well above CONSTR_VIOL_TOL) for the KKT dual check.
const ACTIVE_SLACK = 1e-4

function _profile_samplers(network, load_lu, der_lu)
    ls = LoadScaler(network, Dict("noise_type" => "ScaledUniform", "sigma" => 0.10, "l" => load_lu[1], "u" => load_lu[2]))
    ds = DERScaler(network, Dict("frac_l" => der_lu[1], "frac_u" => der_lu[2]))
    os = ObjScaler(network, Dict("wmin" => [0, 0], "wmax" => [1, 10]))
    return ls, ds, os
end

function _set_profile!(network, ls, ds, os, seed)
    rng = MersenneTwister(seed)
    pd, qd = Random.rand(rng, ls)
    _set_loads!(network, pd, qd)
    _set_DERs!(network, Random.rand(rng, ds))
    _set_obj_weights!(network, Random.rand(rng, os))
    return nothing
end

"""Solve the HARD OPF on the same cache by fixing every soft slack to 0."""
function _solve_hard!(cache, network)
    for (s, _) in SOFT_PAIRS, x in cache.model[s]
        fix(x, 0.0; force=true)
    end
    res = solve_cached_opf!(cache, network)
    for (s, _) in SOFT_PAIRS, x in cache.model[s]
        unfix(x)
        set_lower_bound(x, 0.0)
    end
    return res
end

"""
Hard-constraint residual r of a soft constraint (r > 0 means the hard limit is violated).
JuMP normalizes `x <= ub + s` to `x - s <= ub` and `x >= lb - s` to `x + s >= lb`.
"""
function _hard_residual(c, s_val)
    set = constraint_object(c).set
    if set isa MOI.LessThan
        return (value(c) + s_val) - set.upper
    elseif set isa MOI.GreaterThan
        return set.lower - (value(c) - s_val)
    end
    error("unexpected constraint set $(typeof(set))")
end

@testset "ObjScaler: economic weights uniform, penalty weights log-uniform" begin
    network = create_network_data(CASE_13)
    @test length(network.obj_weights) == N_OBJ_WEIGHTS == 6
    @test OBJ_WEIGHT_NAMES == ("w_cost", "w_curtail", "rho_v", "rho_i", "rho_theta", "rho_sub")

    obj_cfg = Dict("wmin" => [0.0, 0.0], "wmax" => [1.0, 10.0])
    pen_cfg = Dict("rho_min" => [100.0, 10.0, 1.0, 1.0], "rho_max" => [1000.0, 1e4, 1.0, 1e3])
    os = ObjScaler(network, obj_cfg, pen_cfg)
    rng = MersenneTwister(7)
    n = 40_000
    W = reduce(hcat, [Random.rand(rng, os) for _ in 1:n])
    @test size(W) == (6, n)
    @test all(0.0 .<= W[1, :] .<= 1.0) && all(0.0 .<= W[2, :] .<= 10.0)
    @test all(100.0 .<= W[3, :] .<= 1000.0) && all(10.0 .<= W[4, :] .<= 1e4)
    @test all(W[5, :] .== 1.0)  # degenerate range rho_min == rho_max
    @test all(1.0 .<= W[6, :] .<= 1e3)

    # Under log-uniform, (log ρ - log lo)/(log hi - log lo) ~ U(0,1): mean 1/2, var 1/12.
    # Under uniform, w/hi ~ U(0,1). 6σ bounds on the sample mean (σ = sqrt(1/12/n)).
    tol6σ = 6 * sqrt(1 / 12 / n)
    u_rho_v = (log.(W[3, :]) .- log(100.0)) ./ (log(1000.0) - log(100.0))
    u_rho_i = (log.(W[4, :]) .- log(10.0)) ./ (log(1e4) - log(10.0))
    @test abs(mean(u_rho_v) - 0.5) < tol6σ
    @test abs(mean(u_rho_i) - 0.5) < tol6σ
    u_rho_sub = log.(W[6, :]) ./ log(1e3)
    @test abs(mean(u_rho_sub) - 0.5) < tol6σ
    @test abs(mean(W[2, :] ./ 10.0) - 0.5) < tol6σ
    # Log-uniform is NOT uniform: the linear mean of ρ_V sits well below the midpoint 550.
    # E[ρ] = (hi - lo)/ln(hi/lo) = 900/ln(10) ≈ 390.9.
    @test abs(mean(W[3, :]) - 900 / log(10.0)) < 6 * std(W[3, :]) / sqrt(n)

    # Default penalty block: [100, 1000] for ρ_V, ρ_I, ρ_θ and [100, 200] for ρ_sub.
    os_default = ObjScaler(network, obj_cfg)
    @test os_default.wmin[3:6] == [100.0, 100.0, 100.0, 100.0]
    @test os_default.wmax[3:6] == [1000.0, 1000.0, 1000.0, 200.0]
    @test os_default.log_scale == [false, false, true, true, true, true]

    # Validation (a legacy 3-entry penalty block is rejected, not padded).
    @test_throws ErrorException ObjScaler(network, obj_cfg, Dict("rho_min" => [0.0, 1.0, 1.0, 1.0], "rho_max" => [1.0, 1.0, 1.0, 1.0]))
    @test_throws ErrorException ObjScaler(network, obj_cfg, Dict("rho_min" => [1.0, 1.0, 1.0, 2.0], "rho_max" => [1.0, 1.0, 1.0, 1.0]))
    @test_throws ErrorException ObjScaler(network, obj_cfg, Dict("rho_min" => [1.0, 1.0, 1.0], "rho_max" => [1.0, 1.0, 1.0]))
    @test_throws ErrorException ObjScaler(network, Dict("wmin" => [0.0], "wmax" => [1.0]), pen_cfg)
end

@testset "Soft OPF on IEEE13: exact penalty, complementarity, KKT, objective decomposition" begin
    network = create_network_data(CASE_13)
    cache = build_opf_cache(network, Ipopt.Optimizer; warm_start=compute_warm_start(network))
    model = cache.model
    n_rows = sum(count(v) for v in network.branch_valid)
    @test length(model[:s_imax]) == length(model[:c_imax]) == n_rows
    @test length(model[:s_vmin]) == network.n_net
    @test length(model[:s_sub]) == length(model[:c_sub]) == network.n_slk
    # Soft substation limit: no residual ±smax boxes on the per-phase substation powers.
    @test !any(has_lower_bound.(model[:p_sub_k])) && !any(has_upper_bound.(model[:p_sub_k]))
    @test !any(has_lower_bound.(model[:q_sub_k])) && !any(has_upper_bound.(model[:q_sub_k]))

    n_hard_feasible = 0
    n_hard_infeasible = 0
    n_kkt_checked = 0
    regimes = (
        ((0.2, 1.5), (0.0, 1.0), 1:8),        # base: mostly hard-feasible
        ((0.7, 1.0), (0.0, 0.3), 1461:1466),  # high load / low DER: mostly hard-infeasible
    )
    for (load_lu, der_lu, seeds) in regimes
        ls, ds, os = _profile_samplers(network, load_lu, der_lu)
        for seed in seeds
            _set_profile!(network, ls, ds, os, seed)
            w = copy(network.obj_weights)
            @test all(100.0 .<= w[3:5] .<= 1000.0)
            @test 100.0 <= w[6] <= 200.0

            hard = _solve_hard!(cache, network)
            soft = solve_cached_opf!(cache, network)

            # Soft model must always solve (power balance + DER/sub limits are satisfiable here).
            @test is_opf_feasible(soft["meta"])

            # Complementarity at the optimum: s = max(0, r) for every soft constraint, and s >= 0 exactly
            # (bound_relax_factor = 0).
            for (sname, cname) in SOFT_PAIRS
                svals = value.(model[sname])
                @test all(svals .>= 0.0)
                r = [_hard_residual(c, sv) for (c, sv) in zip(model[cname], svals)]
                @test maximum(abs.(svals .- max.(r, 0.0)); init=0.0) <= CONSTR_VIOL_TOL
            end

            # Objective decomposition: Ipopt objective == w·economic + Σ ρ_k * Σ s_k.
            econ = w[1] * value(model[:obj_cost]) + w[2] * value(model[:obj_curtailment])
            pen = w[3] * value(model[:pen_v]) + w[4] * value(model[:pen_i]) + w[5] * value(model[:pen_theta]) +
                w[6] * value(model[:pen_sub])
            z_soft = soft["meta"]["primal_objective_value"]
            @test isapprox(z_soft, econ + pen; rtol=1e-12, atol=1e-12)

            # KKT stationarity in s: ρ - λ - z_s = 0 with z_s = 0 when s > 0  =>  |λ| = ρ on active slacks.
            for (sname, cname) in SOFT_PAIRS
                rho = w[SLACK_RHO_INDEX[sname]]
                for (x, c) in zip(model[sname], model[cname])
                    value(x) > ACTIVE_SLACK || continue
                    @test isapprox(abs(dual(c)), rho; rtol=DUAL_RTOL)
                    n_kkt_checked += 1
                end
            end

            if is_opf_feasible(hard["meta"])
                n_hard_feasible += 1
                # Exact penalty: ρ ∈ [100, 1000] exceeds the hard multipliers here, so the soft optimum is the
                # hard optimum (zero slack) — verified by objective equality and zero total violation.
                total_slack = sum(sum(value.(model[s]); init=0.0) for (s, _) in SOFT_PAIRS)
                @test total_slack <= CONSTR_VIOL_TOL
                @test isapprox(z_soft, hard["meta"]["primal_objective_value"]; rtol=OBJ_RTOL)
            else
                n_hard_infeasible += 1
                # The hard problem is infeasible, so some hard limit must be violated in the soft optimum.
                total_slack = sum(sum(value.(model[s]); init=0.0) for (s, _) in SOFT_PAIRS)
                @test total_slack > ACTIVE_SLACK
            end
        end
    end
    # Both branches above must actually be exercised.
    @test n_hard_feasible >= 3
    @test n_hard_infeasible >= 3
    @test n_kkt_checked > 0
    println("KKT dual checks on active slacks: $n_kkt_checked")
end

@testset "Soft OPF on IEEE13: small ρ trades violation for cost" begin
    # On a hard-infeasible sample, lowering ρ can only lower the optimal objective and can only raise the
    # weighted violation it accepts (monotonicity of parametric optima in ρ, for the same local branch).
    network = create_network_data(CASE_13)
    cache = build_opf_cache(network, Ipopt.Optimizer; warm_start=compute_warm_start(network))
    ls, ds, os = _profile_samplers(network, (0.7, 1.0), (0.0, 0.3))
    _set_profile!(network, ls, ds, os, 1461)
    @test !is_opf_feasible(_solve_hard!(cache, network)["meta"])

    w_econ = network.obj_weights[1:2]
    z_prev = Inf
    viol_prev = -Inf
    for rho in (1e3, 1e2, 1.0, 1e-2)
        network.obj_weights .= vcat(w_econ, fill(rho, 4))
        res = solve_cached_opf!(cache, network)
        @test is_opf_feasible(res["meta"])
        z = res["meta"]["primal_objective_value"]
        viol = value(cache.model[:pen_v]) + value(cache.model[:pen_i]) + value(cache.model[:pen_theta]) +
            value(cache.model[:pen_sub])
        @test z <= z_prev * (1 + OBJ_RTOL)
        @test viol >= viol_prev - CONSTR_VIOL_TOL
        z_prev, viol_prev = z, viol
    end
end

@testset "Soft OPF on IEEE13: substation overload is priced (complementarity, KKT, monotone in ρ_sub)" begin
    # No DER and ΣP_load = 0.9·Σsmax: the per-phase substation limit binds on the loaded phases.
    # The OPF must still solve, with s_sub = relu(|S_k|² - smax_k²) and |λ_sub,k| = ρ_sub on active phases.
    network = create_network_data(CASE_13)
    cache = build_opf_cache(network, Ipopt.Optimizer; warm_start=compute_warm_start(network))
    model = cache.model
    ls, _, os = _profile_samplers(network, (0.7, 1.0), (0.0, 1.0))
    rng = MersenneTwister(1461)
    pd, qd = Random.rand(rng, ls)
    _set_DERs!(network, zeros(network.n_der_unit))
    _set_obj_weights!(network, Random.rand(rng, os))
    k = 0.9 * sum(network.sub_smax_phase) / sum(pd)
    _set_loads!(network, k .* pd, k .* qd)
    smax2 = network.sub_smax_phase .^ 2

    pen_prev = Inf
    z_prev = -Inf
    n_active = 0
    # ρ_sub spans the sampled range [100, 200] plus a larger value. Ipopt's stationarity accuracy is absolute
    # at the scale of the largest objective coefficients (tol·max ρ ≈ 1e-8·ρ_V ~ 1e-5 here), so |λ| = ρ_sub
    # at DUAL_RTOL is only resolvable for ρ_sub comparable to the other weights (measured in
    # experiments/soft_substation/probe_kkt_sub.jl).
    for rho_sub in (100.0, 150.0, 200.0, 1000.0)
        network.obj_weights[6] = rho_sub
        res = solve_cached_opf!(cache, network)
        @test is_opf_feasible(res["meta"])
        s_sub = value.(model[:s_sub])
        smag2 = value.(model[:p_sub_k]) .^ 2 .+ value.(model[:q_sub_k]) .^ 2
        @test all(s_sub .>= 0.0)
        @test maximum(abs.(s_sub .- max.(smag2 .- smax2, 0.0))) <= CONSTR_VIOL_TOL
        @test value(model[:pen_sub]) ≈ sum(s_sub) rtol = 1e-12 atol = 1e-12
        for (s, c) in zip(model[:s_sub], model[:c_sub])
            value(s) > ACTIVE_SLACK || continue
            @test isapprox(abs(dual(c)), rho_sub; rtol=DUAL_RTOL)
            n_active += 1
        end
        # Raising ρ_sub can only lower the accepted substation violation and raise the optimal objective.
        pen = value(model[:pen_sub])
        z = res["meta"]["primal_objective_value"]
        @test pen <= pen_prev + CONSTR_VIOL_TOL
        @test z >= z_prev * (1 - OBJ_RTOL)
        pen_prev, z_prev = pen, z
    end
    # The substation slack must actually be exercised (otherwise this test proves nothing).
    @test n_active > 0
    println("Active substation-slack KKT checks: $n_active")
end
