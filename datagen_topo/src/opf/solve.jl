"""
    OPFSolveCache

JuMP model built once per fixed topology (`branch_status_key` snapshot).

Sample-varying loads, DER availability, objective weights and soft-limit penalty weights
(`rho = [ρ_V, ρ_I, ρ_θ]`) are JuMP `Parameter`s
updated via [`sync_params!`](@ref) before each [`solve_cached_opf!`](@ref). Voltage warm start
uses `u_start` from a linear solve or an explicit override at build time.
"""
mutable struct OPFSolveCache
    model::JuMP.Model
    data::OPFData
    branch_status_key::Vector{Bool}
    p_load_param::Vector{JuMP.VariableRef}
    q_load_param::Vector{JuMP.VariableRef}
    der_pavail_param::Vector{JuMP.VariableRef}
    w_cost::JuMP.VariableRef
    w_curtail::JuMP.VariableRef
    rho::Vector{JuMP.VariableRef}
    vr::Vector{JuMP.VariableRef}
    vi::Vector{JuMP.VariableRef}
    u_start::Vector{ComplexF64}
    build_time::Float64
    report_build_time::Bool
end

"""Require `data.branch_status` unchanged since the cache was built (topology-fixed cache)."""
function assert_topology_matches!(cache::OPFSolveCache, data::OPFData)
    if data.branch_status != cache.branch_status_key
        throw(ArgumentError(
            "branch_status changed since OPFSolveCache was built; create a new cache for the new topology.",
        ))
    end
    return nothing
end

"""
    sync_params!(cache::OPFSolveCache, data::OPFData)

Push sample-varying JuMP parameters: ``p_load``, ``q_load``, per-unit DER availability, and
objective weights ``[w_cost, w_curtail, ρ_V, ρ_I, ρ_θ]``. Topology must match the cache snapshot.
"""
function sync_params!(cache::OPFSolveCache, data::OPFData)
    assert_topology_matches!(cache, data)
    n_net = data.n_net
    length(data.p_load) == n_net || throw(DimensionMismatch("p_load length must equal n_net"))
    length(data.q_load) == n_net || throw(DimensionMismatch("q_load length must equal n_net"))
    length(data.der_pavail_ref) == data.n_der_unit || throw(DimensionMismatch("der_pavail_ref length must equal n_der_unit"))
    length(data.obj_weights) == N_OBJ_WEIGHTS || throw(DimensionMismatch("obj_weights must have length $N_OBJ_WEIGHTS"))
    all(data.obj_weights .>= 0.0) || throw(ArgumentError("obj_weights must be nonnegative; got $(data.obj_weights)"))

    for i in 1:n_net
        set_parameter_value(cache.p_load_param[i], data.p_load[i])
        set_parameter_value(cache.q_load_param[i], data.q_load[i])
    end
    for g in 1:data.n_der_unit
        set_parameter_value(cache.der_pavail_param[g], data.der_pavail_ref[g])
    end
    set_parameter_value(cache.w_cost, data.obj_weights[1])
    set_parameter_value(cache.w_curtail, data.obj_weights[2])
    for (k, rho_k) in enumerate(cache.rho)
        set_parameter_value(rho_k, data.obj_weights[N_ECON_WEIGHTS + k])
    end
    return nothing
end

"""Set JuMP start values for ``vr``, ``vi`` from ``cache.u_start`` (real/imag parts)."""
function set_voltage_warm_start!(cache::OPFSolveCache)
    set_start_value.(cache.vr, real.(cache.u_start))
    set_start_value.(cache.vi, imag.(cache.u_start))
    return nothing
end

"""
    build_opf_cache(data, optimizer; warm_start, v_slack_local_override)

Build a reusable [`OPFSolveCache`](@ref): JuMP model, parameter handles, and initial voltage warm start.
Records `build_time` once; subsequent solves use [`solve_cached_opf!`](@ref).
"""
function build_opf_cache(
    data::OPFData,
    optimizer;
    warm_start::Union{Nothing, AbstractVector{<:Complex}}=nothing,
    v_slack_local_override::Union{Nothing, AbstractVector{<:Complex}}=nothing,
)
    tbuild = @elapsed cache = _build_opf_cache(
        data,
        optimizer;
        warm_start=warm_start,
        v_slack_local_override=v_slack_local_override,
    )
    cache.build_time = tbuild
    cache.report_build_time = true
    sync_params!(cache, data)
    set_voltage_warm_start!(cache)
    set_silent(cache.model)
    return cache
end

"""Return only the JuMP `Model` from [`build_opf_cache`](@ref) (same build, no cache wrapper)."""
function build_opf_model(
    data::OPFData,
    optimizer;
    warm_start::Union{Nothing, AbstractVector{<:Complex}}=nothing,
    v_slack_local_override::Union{Nothing, AbstractVector{<:Complex}}=nothing,
)
    return build_opf_cache(
        data,
        optimizer;
        warm_start=warm_start,
        v_slack_local_override=v_slack_local_override,
    ).model
end

"""Extract primal decision variables for HDF5 / downstream Python (`vr`, `vi`, `pder`, `qder`, substation)."""
function extract_primal(model::JuMP.Model, data::OPFData)
    return Dict{String, Any}(
        "vr" => JuMP.value.(model[:vr]),
        "vi" => JuMP.value.(model[:vi]),
        "psub" => JuMP.value(model[:psub]),
        "qsub" => JuMP.value(model[:qsub]),
        "psub_phase" => JuMP.value.(model[:p_sub_k]),
        "qsub_phase" => JuMP.value.(model[:q_sub_k]),
        "pder" => JuMP.value.(model[:pder]),
        "qder" => JuMP.value.(model[:qder]),
    )
end

"""Solver status, timings, and objective values (NaN when primal/dual not available)."""
function extract_metadata(model::JuMP.Model)
    tst = termination_status(model)
    pst = primal_status(model)
    dst = dual_status(model)

    ok_statuses = [MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT, MOI.INFEASIBLE_POINT]

    z_primal = pst in ok_statuses ? objective_value(model) : NaN
    z_dual = dst in ok_statuses ? dual_objective_value(model) : NaN

    return Dict{String, Any}(
        "termination_status" => string(tst),
        "primal_status" => string(pst),
        "dual_status" => string(dst),
        "solve_time" => solve_time(model),
        "primal_objective_value" => z_primal,
        "dual_objective_value" => z_dual,
    )
end

const OPF_FEASIBLE_TERMINATION_STATUSES = (
    "LOCALLY_SOLVED",
    "ALMOST_LOCALLY_SOLVED",
)

const OPF_INFEASIBLE_TERMINATION_STATUSES = (
    "LOCALLY_INFEASIBLE",
    "INFEASIBLE",
    "INFEASIBLE_OR_UNBOUNDED",
)

"""True when Ipopt (or another solver) reports a feasible local solve."""
function is_opf_feasible(meta::AbstractDict)
    return meta["termination_status"] in OPF_FEASIBLE_TERMINATION_STATUSES
end

"""
True when the solver reports infeasibility via `termination_status`.

Ipopt sets `termination_status = LOCALLY_INFEASIBLE` with `primal_status = INFEASIBLE_POINT`;
do not key off `primal_status == LOCALLY_INFEASIBLE` (that string is not a MOI result code).
"""
function is_opf_infeasible(meta::AbstractDict)
    return meta["termination_status"] in OPF_INFEASIBLE_TERMINATION_STATUSES
end

"""Combined result dict with keys `"meta"` and `"primal"` for one OPF solve."""
function extract_result(model::JuMP.Model, data::OPFData)
    return Dict{String, Any}(
        "meta" => extract_metadata(model),
        "primal" => extract_primal(model, data),
    )
end

"""
    solve_cached_opf!(cache, data)

Sync parameters, warm-start voltages, optimize, and extract result. Adds `build_time` (first
solve only), `update_time`, and `extract_time` to `"meta"`.
"""
function solve_cached_opf!(cache::OPFSolveCache, data::OPFData)
    tupdate = @elapsed begin
        sync_params!(cache, data)
        set_voltage_warm_start!(cache)
    end
    JuMP.optimize!(cache.model)
    textract = @elapsed result = extract_result(cache.model, data)
    result["meta"]["build_time"] = cache.report_build_time ? cache.build_time : 0.0
    cache.report_build_time = false
    result["meta"]["update_time"] = tupdate
    result["meta"]["extract_time"] = textract
    return result
end

"""Solve using the `data` snapshot stored when the cache was built."""
function solve_cached_opf!(cache::OPFSolveCache)
    return solve_cached_opf!(cache, cache.data)
end

"""One-shot: [`build_opf_cache`](@ref) then [`solve_cached_opf!`](@ref); no cache reuse."""
function build_and_solve_opf(
    data::OPFData,
    optimizer;
    warm_start::Union{Nothing, AbstractVector{<:Complex}}=nothing,
    v_slack_local_override::Union{Nothing, AbstractVector{<:Complex}}=nothing,
)
    cache = build_opf_cache(
        data,
        optimizer;
        warm_start=warm_start,
        v_slack_local_override=v_slack_local_override,
    )
    return solve_cached_opf!(cache, data)
end
