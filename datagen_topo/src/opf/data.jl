# Multiphase distribution AC-OPF for dataset generation (JuMP + FeederFlow).
# Topology changes are switch open/close stamps on Y_scaled; repeated solves reuse OPFSolveCache.

using JuMP
using LinearAlgebra
using Random
using FeederFlow
import HSL_jll

const MOI = JuMP.MOI

# Default voltage magnitude bounds (per-unit on each node's local base).
const VMIN_DEFAULT = 0.95
const VMAX_DEFAULT = 1.05

const VMIN_DEFAULT_LARGE    = 0.94 # relax voltage limits a little bit for large networks to avoid too many infeasible solutions
const VMAX_DEFAULT_LARGE    = 1.06

# Max |tan(θ)| between branch endpoints; tan(60°) ≈ 1.732.
const DVA_MAX_DEFAULT = tan(pi / 3)
const IMAX_DEFAULT_AMPS = 600.0 # fallback branch current rating when line amps are zero

# Objective weight layout: [w_cost, w_curtail, ρ_V, ρ_I, ρ_θ].
# ρ_* are L1 penalty weights on the soft voltage / branch-current / angle-difference slacks.
const OBJ_WEIGHT_NAMES = ("w_cost", "w_curtail", "rho_v", "rho_i", "rho_theta")
const N_OBJ_WEIGHTS = length(OBJ_WEIGHT_NAMES)
const N_ECON_WEIGHTS = 2  # leading [w_cost, w_curtail]; the remaining entries are penalty weights
const PENALTY_WEIGHT_DEFAULT = 1e3

const HSL_PATH = get(ENV, "ACOPF_HSL_PATH", HSL_jll.libhsl_path) # Override via ACOPF_HSL_PATH env var; defaults to the HSL_jll-provided library.

# Y-bus regularization for FeederFlow.build_y (transformer/regulator stamping).
const YBUS_EPSILON_DEFAULT = 1e-5
const YBUS_EPSILON_IEEE13 = 1e-8
const YBUS_EPSILON_IEEE13_CASES = ("IEEE13Nodeckt", "IEEE13_test")
# Max ||Y_nn*u + Y_ns*u_slack|| / max(||Y_ns*u_slack||, 1) for linear warm starts.
const WARM_START_MAX_REL_RESIDUAL = 1e-8

"""
    default_ybus_epsilon(case_name)

Case-specific ``epsilon`` passed to ``FeederFlow.build_y`` when stamping the Y-bus.
IEEE 13 cases need smaller regularization than the default to avoid over-damping weak ties.
"""
function default_ybus_epsilon(case_name::AbstractString)
    return case_name in YBUS_EPSILON_IEEE13_CASES ? YBUS_EPSILON_IEEE13 : YBUS_EPSILON_DEFAULT
end

"""
    OPFData

Mutable container for a multiphase distribution AC-OPF instance: indexing, scaled
admittance ``Y_scaled``, switch/topology state, load/DER profiles, and JuMP-ready catalogs.

Node ordering matches FeederFlow: non-slack network nodes ``1:n_net``, then slack nodes
``n_net+1:n_total``. Switch lines are the only branches whose open/closed status can change;
``patch_switch_admittance!`` rebuilds ``Y_scaled`` from ``Y_scaled_base`` via Woodbury-style
per-switch stamps. Warm-start blocks ``Y_nn``, ``Y_ns`` support ``I_net = Y_nn * u_net + Y_ns * u_slack``.
"""
mutable struct OPFData
    case::String

    n_net::Int # network (non-slack) phase nodes
    n_slk::Int # slack phase nodes
    n_total::Int # total phase nodes

    n_der_unit::Int

    v_scale::Vector{Float64}
    v_scale_slack::Vector{Float64}
    v_scale_full::Vector{Float64}

    # Load
    p_load_ref::Vector{Float64}
    q_load_ref::Vector{Float64}
    p_load::Vector{Float64}
    q_load::Vector{Float64}

    # DER
    der_node::Vector{Int}
    der_phase_to_unit::Vector{Int}
    der_pavail_ref::Vector{Float64}
    der_smax::Vector{Float64}
    der_pmax::Vector{Float64}
    der_qmax::Vector{Float64}
    der_qmin::Vector{Float64}
    der_cost::Vector{Vector{Float64}}
    der_unit_to_phase::Vector{Vector{Int}}
    node_to_der_phase::Vector{Vector{Int}}

    # Slack
    sub_cost::Vector{Float64}
    sub_smax_phase::Vector{Float64}
    v_sub_ref::Vector{ComplexF64}

    # Switch catalog and status
    switch_idx::Vector{Int} # line indices in branch-status ordering
    switch_states::Dict{String, Bool}
    switch_name_to_idx::Dict{String, Int}
    switch_from_idx::Vector{Vector{Int}}
    switch_to_idx::Vector{Vector{Int}}
    switch_valid::Vector{Vector{Bool}}
    switch_y_series::Vector{Matrix{ComplexF64}}
    switch_y_shunt::Vector{Matrix{ComplexF64}}
    switch_is_closed_base::Vector{Bool}
    switch_is_closed::Vector{Bool}
    branch_status::Vector{Bool}

    # Indexing and branch constraint catalog
    all_index::Dict{Tuple{String, Int}, Int}
    all_is_slack::Vector{Bool}
    all_to_slack_local::Vector{Int}
    branch_from_idx::Vector{Vector{Int}}
    branch_to_idx::Vector{Vector{Int}}
    branch_valid::Vector{Vector{Bool}}
    branch_y_series::Vector{Matrix{ComplexF64}}
    branch_imax::Vector{Float64}

    # Voltage
    vmin::Vector{Float64}
    vmax::Vector{Float64}
    dvamax::Float64

    # Objective
    obj_weights::Vector{Float64}

    # Local-coordinate admittance matrices
    Y_scaled_base::Matrix{ComplexF64}
    Y_scaled::Matrix{ComplexF64}

    # Warm-start blocks for I_net = Y_nn * u_net + Y_ns * u_slack
    Y_nn::Matrix{ComplexF64}
    Y_ns::Matrix{ComplexF64}
end

"""Normalize generator/substation cost coefficients to ``[quadratic, linear]`` (MATPOWER-style)."""
function _as_cost_vector(raw_coeff)
    if raw_coeff isa AbstractVector
        return Float64.(raw_coeff)
    end
    # Scalar fallback is treated as linear-only and stored in [quad, lin] order.
    return [0.0, Float64(raw_coeff)]
end

"""Split a cost vector into ``(c_lin, c_quad)``; 1-term vectors are treated as linear-only."""
function _linear_quadratic_terms(cost_coeff::AbstractVector{<:Real})
    # Cost vectors follow [quadratic, linear] order (MATPOWER-style for 2-term use).
    # Keep compatibility with 1-term vectors by treating them as linear-only.
    if length(cost_coeff) >= 2
        c_quad = Float64(cost_coeff[1])
        c_lin = Float64(cost_coeff[2])
        return c_lin, c_quad
    elseif length(cost_coeff) == 1
        c_lin = Float64(cost_coeff[1])
        c_quad = 0.0
        return c_lin, c_quad
    end

    c_lin = 0.0
    c_quad = 0.0
    return c_lin, c_quad
end

_line_status_vector(network) = [line.is_closed for line in network.lines]

bus_phase_key(bus, phase) = (String(bus), Int(phase))
bus_phase_key(bp::FeederFlow.BusPhase) = (String(bp.bus), Int(bp.phase))

"""
    build_all_index_map(ybus)

Map ``(bus_name, phase)`` to global node index in ``ybus.all_order`` (network + slack).
"""
function build_all_index_map(ybus)
    all_index = Dict{Tuple{String, Int}, Int}()
    for (idx, bp) in enumerate(ybus.all_order)
        all_index[bus_phase_key(bp)] = idx
    end
    return all_index
end

"""Global node index for a line terminal phase, or ``0`` if that bus-phase is not in the Y-bus."""
function node_index_or_zero(all_index::Dict{Tuple{String, Int}, Int}, line_end, phase::Int)
    return get(all_index, bus_phase_key(line_end.bus, phase), 0)
end

"""
    build_line_terminal_cache(line, all_index)

Per-phase from/to global indices and validity flags for a multiphase line or switch.
Invalid entries (missing bus-phase in Y-bus) are marked with index ``0``.
"""
function build_line_terminal_cache(line, all_index::Dict{Tuple{String, Int}, Int})
    nph = length(line.phases)
    from_idx = Vector{Int}(undef, nph)
    to_idx = Vector{Int}(undef, nph)
    valid = Vector{Bool}(undef, nph)

    for (k, phase) in enumerate(line.phases)
        i_f = node_index_or_zero(all_index, line.from, phase)
        i_t = node_index_or_zero(all_index, line.to, phase)
        from_idx[k] = i_f
        to_idx[k] = i_t
        valid[k] = (i_f > 0 && i_t > 0)
    end

    return (from_idx=from_idx, to_idx=to_idx, valid=valid)
end

"""
    normalized_imax(normamps, emergamps, i_base)

Branch current limit in per-unit on system base; uses ``max(normamps, emergamps)`` from
FeederFlow, or ``IMAX_DEFAULT_AMPS / i_base`` when both ratings are zero.
"""
function normalized_imax(normamps, emergamps, i_base::Float64)
    imax_pu = max(Float64(normamps), Float64(emergamps)) # already coverted in FeederFlow parsing

    if imax_pu > 0.0
        return imax_pu
    end

    i_base > 0.0 || throw(ArgumentError("Base current must be positive; got i_base=$(i_base)"))
    return IMAX_DEFAULT_AMPS / i_base
end

"""
    build_branch_catalog(network, all_index, i_base)

JuMP branch-flow catalog for **closed, non-switch** lines only: terminal indices, series
admittance blocks, and per-unit ``I_max`` for the SOC current constraints in the OPF model.
"""
function build_branch_catalog(network, all_index::Dict{Tuple{String, Int}, Int}, i_base::Float64)
    branch_from_idx = Vector{Vector{Int}}()
    branch_to_idx = Vector{Vector{Int}}()
    branch_valid = Vector{Vector{Bool}}()
    branch_y_series = Vector{Matrix{ComplexF64}}()
    branch_imax = Float64[]

    for line in network.lines
        (line.is_switch || !line.is_closed) && continue

        admittance = FeederFlow.switch_line_admittance(network, line)
        admittance === nothing && continue
        y_series, _ = admittance

        terminal_cache = build_line_terminal_cache(line, all_index)
        push!(branch_from_idx, terminal_cache.from_idx)
        push!(branch_to_idx, terminal_cache.to_idx)
        push!(branch_valid, terminal_cache.valid)
        push!(branch_y_series, y_series)
        push!(branch_imax, normalized_imax(line.normamps, line.emergamps, i_base))
    end

    return (
        branch_from_idx = branch_from_idx,
        branch_to_idx = branch_to_idx,
        branch_valid = branch_valid,
        branch_y_series = branch_y_series,
        branch_imax = branch_imax,
    )
end

"""
    build_switch_catalog(network, all_index)

Catalog switch lines separately from the branch catalog: admittance stamps, terminal maps,
and mapping from switch name → compact switch id. ``switch_idx`` stores line indices in
``branch_status`` ordering (all ``network.lines``).
"""
function build_switch_catalog(network, all_index::Dict{Tuple{String, Int}, Int})
    switch_idx = Int[]
    switch_name_to_idx = Dict{String, Int}()
    switch_from_idx = Vector{Vector{Int}}()
    switch_to_idx = Vector{Vector{Int}}()
    switch_valid = Vector{Vector{Bool}}()
    switch_y_series = Vector{Matrix{ComplexF64}}()
    switch_y_shunt = Vector{Matrix{ComplexF64}}()
    switch_is_closed_base = Bool[]
    switch_is_closed = Bool[]

    for (line_idx, line) in enumerate(network.lines)
        line.is_switch || continue

        admittance = FeederFlow.switch_line_admittance(network, line)
        admittance === nothing && continue
        y_series, y_shunt = admittance

        terminal_cache = build_line_terminal_cache(line, all_index)
        switch_id = length(switch_from_idx) + 1

        switch_name_to_idx[line.name] = switch_id
        push!(switch_idx, line_idx)
        push!(switch_from_idx, terminal_cache.from_idx)
        push!(switch_to_idx, terminal_cache.to_idx)
        push!(switch_valid, terminal_cache.valid)
        push!(switch_y_series, y_series)
        push!(switch_y_shunt, y_shunt)
        push!(switch_is_closed_base, line.is_closed_base)
        push!(switch_is_closed, line.is_closed)
    end

    switch_states = Dict(name => switch_is_closed[idx] for (name, idx) in switch_name_to_idx)

    return (
        switch_idx = switch_idx,
        switch_states = switch_states,
        switch_name_to_idx = switch_name_to_idx,
        switch_from_idx = switch_from_idx,
        switch_to_idx = switch_to_idx,
        switch_valid = switch_valid,
        switch_y_series = switch_y_series,
        switch_y_shunt = switch_y_shunt,
        switch_is_closed_base = switch_is_closed_base,
        switch_is_closed = switch_is_closed,
    )
end



"""
    build_load_reference_vectors(network, ybus, noload)

Compute reference P/Q load (p.u.) at each non-slack phase node from the no-load voltage profile.

Multi-phase loads use branch-pair powers from FeederFlow; single-phase loads use terminal
current ``I = conj(S/V)`` at the no-load voltage.
"""
function build_load_reference_vectors(network, ybus, noload)
    n = length(ybus.network_order)
    p_load_ref = zeros(Float64, n)
    q_load_ref = zeros(Float64, n)

    for load in network.loads
        pairs = FeederFlow.branch_pairs(load)
        isempty(pairs) && continue
        powers = FeederFlow.branch_powers(load, length(pairs), network.base)

        for (pair, sbranch) in zip(pairs, powers)
            p_idx = FeederFlow.lookup_node_index(ybus, load.bus.bus, pair[1])
            p_idx == 0 && continue

            vp = get(noload.phase_voltages, FeederFlow.BusPhase(load.bus.bus, pair[1]), nothing)
            vp === nothing && continue

            if pair[2] == 0
                iszero(vp) && continue
                current = conj(sbranch / vp)
                terminal_power = vp * conj(current)
                p_load_ref[p_idx] += real(terminal_power)
                q_load_ref[p_idx] += imag(terminal_power)
                continue
            end

            q_idx = FeederFlow.lookup_node_index(ybus, load.bus.bus, pair[2])
            q_idx == 0 && continue

            vq = get(noload.phase_voltages, FeederFlow.BusPhase(load.bus.bus, pair[2]), nothing)
            vq === nothing && continue

            branch_voltage = vp - vq
            iszero(branch_voltage) && continue

            current = conj(sbranch / branch_voltage)
            terminal_power_p = vp * conj(current)
            terminal_power_q = -vq * conj(current)

            p_load_ref[p_idx] += real(terminal_power_p)
            q_load_ref[p_idx] += imag(terminal_power_p)
            p_load_ref[q_idx] += real(terminal_power_q)
            q_load_ref[q_idx] += imag(terminal_power_q)
        end
    end

    return p_load_ref, q_load_ref
end

"""Convenience wrapper: run no-load power flow, then [`build_load_reference_vectors`](@ref)."""
function build_load_reference_vectors(network, ybus)
    v_slack = FeederFlow.source_slack(network.source, network.base)
    noload = FeederFlow.compute_no_load(ybus; v_slack=v_slack)
    return build_load_reference_vectors(network, ybus, noload)
end

"""
    build_load_reference_vectors2(network, ybus)

Fallback reference loads: split each load's ``p_pu``, ``q_pu`` equally across its bus phases
(without a no-load voltage solve). Used when the full no-load-based allocation is unavailable.
"""
function build_load_reference_vectors2(network, ybus)
    # Simpler fallback that spreads each load equally across connected phases.
    n = length(ybus.network_order)
    p_load_ref = zeros(Float64, n)
    q_load_ref = zeros(Float64, n)
    for load in network.loads
        nph = length(load.bus.phases)
        p_phase = load.p_pu / nph
        q_phase = load.q_pu / nph
        for phase in load.bus.phases
            idx = FeederFlow.lookup_node_index(ybus, load.bus.bus, phase)
            idx > 0 || continue
            p_load_ref[idx] += p_phase
            q_load_ref[idx] += q_phase
        end
    end
    return p_load_ref, q_load_ref
end


"""
    map_generators(network, ybus)

Map FeederFlow generators to per-phase DER slots and physical DER units.

Each connected phase gets one OPF ``pder``/``qder`` variable; active power cap per phase is
``gen.p_pu / n_phases``. Unit-level availability ``der_pavail`` is stored once per generator.
"""
function map_generators(network, ybus)
    der_node = Int[]
    der_phase_to_unit = Int[]
    der_pavail = Float64[]
    der_pmax = Float64[]
    der_qmax = Float64[]
    der_qmin = Float64[]
    der_cost = Vector{Vector{Float64}}()

    for gen in network.generators
        mapped_node_idx = Int[]
        for phase in gen.bus.phases
            idx = FeederFlow.lookup_node_index(ybus, gen.bus.bus, phase)
            idx > 0 && push!(mapped_node_idx, idx)
        end
        isempty(mapped_node_idx) && continue

        der_unit_id = length(der_pavail) + 1
        push!(der_pavail, gen.p_pu)

        pmax_per_phase = gen.p_pu / length(mapped_node_idx)
        for idx in mapped_node_idx
            push!(der_node, idx)
            push!(der_phase_to_unit, der_unit_id)
            push!(der_pmax, pmax_per_phase)
            push!(der_cost, _as_cost_vector(gen.cost_coeff))
        end
    end

    der_qmax = copy(der_pmax)
    der_qmin = -der_pmax
    return der_node, der_phase_to_unit, der_pavail, der_pmax, der_qmax, der_qmin, der_cost
end

"""
    build_der_group_maps(der_node, der_phase_to_unit, n_net, n_der_unit)

Inverse maps: ``der_unit_to_phase[g]`` lists phase-slot indices for unit ``g``;
``node_to_der_phase[i]`` lists phase slots at network node ``i`` (for nodal balance constraints).
"""
function build_der_group_maps(der_node, der_phase_to_unit, n_net, n_der_unit)
    length(der_node) == length(der_phase_to_unit) || throw(DimensionMismatch("der_node and der_phase_to_unit must have equal length"))

    der_unit_to_phase = [Int[] for _ in 1:n_der_unit]
    node_to_der_phase = [Int[] for _ in 1:n_net]

    for (k, unit_id) in enumerate(der_phase_to_unit)
        1 <= unit_id <= n_der_unit || throw(ArgumentError("Invalid der unit id $unit_id for DER slot $k"))
        node_idx = der_node[k]
        1 <= node_idx <= n_net || throw(ArgumentError("Invalid DER node index $node_idx for DER slot $k"))

        push!(der_unit_to_phase[unit_id], k)
        push!(node_to_der_phase[node_idx], k)
    end

    return der_unit_to_phase, node_to_der_phase
end

"""
    build_all_node_partition_maps(ybus, n_net, n_slk)

Flags which global indices are slack and maps each slack node to its local slack index ``1:n_slk``.
"""
function build_all_node_partition_maps(ybus, n_net, n_slk)
    n_total = n_net + n_slk
    length(ybus.all_order) == n_total || throw(DimensionMismatch("all_order length does not match n_total"))

    slack_lookup = Dict(bp => k for (k, bp) in enumerate(ybus.slack_order))
    all_is_slack = falses(n_total)
    all_to_slack_local = zeros(Int, n_total)

    for (i, bp) in enumerate(ybus.all_order)
        k = get(slack_lookup, bp, 0)
        if k > 0
            all_is_slack[i] = true
            all_to_slack_local[i] = k
        end
    end

    return all_is_slack, all_to_slack_local
end

"""Assert FeederFlow indexing and DER maps are consistent before building the OPF model."""
function validate_indexing_invariants(ybus, n_net, n_slk, n_total, der_node, der_phase_to_unit, der_pavail, all_is_slack, all_to_slack_local)
    @assert n_net == length(ybus.network_order)
    @assert n_slk == length(ybus.slack_order)
    @assert n_total == length(ybus.all_order)
    @assert ybus.all_order == vcat(ybus.network_order, ybus.slack_order)

    for (idx, bp) in enumerate(ybus.network_order)
        @assert get(ybus.network_index, bp, 0) == idx
    end
    for (idx, bp) in enumerate(ybus.all_order)
        @assert get(ybus.all_index, bp, 0) == idx
    end

    @assert all(!all_is_slack[i] for i in 1:n_net)
    @assert all(all_is_slack[i] for i in (n_net + 1):n_total)
    @assert all(all_to_slack_local[i] == 0 for i in 1:n_net)
    @assert all(1 <= all_to_slack_local[i] <= n_slk for i in (n_net + 1):n_total)

    length(der_node) == length(der_phase_to_unit) || throw(DimensionMismatch("der index arrays are inconsistent"))
    all(1 <= idx <= n_net for idx in der_node) || throw(ArgumentError("der_node has out-of-range indices"))

    n_der_unit = length(der_pavail)
    all(1 <= g <= n_der_unit for g in der_phase_to_unit) || throw(ArgumentError("der_phase_to_unit has out-of-range ids"))

    return nothing
end

"""
    estimate_substation_smax(network)

Estimate substation apparent-power cap (p.u.) from slack-side transformer/regulator kVA,
with floor ``max(2 * total_load_MVA, 0.2)`` when equipment ratings are missing.
"""
function estimate_substation_smax(network)
    kva_candidates = Float64[]
    for tr in vcat(collect(network.transformers), collect(network.regulators))
        for w in tr.windings
            w.bus.bus == network.slack_bus || continue
            push!(kva_candidates, w.kva)
        end
    end
    raw = isempty(kva_candidates) ? 0.0 : maximum(kva_candidates) * 1000.0 / network.base.Sbase
    p_total = sum(load.p_pu for load in network.loads)
    q_total = sum(load.q_pu for load in network.loads)
    demand = hypot(p_total, q_total)
    return max(raw, 2.0 * demand + 1e-3, 0.2)
end

"""
    estimate_substation_smax_phase(network, n_slk)

Per-phase apparent-power cap (p.u.): equal split of [`estimate_substation_smax`](@ref) across slack phases.
"""
function estimate_substation_smax_phase(network, n_slk::Int)
    n_slk > 0 || throw(ArgumentError("n_slk must be positive"))
    per_phase = estimate_substation_smax(network) / n_slk
    return fill(per_phase, n_slk)
end

"""
    OPFData(case_name, path; epsilon)

Construct network data from a FeederFlow case file: Y-bus, loads, DERs, switches, branch catalog,
and patched admittance for the initial switch topology.

# Arguments
- `case_name`: label stored in `data.case` (also selects default Y-bus regularization).
- `path`: path to the OpenDSS / FeederFlow case file.
- `epsilon`: regularization for `FeederFlow.build_y` (see [`default_ybus_epsilon`](@ref)).
"""
function OPFData(
    case_name::AbstractString,
    path::AbstractString;
    epsilon::Float64=default_ybus_epsilon(case_name),
)
    network = FeederFlow.parse_file(path)
    ybus = FeederFlow.build_y(network; regulator_model=:nonideal, epsilon=epsilon)
    v_sub_ref = FeederFlow.source_slack(network.source, network.base)
    noload = FeederFlow.compute_no_load(ybus; v_slack=v_sub_ref)

    p_load_ref, q_load_ref = build_load_reference_vectors(network, ybus, noload)
    p_load = 1.0 .* p_load_ref
    q_load = 1.0 .* q_load_ref

    n_net = length(ybus.network_order)
    n_slk = length(ybus.slack_order)
    n_total = n_net + n_slk

    v_scale = [network.buses[bp.bus].vbase / network.base.Vbase for bp in ybus.network_order]
    v_scale_slack = [FeederFlow.kv_to_vbase(network.source.basekv, network.source.phases) / network.base.Vbase for bp in ybus.slack_order]
    v_scale_full = vcat(v_scale, v_scale_slack)

    all_index = build_all_index_map(ybus)
    Y_scaled_base = FeederFlow.scaled_ybus_matrix(ybus, v_scale_full)

    der_node, der_phase_to_unit, der_pavail_ref, der_pmax, der_qmax, der_qmin, der_cost = map_generators(network, ybus)
    der_smax = copy(der_pmax)

    n_der_unit = length(der_pavail_ref)
    der_unit_to_phase, node_to_der_phase = build_der_group_maps(der_node, der_phase_to_unit, n_net, n_der_unit)

    all_is_slack, all_to_slack_local = build_all_node_partition_maps(ybus, n_net, n_slk)
    validate_indexing_invariants(
        ybus, n_net, n_slk, n_total,
        der_node, der_phase_to_unit, der_pavail_ref,
        all_is_slack, all_to_slack_local,
    )

    sub_smax_phase = estimate_substation_smax_phase(network, n_slk)
    sub_cost = _as_cost_vector(network.source.cost_coeff)

    i_base = network.base.Sbase / network.base.Vbase
    branch_catalog = build_branch_catalog(network, all_index, i_base)
    switch_catalog = build_switch_catalog(network, all_index)

    branch_status = _line_status_vector(network)

    if n_net > 5000
        vmin = fill(VMIN_DEFAULT_LARGE, n_total)
        vmax = fill(VMAX_DEFAULT_LARGE, n_total)
        @info "Voltage limits: [$(VMIN_DEFAULT_LARGE), $(VMAX_DEFAULT_LARGE)]"
    else
        vmin = fill(VMIN_DEFAULT, n_total)
        vmax = fill(VMAX_DEFAULT, n_total)
        @info "Voltage limits: [$(VMIN_DEFAULT), $(VMAX_DEFAULT)]"
    end
    dvamax = DVA_MAX_DEFAULT

    # Weighted objective [w_cost, w_curtail, ρ_V, ρ_I, ρ_θ]: economic cost only, default soft penalties.
    obj_weights = vcat([1.0, 0.0], fill(PENALTY_WEIGHT_DEFAULT, N_OBJ_WEIGHTS - N_ECON_WEIGHTS))

    data = OPFData(
        String(case_name),
        n_net, n_slk, n_total, n_der_unit,
        v_scale, v_scale_slack, v_scale_full,
        p_load_ref, q_load_ref, p_load, q_load,
        der_node, der_phase_to_unit, der_pavail_ref,
        der_smax, der_pmax, der_qmax, der_qmin, der_cost,
        der_unit_to_phase, node_to_der_phase,
        sub_cost, sub_smax_phase, v_sub_ref,
        switch_catalog.switch_idx,
        switch_catalog.switch_states,
        switch_catalog.switch_name_to_idx,
        switch_catalog.switch_from_idx,
        switch_catalog.switch_to_idx,
        switch_catalog.switch_valid,
        switch_catalog.switch_y_series,
        switch_catalog.switch_y_shunt,
        switch_catalog.switch_is_closed_base,
        switch_catalog.switch_is_closed,
        branch_status,
        all_index,
        all_is_slack,
        all_to_slack_local,
        branch_catalog.branch_from_idx,
        branch_catalog.branch_to_idx,
        branch_catalog.branch_valid,
        branch_catalog.branch_y_series,
        branch_catalog.branch_imax,
        vmin, vmax, dvamax,
        obj_weights,
        Y_scaled_base,
        copy(Y_scaled_base),
        zeros(ComplexF64, n_net, n_net),
        zeros(ComplexF64, n_net, n_slk),
    )

    patch_switch_admittance!(data)
    return data
end

"""Load `OPFData` from a case file path; case name inferred from basename."""
function create_network_data(case_path::AbstractString)
    case_name = splitext(basename(case_path))[1]
    return OPFData(case_name, case_path)
end

"""Load `OPFData` with an explicit case label (e.g. `IEEE13Nodeckt`)."""
function create_network_data(case_name::AbstractString, case_path::AbstractString)
    return OPFData(case_name, case_path)
end
