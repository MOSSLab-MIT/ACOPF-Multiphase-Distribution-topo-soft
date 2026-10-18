"""
    TopologySampler

Policy-driven topology sampler using `topology_policy.json`.

Fields:
- `base_status`: baseline branch status (all lines)
- `policy`: [`TopologyPolicy`](@ref) with `perturb_sw`, `repair_nc`, `changing_switches`
- `graph_cache`: bus-level graph for radial / connectivity checks
- `flip_probability`: Bernoulli flip probability per `perturb_line_idx`
- `radial_ratio`: probability of attempting radial repair path
"""
struct TopologySampler
    base_status::Vector{Bool}
    policy::TopologyPolicy
    graph_cache::TopologyGraphCache
    line_idx_to_name::Dict{Int, String}
    flip_probability::Float64
    radial_ratio::Float64
    max_topology_attempts::Int
end

function changing_switches(ts::TopologySampler)
    return ts.policy.changing_switches
end

function _unique_sorted_indices(indices::AbstractVector{<:Integer}, upper::Int, name::AbstractString)
    out = sort(unique(Int.(indices)))
    for idx in out
        1 <= idx <= upper || error("$name contains out-of-range index $idx (expected 1:$upper).")
    end
    return out
end

function _make_mask(upper::Int, indices::Vector{Int})
    mask = falses(upper)
    isempty(indices) || (mask[indices] .= true)
    return mask
end

function _flip_perturb_switches!(
    status::Vector{Bool},
    perturb_line_idx::Vector{Int},
    rng::AbstractRNG,
    flip_probability::Float64,
)
    flipped = Int[]
    for idx in perturb_line_idx
        if rand(rng) < flip_probability
            status[idx] = !status[idx]
            push!(flipped, idx)
        end
    end
    return flipped
end

function _topology_meta(;
    is_radial::Bool,
    is_connected::Bool=false,
    repair_opened_lines::Vector{Int}=Int[],
    repair_opened_via_hint::Vector{Int}=Int[],
    repair_opened_via_greedy::Vector{Int}=Int[],
    perturb_flipped_lines::Vector{Int}=Int[],
    repair_failed_downgrade::Bool=false,
    radial_intent::Bool=false,
    topology_fixed::Bool=false,
    perturbation_active::Bool=true,
    final_status::Union{Nothing, Vector{Bool}}=nothing,
)
    return Dict{Symbol, Any}(
        :is_radial => is_radial,
        :is_connected => is_connected,
        :repair_opened_lines => repair_opened_lines,
        :repair_opened_via_hint => repair_opened_via_hint,
        :repair_opened_via_greedy => repair_opened_via_greedy,
        :perturb_flipped_lines => perturb_flipped_lines,
        :repair_failed_downgrade => repair_failed_downgrade,
        :radial_intent => radial_intent,
        :topology_fixed => topology_fixed,
        :perturbation_active => perturbation_active,
        :final_status => final_status,
    )
end

function _validate_base_topology_connected!(ts::TopologySampler)
    is_loads_connected(ts.graph_cache, ts.base_status) && return nothing
    error(
        "Base feeder topology is not load-connected. " *
        "topology_policy has empty perturb_sw and repair_nc; fix the DSS model or graph coupling.",
    )
end

function _fixed_topology_sample(ts::TopologySampler)
    status = copy(ts.base_status)
    is_radial = is_radial_feeder(ts.graph_cache, status)
    is_connected = is_loads_connected(ts.graph_cache, status)
    return status, _topology_meta(;
        is_radial=is_radial,
        is_connected=is_connected,
        topology_fixed=true,
        perturbation_active=false,
        final_status=status,
    )
end

"""
    sample_topology(rng, ts) -> (branch_status, meta)

`meta` keys: `is_radial`, `repair_opened_lines`, `repair_failed_downgrade`, `radial_intent`,
`topology_fixed`, `perturbation_active`.
"""
function sample_topology(rng::AbstractRNG, ts::TopologySampler)
    0.0 <= ts.flip_probability <= 1.0 || throw(ArgumentError("flip_probability must be in [0, 1]."))
    0.0 <= ts.radial_ratio <= 1.0 || throw(ArgumentError("radial_ratio must be in [0, 1]."))

    if policy_topology_fixed(ts.policy)
        return _fixed_topology_sample(ts)
    end

    perturbation_active = policy_perturbation_active(ts.policy)
    radial_intent = perturbation_active && (rand(rng) < ts.radial_ratio)
    status = copy(ts.base_status)
    perturb_flipped = Int[]
    if perturbation_active
        perturb_flipped = _flip_perturb_switches!(
            status,
            ts.policy.perturb_line_idx,
            rng,
            ts.flip_probability,
        )
    end

    repair_opened_lines = Int[]
    repair_opened_via_hint = Int[]
    repair_opened_via_greedy = Int[]
    repair_failed_downgrade = false
    is_radial = false

    if radial_intent && !isempty(ts.policy.repair_line_idx)
        repaired = minimal_nc_repair(
            status,
            ts.graph_cache,
            ts.policy.repair_line_idx;
            repair_hints=ts.policy.repair_hints,
            line_idx_to_name=ts.line_idx_to_name,
            perturb_line_idx=ts.policy.perturb_line_idx,
            base_status=ts.base_status,
        )
        if repaired === nothing
            repair_failed_downgrade = true
            is_radial = false
        else
            status, repair_opened_lines, repair_opened_via_hint, repair_opened_via_greedy = repaired
            is_radial = is_radial_feeder(ts.graph_cache, status)
        end
    elseif radial_intent
        is_radial = is_radial_feeder(ts.graph_cache, status)
        if !is_radial
            repair_failed_downgrade = true
        end
    elseif !perturbation_active
        # repair-only policy: nominal switches; report actual radiality of base case.
        is_radial = is_radial_feeder(ts.graph_cache, status)
    else
        # radial_intent=false: NC repair skipped, but topology may still be a tree
        # (e.g. perturb flip missed or tie left open). Always report structural radiality.
        is_radial = is_radial_feeder(ts.graph_cache, status)
    end

    is_connected = is_loads_connected(ts.graph_cache, status)

    return status, _topology_meta(;
        is_radial=is_radial,
        is_connected=is_connected,
        repair_opened_lines=repair_opened_lines,
        repair_opened_via_hint=repair_opened_via_hint,
        repair_opened_via_greedy=repair_opened_via_greedy,
        perturb_flipped_lines=perturb_flipped,
        repair_failed_downgrade=repair_failed_downgrade,
        radial_intent=radial_intent,
        topology_fixed=false,
        perturbation_active=perturbation_active,
        final_status=status,
    )
end

function sample_topology_connected(
    rng::AbstractRNG,
    ts::TopologySampler;
    max_attempts::Int=ts.max_topology_attempts,
)
    if policy_topology_fixed(ts.policy)
        _validate_base_topology_connected!(ts)
        return _fixed_topology_sample(ts)
    end

    if !policy_perturbation_active(ts.policy)
        # No random flips: topology only changes via repair path (deterministic given rng draw).
        max_attempts = min(max_attempts, 2)
    end

    for _ in 1:max_attempts
        status, meta = sample_topology(rng, ts)
        if is_loads_connected(ts.graph_cache, status)
            return status, meta
        end
    end
    error("Unable to sample a connected topology after $max_attempts attempts.")
end

function Random.rand(rng::AbstractRNG, ts::TopologySampler)
    status, _ = sample_topology_connected(rng, ts)
    return status
end

function TopologySampler(data::OPFData, options::Dict, case_path::AbstractString)
    base_status = copy(data.branch_status)
    policy = load_topology_policy(dirname(abspath(case_path)), data)
    graph_cache = build_topology_graph_cache(case_path)

    line_idx_to_name = Dict{Int, String}()
    for (name, sid) in data.switch_name_to_idx
        line_idx_to_name[data.switch_idx[sid]] = name
    end

    flip_probability = Float64(get(options, "flip_probability", get(options, "p_on", 0.3)))
    radial_ratio = Float64(get(options, "radial_ratio", 0.5))
    max_topology_attempts = Int(get(options, "max_topology_attempts", 200))

    (0.0 <= flip_probability <= 1.0) || error("flip_probability must be in [0, 1].")
    (0.0 <= radial_ratio <= 1.0) || error("radial_ratio must be in [0, 1].")
    max_topology_attempts > 0 || error("max_topology_attempts must be positive.")

    if !policy_perturbation_active(policy)
        if policy_topology_fixed(policy)
            @info "topology_policy has empty perturb_sw and repair_nc; using fixed nominal topology for all scenarios."
        else
            @info "topology_policy has empty perturb_sw; flip_probability is ignored (repair_nc only)."
        end
    end

    return TopologySampler(
        base_status,
        policy,
        graph_cache,
        line_idx_to_name,
        flip_probability,
        radial_ratio,
        max_topology_attempts,
    )
end

function TopologySampler(data::OPFData, options::Dict)
    case_path = get(options, "case_path", nothing)
    case_path === nothing && error(
        "TopologySampler requires topology_policy.json; pass options[\"case_path\"] to the feeder DSS file.",
    )
    return TopologySampler(data, options, String(case_path))
end
