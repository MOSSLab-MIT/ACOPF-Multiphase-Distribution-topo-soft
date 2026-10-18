"""
Human-readable topology logging for data generation.

Used when a new scenario topology is sampled (scenario-mode datasets).
"""

"""Switch state label for logs: `C` = closed (energized), `O` = open."""
function _state_char(closed::Bool)
    return closed ? "C" : "O"
end

function line_switch_name(ts::TopologySampler, line_idx::Int)
    return get(ts.line_idx_to_name, line_idx, "line_$line_idx")
end

"""
    switch_flip_label(ts, line_idx, base_status, status) -> String

e.g. `cb_102:O→C` (open→closed).
"""
function switch_flip_label(
    ts::TopologySampler,
    line_idx::Int,
    base_status::AbstractVector{Bool},
    status::AbstractVector{Bool},
)
    name = line_switch_name(ts, line_idx)
    return "$(name):$(_state_char(base_status[line_idx]))→$(_state_char(status[line_idx]))"
end

"""
    format_topology_scenario_line(ts, topo_meta; scenario_id, topology_seed) -> String

One-line summary: radial intent, radiality, connectivity, perturb flips, NC repair opens.
"""
function format_topology_scenario_line(
    ts::TopologySampler,
    topo_meta::AbstractDict;
    scenario_id::Union{Int, Nothing}=nothing,
    topology_seed::Union{Int, Nothing}=nothing,
)
    parts = String[]
    scenario_id !== nothing && push!(parts, "scenario=$(scenario_id)")
    topology_seed !== nothing && push!(parts, "topo_seed=$(topology_seed)")
    topology_seed === nothing && get(topo_meta, :topology_fixed, false) &&
        push!(parts, "topo_seed=nominal")

    push!(parts, "radial_intent=$(get(topo_meta, :radial_intent, false))")
    push!(parts, "is_radial=$(get(topo_meta, :is_radial, false))")
    push!(parts, "connected=$(get(topo_meta, :is_connected, false))")

    if get(topo_meta, :topology_fixed, false)
        push!(parts, "mode=fixed_nominal")
    elseif !get(topo_meta, :perturbation_active, true)
        push!(parts, "mode=repair_only")
    else
        push!(parts, "mode=perturb")
    end

    get(topo_meta, :repair_failed_downgrade, false) && push!(parts, "repair_downgrade=true")

    base = ts.base_status
    status_after = get(topo_meta, :final_status, nothing)
    if status_after !== nothing
        flipped = get(topo_meta, :perturb_flipped_lines, Int[])
        if isempty(flipped)
            push!(parts, "perturb_flips=none")
        else
            labels = [switch_flip_label(ts, idx, base, status_after) for idx in flipped]
            push!(parts, "perturb_flips=$(join(labels, ", "))")
        end

        opened = get(topo_meta, :repair_opened_lines, Int[])
        if isempty(opened)
            push!(parts, "repair_nc_opened=none")
        else
            names = [line_switch_name(ts, idx) for idx in opened]
            push!(parts, "repair_nc_opened=$(join(names, ", "))")
        end

        hint_opened = get(topo_meta, :repair_opened_via_hint, Int[])
        greedy_opened = get(topo_meta, :repair_opened_via_greedy, Int[])
        if !isempty(hint_opened) || !isempty(greedy_opened)
            hint_names = [line_switch_name(ts, idx) for idx in hint_opened]
            greedy_names = [line_switch_name(ts, idx) for idx in greedy_opened]
            push!(parts, "repair_hint=$(isempty(hint_names) ? "none" : join(hint_names, ", "))")
            push!(parts, "repair_greedy=$(isempty(greedy_names) ? "none" : join(greedy_names, ", "))")
        end
    end

    return "[Topology Scenario] " * join(parts, " | ")
end

"""
    log_topology_scenario_start!(ts, topo_meta; scenario_id, topology_seed)

Log a single-line topology summary when a new scenario begins (`@info`).
"""
function log_topology_scenario_start!(
    ts::TopologySampler,
    topo_meta::AbstractDict;
    scenario_id::Union{Int, Nothing}=nothing,
    topology_seed::Union{Int, Nothing}=nothing,
)
    @info format_topology_scenario_line(
        ts,
        topo_meta;
        scenario_id=scenario_id,
        topology_seed=topology_seed,
    )
    return nothing
end
