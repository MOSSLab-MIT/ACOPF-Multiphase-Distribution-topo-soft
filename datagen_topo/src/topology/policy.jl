using JSON

"""
    TopologyPolicy

Per-feeder topology perturbation policy loaded from `topology_policy.json`.
"""
struct TopologyPolicy
    schema_version::Int
    perturb_sw::Vector{String}
    repair_nc::Vector{String}
    repair_hints::Dict{String, String}
    perturb_line_idx::Vector{Int}
    repair_line_idx::Vector{Int}
    changing_switches::Vector{Int}
end

function topology_policy_path(policy_dir::AbstractString)
    return joinpath(policy_dir, "topology_policy.json")
end

function _normalize_switch_name(name::AbstractString)
    return lowercase(strip(String(name)))
end

function _line_index_for_switch(data::OPFData, name::AbstractString)
    key = _normalize_switch_name(name)
    haskey(data.switch_name_to_idx, key) ||
        error("Switch '$name' (normalized '$key') is not present in feeder switch catalog.")
    switch_id = data.switch_name_to_idx[key]
    return data.switch_idx[switch_id]
end

function _validate_repair_nc_at_base!(data::OPFData, repair_nc::Vector{String}, repair_line_idx::Vector{Int})
    for (name, line_idx) in zip(repair_nc, repair_line_idx)
        switch_id = data.switch_name_to_idx[name]
        data.switch_is_closed_base[switch_id] ||
            error("repair_nc switch '$name' (line $line_idx) must be normally closed at base.")
    end
    return nothing
end

"""
    load_topology_policy(grid_dir, data::OPFData) -> TopologyPolicy

Load required `topology_policy.json` from `grid_dir` and map switch names to 1-based line indices.
"""
function load_topology_policy(policy_dir::AbstractString, data::OPFData)
    path = topology_policy_path(policy_dir)
    isfile(path) || error("Missing required topology policy file: $path")

    raw = JSON.parsefile(path)
    schema_version = Int(get(raw, "schema_version", 0))
    schema_version == 1 || error("Unsupported topology_policy schema_version=$schema_version (expected 1).")

    perturb_sw = [_normalize_switch_name(n) for n in get(raw, "perturb_sw", String[])]
    repair_nc = [_normalize_switch_name(n) for n in get(raw, "repair_nc", String[])]
    repair_hints_raw = get(raw, "repair_hints", Dict{String, Any}())
    repair_hints = Dict{String, String}(
        _normalize_switch_name(k) => _normalize_switch_name(v) for (k, v) in repair_hints_raw
    )

    perturb_line_idx = Int[_line_index_for_switch(data, name) for name in perturb_sw]
    repair_line_idx = Int[_line_index_for_switch(data, name) for name in repair_nc]
    changing_switches = sort(unique(vcat(perturb_line_idx, repair_line_idx)))

    expected_len = length(unique(vcat(perturb_line_idx, repair_line_idx)))
    length(changing_switches) == expected_len ||
        error("changing_switches length $(length(changing_switches)) != unique perturb∪repair length $expected_len")

    for (pert_name, repair_name) in repair_hints
        pert_name in perturb_sw ||
            error("repair_hints key '$pert_name' is not listed in perturb_sw.")
        repair_name in repair_nc ||
            error("repair_hints value '$repair_name' is not listed in repair_nc.")
    end

    _validate_repair_nc_at_base!(data, repair_nc, repair_line_idx)

    return TopologyPolicy(
        schema_version,
        perturb_sw,
        repair_nc,
        repair_hints,
        perturb_line_idx,
        repair_line_idx,
        changing_switches,
    )
end

"""True when `perturb_sw` lists at least one switch (random flips may change topology)."""
policy_perturbation_active(policy::TopologyPolicy) = !isempty(policy.perturb_line_idx)

"""True when neither perturb nor repair switches are configured (nominal topology only)."""
policy_topology_fixed(policy::TopologyPolicy) = isempty(policy.changing_switches)

"""True when perturbation is off but NC repair switches may still be opened."""
policy_repair_only(policy::TopologyPolicy) =
    isempty(policy.perturb_line_idx) && !isempty(policy.repair_line_idx)

function changing_switch_status(policy::TopologyPolicy, branch_status::AbstractVector{Bool})
    return branch_status[policy.changing_switches]
end

function expand_changing_switch_status(
    policy::TopologyPolicy,
    base_status::AbstractVector{Bool},
    compact_status::AbstractVector{Bool},
)
    length(compact_status) == length(policy.changing_switches) || throw(DimensionMismatch(
        "changing_switch_status length $(length(compact_status)); expected $(length(policy.changing_switches)).",
    ))
    full = copy(base_status)
    full[policy.changing_switches] .= compact_status
    return full
end
