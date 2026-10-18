using Random

function _sample_topology_with_retries(
    ts::TopologySampler,
    seed_base::Int,
    scenario_idx::Int,
    max_attempts::Int,
)
    last_error = ""
    for attempt in 1:max_attempts
        topo_seed = seed_base + (scenario_idx - 1) * 1_000_003 + attempt
        rng = MersenneTwister(topo_seed)
        try
            branch_status, meta = sample_topology_connected(rng, ts)
            return branch_status, meta, topo_seed
        catch err
            last_error = sprint(showerror, err)
        end
    end
    error(
        "Unable to sample topology for scenario $scenario_idx after $max_attempts attempts. Last error: $last_error",
    )
end

function _expand_compact_status(ts::TopologySampler, compact_status)
    return expand_changing_switch_status(ts.policy, ts.base_status, compact_status)
end

function build_topology_plan(
    ts::TopologySampler,
    num_scenarios::Int,
    seed_base::Int;
    topology_retry_attempts::Int=200,
)
    plan = Dict{String, Any}[]

    if policy_topology_fixed(ts.policy)
        try
            _validate_base_topology_connected!(ts)
            branch_status, topo_meta = _fixed_topology_sample(ts)
            compact_status = changing_switch_status(ts.policy, branch_status)
            is_radial = Bool(get(topo_meta, :is_radial, false))
            for scenario_idx in 1:num_scenarios
                push!(plan, Dict{String, Any}(
                    "scenario_id" => scenario_idx,
                    "branch_status" => copy(branch_status),
                    "changing_switch_status" => copy(compact_status),
                    "is_radial" => is_radial,
                    "topology_seed" => nothing,
                    "topology_source" => "fixed_nominal",
                    "topology_meta" => topo_meta,
                ))
            end
        catch err
            err_msg = sprint(showerror, err)
            for scenario_idx in 1:num_scenarios
                push!(plan, Dict{String, Any}(
                    "scenario_id" => scenario_idx,
                    "topology_error" => err_msg,
                ))
            end
        end
        return plan
    end

    for scenario_idx in 1:num_scenarios
        try
            topo_seed = seed_base + (scenario_idx - 1) * 1_000_003
            rng = MersenneTwister(topo_seed)
            branch_status, topo_meta = sample_topology_connected(rng, ts)
            compact_status = changing_switch_status(ts.policy, branch_status)
            push!(plan, Dict{String, Any}(
                "scenario_id" => scenario_idx,
                "branch_status" => branch_status,
                "changing_switch_status" => compact_status,
                "is_radial" => Bool(get(topo_meta, :is_radial, false)),
                "topology_seed" => topo_seed,
                "topology_source" => "sampled",
                "topology_meta" => topo_meta,
            ))
        catch err
            push!(plan, Dict{String, Any}(
                "scenario_id" => scenario_idx,
                "topology_error" => sprint(showerror, err),
            ))
        end
    end

    return plan
end
