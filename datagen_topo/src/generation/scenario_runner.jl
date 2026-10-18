import Ipopt
using Random

function run_scenario_generation!(;
    network,
    ts::TopologySampler,
    ls,
    ds,
    os,
    float_type,
    root_dir::AbstractString,
    case_name::AbstractString,
    topology_cfg::Dict,
    num_scenarios::Int,
    feasible_samples_per_scenario::Int,
    infeasible_samples_per_scenario::Union{Nothing, Int}=nothing,
    max_attempts_multiplier::Int,
    seed_base::Int,
    seed_start::Int,
    variant_label::AbstractString,
    infeasible_filename::AbstractString,
    feasible_filename::AbstractString,
    transform_profile!::Function,
)
    num_scenarios > 0 || error("num_scenarios must be positive.")
    feasible_samples_per_scenario > 0 || error("feasible_samples_per_scenario must be positive.")
    max_attempts_multiplier > 0 || error("max_attempts_multiplier must be positive.")

    dataset_case_dir = joinpath(root_dir, "datasets", case_name)
    mkpath(dataset_case_dir)

    max_attempts_per_scenario = feasible_samples_per_scenario * max_attempts_multiplier
    topology_plan = build_topology_plan(ts, num_scenarios, seed_base)

    D_infeasible = init_dataset(float_type)
    D_feasible = init_dataset(float_type)

    global_seed_counter = seed_start
    total_feasible_attempts = 0
    total_infeasible_attempts = 0
    topology_failures = 0
    completed_scenarios = 0
    partial_scenarios = 0
    failed_scenarios = 0

    for scenario in topology_plan
        scenario_id = Int(scenario["scenario_id"])

        if haskey(scenario, "topology_error")
            topology_failures += 1
            failed_scenarios += 1
            @warn "Scenario $scenario_id skipped: $(scenario["topology_error"])"
            continue
        end

        branch_status = scenario["branch_status"]
        changing_switch_status = scenario["changing_switch_status"]
        scenario_is_radial = Bool(get(scenario, "is_radial", false))

        if haskey(scenario, "topology_meta")
            log_topology_scenario_start!(
                ts,
                scenario["topology_meta"];
                scenario_id=scenario_id,
                topology_seed=get(scenario, "topology_seed", nothing),
            )
        end

        attempts = 0
        infeasible_count = 0
        feasible_count = 0

        update_branch_status!(network, branch_status)
        scenario_warm_start = compute_warm_start(network)
        opf_cache = build_opf_cache(
            network,
            Ipopt.Optimizer;
            warm_start=scenario_warm_start,
        )

        while attempts < max_attempts_per_scenario
            need_infeasible = infeasible_samples_per_scenario !== nothing && infeasible_count < infeasible_samples_per_scenario
            need_feasible = feasible_count < feasible_samples_per_scenario
            (need_infeasible || need_feasible) || break

            seed = global_seed_counter
            global_seed_counter += 1
            attempts += 1

            rng = MersenneTwister(seed)
            pd, qd = Random.rand(rng, ls)
            pder_avai = Random.rand(rng, ds)
            obj_weights = Random.rand(rng, os)
            transform_profile!(pd, qd, pder_avai, seed)

            _set_loads!(network, pd, qd)
            _set_DERs!(network, pder_avai)
            _set_obj_weights!(network, obj_weights)

            res = solve_cached_opf!(opf_cache, network)

            println(
                "Scenario: $scenario_id, seed: $seed, " *
                "build_time: $(round(res["meta"]["build_time"], digits=4))s, " *
                "update_time: $(round(res["meta"]["update_time"], digits=4))s, " *
                "solve_time: $(round(res["meta"]["solve_time"], digits=4))s, " *
                "termination: $(res["meta"]["termination_status"])",
            )

            is_feasible = is_opf_feasible(res["meta"])
            is_infeasible = is_opf_infeasible(res["meta"])
            if is_feasible
                total_feasible_attempts += 1
            elseif is_infeasible
                total_infeasible_attempts += 1
            end

            if scenario_id == 1 && attempts == 1
                @info "Doing warm-up. Skip storing!"
                continue
            end

            if is_infeasible
                append_sample!(
                    D_infeasible,
                    scenario_id,
                    seed,
                    pd,
                    qd,
                    pder_avai,
                    changing_switch_status,
                    scenario_is_radial,
                    obj_weights,
                    res,
                )
                infeasible_count += 1
            end

            if is_feasible && need_feasible
                append_sample!(
                    D_feasible,
                    scenario_id,
                    seed,
                    pd,
                    qd,
                    pder_avai,
                    changing_switch_status,
                    scenario_is_radial,
                    obj_weights,
                    res,
                )
                feasible_count += 1
            end
        end

        target_infeasible = infeasible_samples_per_scenario === nothing ? 0 : infeasible_samples_per_scenario
        scenario_status = if infeasible_count == target_infeasible && feasible_count == feasible_samples_per_scenario
            "complete"
        elseif infeasible_count == 0 && feasible_count == 0
            "failed"
        else
            "partial"
        end

        if scenario_status == "complete"
            completed_scenarios += 1
        elseif scenario_status == "partial"
            partial_scenarios += 1
        else
            failed_scenarios += 1
        end

        if feasible_count < feasible_samples_per_scenario
            @warn "Scenario $scenario_id produced only $feasible_count/$feasible_samples_per_scenario feasible samples after $attempts attempts."
        end
    end

    infeasible_path = joinpath(dataset_case_dir, infeasible_filename)
    feasible_path = joinpath(dataset_case_dir, feasible_filename)

    dataset_metadata = Dict{String, Any}(
        "scenario_mode" => true,
        "scenario_variant" => variant_label,
        "scenario_count" => num_scenarios,
        "feasible_samples_per_scenario_target" => feasible_samples_per_scenario,
        "infeasible_samples_per_scenario_target" => infeasible_samples_per_scenario,
        "max_attempts_multiplier" => max_attempts_multiplier,
    )

    save_dataset_to_h5(
        D_infeasible,
        infeasible_path;
        float_type=float_type,
        network=network,
        ts=ts,
        topology_cfg=topology_cfg,
        metadata_extra=merge(copy(dataset_metadata), Dict("dataset_kind" => "infeasible_only")),
    )
    save_dataset_to_h5(
        D_feasible,
        feasible_path;
        float_type=float_type,
        network=network,
        ts=ts,
        topology_cfg=topology_cfg,
        metadata_extra=merge(copy(dataset_metadata), Dict("dataset_kind" => "feasible_only")),
    )

    return Dict{String, Any}(
        "completed_scenarios" => completed_scenarios,
        "partial_scenarios" => partial_scenarios,
        "failed_scenarios" => failed_scenarios,
        "topology_failures" => topology_failures,
        "feasible_attempts" => total_feasible_attempts,
        "infeasible_attempts" => total_infeasible_attempts,
        "infeasible_samples" => length(D_infeasible["input"]["seed"]),
        "feasible_samples" => length(D_feasible["input"]["seed"]),
        "feasible_path" => feasible_path,
        "infeasible_path" => infeasible_path,
    )
end

function save_network_config!(
    network,
    ts::TopologySampler,
    save_path::AbstractString,
    base_branch_status::Vector{Bool},
)
    mkpath(dirname(save_path))
    length(base_branch_status) == length(network.branch_status) || error(
        "base_branch_status length $(length(base_branch_status)) does not match network branch count $(length(network.branch_status)).",
    )
    update_branch_status!(network, copy(base_branch_status))

    network_dict = network_to_json_dict(network)
    network_dict["changing_switches"] = copy(changing_switches(ts))
    network_dict["topology_policy"] = Dict(
        "schema_version" => ts.policy.schema_version,
        "perturb_sw" => ts.policy.perturb_sw,
        "repair_nc" => ts.policy.repair_nc,
        "repair_hints" => ts.policy.repair_hints,
    )
    network_dict["topology_sampler"] = Dict(
        "flip_probability" => ts.flip_probability,
        "radial_ratio" => ts.radial_ratio,
    )
    open(save_path, "w") do f
        JSON.print(f, network_dict, 2)
    end

    return nothing
end
