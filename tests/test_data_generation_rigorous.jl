using Test
using Random
using LinearAlgebra
using HDF5
using Ipopt

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const DATAGEN_DIR = joinpath(ROOT_DIR, "datagen_topo")

include(joinpath(DATAGEN_DIR, "datagen_topo.jl"))

const CASE_PATHS = Dict(
    "IEEE13Nodeckt" => joinpath(DATAGEN_DIR, "grid_data", "13_bus", "IEEE13Nodeckt.dss"),
    "IEEE123Master" => joinpath(DATAGEN_DIR, "grid_data", "123_bus", "IEEE123Master.dss"),
)

function warm_start_relative_residual(data::OPFData, u_start)
    v_slack_local = data.v_sub_ref ./ data.v_scale_slack
    mismatch = data.Y_nn * u_start + data.Y_ns * v_slack_local
    rhs_norm = max(norm(data.Y_ns * v_slack_local), 1.0)
    return norm(mismatch) / rhs_norm
end

function run_tiny_generation_case(case_path::AbstractString)
    network = create_network_data(case_path)

    load_cfg = Dict(
        "noise_type" => "ScaledUniform",
        "l" => 0.60,
        "u" => 0.61,
        "sigma" => 1e-3,
    )
    obj_cfg = Dict(
        "wmin" => [1.0, 1.0],
        "wmax" => [1.0, 1.0],
    )
    topology_cfg = Dict(
        "flip_probability" => 0.0,
        "radial_ratio" => 0.0,
        "case_path" => case_path,
    )

    ls = LoadScaler(network, load_cfg)
    ds = DERScaler(network)
    os = ObjScaler(network, obj_cfg)
    ts = TopologySampler(network, topology_cfg)

    temp_root = mktempdir()
    case_name = network.case
    infeasible_filename = case_name * "_tiny_infeasible.h5"
    feasible_filename = case_name * "_tiny_feasible.h5"

    transform_profile! = (pd, qd, pder_avai, seed) -> nothing

    summary = run_scenario_generation!(;
        network=network,
        ts=ts,
        ls=ls,
        ds=ds,
        os=os,
        float_type=Float32,
        root_dir=temp_root,
        case_name=case_name,
        topology_cfg=topology_cfg,
        num_scenarios=1,
        feasible_samples_per_scenario=1,
        max_attempts_multiplier=10,
        seed_base=1234,
        seed_start=1,
        variant_label="rigorous_test",
        infeasible_filename=infeasible_filename,
        feasible_filename=feasible_filename,
        transform_profile! = transform_profile!,
    )

    return (
        summary=summary,
        infeasible_path=summary["infeasible_path"],
        feasible_path=summary["feasible_path"],
        ts=ts,
    )
end

@testset "Warm-start topology-aware behavior" begin
    for (case_name, case_path) in CASE_PATHS
        @testset "Warm-start refresh and residual: $case_name" begin
            data = create_network_data(case_path)

            # Deliberately corrupt warm-start blocks; compute_warm_start must refresh them.
            fill!(data.Y_nn, 0.0 + 0.0im)
            fill!(data.Y_ns, 0.0 + 0.0im)

            u_start = compute_warm_start(data)
            @test length(u_start) == data.n_net
            @test norm(data.Y_nn) > 0.0
            @test norm(data.Y_ns) > 0.0

            rel_res = warm_start_relative_residual(data, u_start)
            @test isfinite(rel_res)
            @test rel_res <= WARM_START_MAX_REL_RESIDUAL
        end
    end

    @testset "Topology perturbation updates warm-start" begin
        data = nothing
        for case_path in values(CASE_PATHS)
            candidate = create_network_data(case_path)
            if !isempty(candidate.switch_idx)
                data = candidate
                break
            end
        end

        if data === nothing
            @test_skip "No switch lines available in configured feeder set."
            return
        end

        @test !isempty(data.switch_idx)

        original_status = copy(data.branch_status)
        line_idx = data.switch_idx[1]

        toggled_status = copy(original_status)
        toggled_status[line_idx] = !toggled_status[line_idx]
        update_branch_status!(data, toggled_status)

        u_start = compute_warm_start(data)
        rel_res = warm_start_relative_residual(data, u_start)
        @test rel_res <= WARM_START_MAX_REL_RESIDUAL
        @test data.branch_status[line_idx] == toggled_status[line_idx]
        @test data.switch_is_closed[1] == toggled_status[line_idx]

        switch_name = first(name for (name, idx) in data.switch_name_to_idx if idx == 1)
        current_states = query_switch_states(data)
        @test haskey(current_states, switch_name)
        @test current_states[switch_name] == data.switch_is_closed[1]

        update_branch_status!(data, original_status)
        u_restored = compute_warm_start(data)
        rel_res_restored = warm_start_relative_residual(data, u_restored)
        @test rel_res_restored <= WARM_START_MAX_REL_RESIDUAL
    end

    @testset "Non-switch status mutation is rejected" begin
        data = create_network_data(CASE_PATHS["IEEE13Nodeckt"])
        all_lines = collect(eachindex(data.branch_status))
        non_switch_lines = setdiff(all_lines, data.switch_idx)
        @test !isempty(non_switch_lines)

        bad_status = copy(data.branch_status)
        bad_idx = first(non_switch_lines)
        bad_status[bad_idx] = !bad_status[bad_idx]

        @test_throws ArgumentError update_branch_status!(data, bad_status)
    end
end

@testset "Rigorous tiny generation across 13/37/123 feeders" begin
    for (case_name, case_path) in CASE_PATHS
        @testset "Tiny scenario generation: $case_name" begin
            result = run_tiny_generation_case(case_path)
            summary = result.summary

            @test isfile(result.feasible_path)
            @test summary["failed_scenarios"] == 0
            @test summary["completed_scenarios"] == 1
            @test summary["feasible_samples"] == 1

            h5open(result.feasible_path, "r") do file
                @test haskey(file, "input/changing_switch_status") || haskey(file, "input/branch_status")

                pd = read(file["input/pd"])
                qd = read(file["input/qd"])
                pder_avai = read(file["input/pder_avai"])
                topo_key = haskey(file, "input/changing_switch_status") ? "changing_switch_status" : "branch_status"
                branch_status = read(file["input/$topo_key"])
                n_net = Int(read(file["metadata/n_net"]))
                n_der_unit = Int(read(file["metadata/n_der_unit"]))
                n_switch_total = Int(read(file["metadata/n_switch"]))
                meta_key = haskey(file, "metadata/changing_switches") ? "changing_switches" : "switchable_lines"
                changing_lines = read(file["metadata/$meta_key"])
                n_samples_feasible = Int(summary["feasible_samples"])

                @test ndims(pd) == 2
                @test size(pd, 1) == n_net
                @test size(pd, 2) == n_samples_feasible
                @test size(qd) == size(pd)

                @test ndims(pder_avai) == 2
                @test size(pder_avai, 1) == n_der_unit
                @test size(pder_avai, 2) == n_samples_feasible

                if !isempty(changing_lines)
                    @test ndims(branch_status) == 2
                    @test size(branch_status, 1) == length(changing_lines)
                    @test size(branch_status, 2) == n_samples_feasible
                    compact_meta = haskey(file, "metadata/topology_status_is_compact") ?
                        "topology_status_is_compact" : "topology_branch_status_is_compact"
                    @test Bool(read(file["metadata/$compact_meta"]))
                end
                @test n_switch_total >= length(changing_lines)
                @test length(changing_switches(result.ts)) == length(changing_lines)

                vr = read(file["primal/vr"])
                vi = read(file["primal/vi"])

                @test ndims(vr) == 2
                @test size(vr, 1) == n_net
                @test size(vr, 2) == n_samples_feasible
                @test size(vi) == size(vr)

                # Cached JuMP model: per-scenario build is amortized; stored samples
                # after the warmup solve should report zero rebuild time.
                @test haskey(file, "meta/build_time")
                @test haskey(file, "meta/update_time")
                build_times = read(file["meta/build_time"])
                update_times = read(file["meta/update_time"])
                @test length(build_times) == n_samples_feasible
                @test length(update_times) == n_samples_feasible
                @test all(build_times .>= 0)
                @test all(update_times .>= 0)
                if n_samples_feasible >= 1
                    @test maximum(build_times) < 5.0
                    @test maximum(update_times) < 1.0
                end
            end
        end
    end
end
