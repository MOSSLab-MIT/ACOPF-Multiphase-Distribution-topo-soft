# Training dataset generation: callable API and direct CLI entrypoint.

include(joinpath(@__DIR__, "..", "..", "datagen_topo.jl"))

using JSON
using Random

const DEFAULT_FLOAT_PRECISION = Float64

# Training load/DER regimes: edit weights and bounds below.

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/training.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --smin 1 --smax 50000

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/training.jl datagen_topo/grid_data/13_bus_test/IEEE13_test.dss --smin 1 --smax 50000

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/training.jl datagen_topo/grid_data/123_bus/IEEE123Master.dss --smin 1 --smax 50000
# julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/training.jl datagen_topo/grid_data/240_bus/240bus_Master.dss --smin 1 --smax 50000
# julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/training.jl datagen_topo/grid_data/906_bus/906bus_Master.dss --smin 1 --smax 50000
# Legacy positional: julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/training.jl "datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss" 1 50000
# Smoke (regime mix): julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/training.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --smin 1 --smax 300
# Regime variant: julia --project=datagen_topo/FeederFlow.jl datagen_topo/src/generation/training.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --smin 1 --smax 300 --load_l 0.2 --load_u 0.5 --der_frac_l 0.6 --der_frac_u 1.0 --output_suffix "low_load_high_der"

const REGIME_LOAD_DEFAULTS = Dict(
    "noise_type" => "ScaledUniform",
    "sigma" => 0.30,
)

const TRAINING_REGIMES = [
    Dict(
        "name" => "base",
        "weight" => 1.0,
        "load" => Dict("l" => 0.2, "u" => 1.5),
        "der" => Dict("frac_l" => 0.0, "frac_u" => 1.0, "sigma" => DEFAULT_DER_SIGMA),
    ),
    Dict(
        "name" => "high_load_low_der",
        "weight" => 0.0,
        "load" => Dict("l" => 0.7, "u" => 1.0),
        "der" => Dict("frac_l" => 0.0, "frac_u" => 0.3, "sigma" => DEFAULT_DER_SIGMA),
    ),
    Dict(
        "name" => "low_load_high_der",
        "weight" => 0.0,
        "load" => Dict("l" => 0.2, "u" => 0.5),
        "der" => Dict("frac_l" => 0.6, "frac_u" => 1.0, "sigma" => DEFAULT_DER_SIGMA),
    ),
]

const DEFAULT_TRAINING_CONFIG = Dict{String, Any}(
    "obj" => Dict{String, Any}(
        "wmin" => [0.5, 0.5],
        "wmax" => [1.0, 10.0],
    ),
    "penalty" => deepcopy(DEFAULT_PENALTY_SAMPLING),
    "cost" => deepcopy(DEFAULT_COST_CONFIG),
    "topology" => Dict{String, Any}(
        "flip_probability" => 0.5,
        "radial_ratio" => 0.5,
    ),
    "randomness" => Dict(
        "smin" => 1,
        "smax" => 1000,
    ),
)


function generate_training_dataset!(args::Vector{String}=String[]; datagen_dir::AbstractString=normpath(joinpath(@__DIR__, "..", "..")))
    root_dir = datagen_dir
    config = deepcopy(DEFAULT_TRAINING_CONFIG)
    positional, flags = parse_cli_tokens(String.(args))

    config_raw = haskey(flags, "config") ? load_config(flags["config"]) : Dict{String, Any}()
    for key in ("obj", "penalty", "cost", "topology", "randomness")
        if haskey(config_raw, key)
            config[key] = deep_merge_config(config[key], config_raw[key])
        end
    end
    config["obj"] = require_obj_block(config)
    cost = resolve_cost_config!(config, flags)
    if haskey(config_raw, "floating_point_type")
        config["floating_point_type"] = config_raw["floating_point_type"]
    end

    grid_dir = resolve_grid_path(
        positional,
        root_dir,
        joinpath("grid_data", "13_bus", "IEEE13Nodeckt.dss"),
    )
    smin, smax = resolve_training_seed_range(positional, flags, config)
    output_suffix = get(flags, "output_suffix", string(get(config_raw, "output_suffix", "")))

    apply_common_sampling_flags!(config, flags)
    config["topology"]["case_path"] = grid_dir

    regime_load_defaults = get(config_raw, "regime_load_defaults", REGIME_LOAD_DEFAULTS)
    regime_defs = get(config_raw, "training_regimes", TRAINING_REGIMES)
    regimes = [deepcopy(r) for r in regime_defs]
    validate_regimes!(regimes, regime_load_defaults)
    apply_regime_cli_overrides!(regimes, flags)

    network = create_network_data(grid_dir)
    apply_cost_config!(network, cost)

    float_type = resolve_float_type(config, DEFAULT_FLOAT_PRECISION)
    @info "Floating-point data will be exported in `$(float_type)`"

    load_scalers, der_scalers = build_regime_scalers(network, regimes)
    os = ObjScaler(network, config["obj"], config["penalty"])
    ts = TopologySampler(network, config["topology"])

    case_name = network.case

    seed_values = collect(smin:smax)
    n_samples = length(seed_values)
    n_samples > 0 || error("No samples to generate (empty seed range).")

    regime_counts = allocate_regime_counts(n_samples, [r["weight"] for r in regimes])
    regime_schedule = round_robin_regime_schedule(regime_counts)
    base_regime = regimes[find_base_regime_index(regimes)]
    changing_sw = changing_switches(ts)

    suffix = output_suffix == "" ? "" : "_" * output_suffix

    @info "Generating instances for case $(case_name)\nSeed range: [$smin, $smax], output_suffix='$output_suffix'"
    @info "topology radial_ratio=$(config["topology"]["radial_ratio"]), flip_probability=$(config["topology"]["flip_probability"])"
    for (idx, regime) in enumerate(regimes)
        load = regime["load"]
        der = regime["der"]
        @info "Regime $(regime["name"])" index=idx target_weight=regime["weight"] count=regime_counts[idx] load_l=load["l"] load_u=load["u"] der_frac_l=der["frac_l"] der_frac_u=der["frac_u"] der_sigma=der["sigma"] load_sigma=load["sigma"]
    end

    n_net = network.n_net
    n_der = network.n_der_unit
    n_changing = length(changing_sw)

    pd_mat = Matrix{Float64}(undef, n_net, n_samples)
    qd_mat = Matrix{Float64}(undef, n_net, n_samples)
    pder_mat = Matrix{Float64}(undef, n_der, n_samples)
    changing_mat = Matrix{Bool}(undef, n_changing, n_samples)
    is_radial_vec = Vector{Bool}(undef, n_samples)
    obj_mat = Matrix{Float64}(undef, N_OBJ_WEIGHTS, n_samples)
    seed_vec = Vector{Int}(undef, n_samples)
    progress_every = max(1, n_samples ÷ 10)

    for (col, s) in enumerate(seed_values)
        rng = MersenneTwister(s)
        r_idx = regime_schedule[col]
        pd, qd = Random.rand(rng, load_scalers[r_idx])
        pder_avai = Random.rand(rng, der_scalers[r_idx])
        branch_status, topo_meta = sample_topology_connected(rng, ts)
        changing_status = changing_switch_status(ts.policy, branch_status)
        obj_weights = Random.rand(rng, os)

        seed_vec[col] = s
        pd_mat[:, col] = pd
        qd_mat[:, col] = qd
        pder_mat[:, col] = pder_avai
        changing_mat[:, col] = changing_status
        is_radial_vec[col] = Bool(topo_meta[:is_radial])
        obj_mat[:, col] = obj_weights

        if col == 1 || col == n_samples || col % progress_every == 0
            @info "Sample progress" completed=col total=n_samples
        end
    end

    regime_id_vec = regime_schedule

    @info "All instances completed."

    D = Dict{String, Any}(
        "input" => Dict{String, Any}(
            "seed" => seed_vec,
            "regime_id" => regime_id_vec,
            "pd" => pd_mat,
            "qd" => qd_mat,
            "pder_avai" => pder_mat,
            "changing_switch_status" => changing_mat,
            "is_radial" => is_radial_vec,
            "obj_weights" => obj_mat,
        ),
    )

    regimes_json = JSON.json([
        Dict("name" => r["name"], "weight" => r["weight"], "load" => r["load"], "der" => r["der"])
        for r in regimes
    ])

    save_path = joinpath(root_dir, "datasets", case_name, case_name * "_training" * suffix * ".h5")
    save_training_dataset_to_h5(
        D,
        save_path;
        float_type=float_type,
        network=network,
        ts=ts,
        base_regime=base_regime,
        regimes_json=regimes_json,
        regime_counts=regime_counts,
    )
    @info "Dataset saved to $save_path"
    @info "All data saved successfully. Done!"

    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    generate_training_dataset!(String.(ARGS))
end
