# Generation entrypoint extracted from gen_scripts/create_testing_dataset.jl.

using JSON
using Random

const DEFAULT_FLOAT_PRECISION = Float64

# Examples:
# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 --output_layout scenario+flat  --load_l 0.3 --load_u 1.5

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 --output_layout scenario+flat  --load_l 0.2 --load_u 0.4 --der_frac_l 0.7 --der_frac_u 1.0 --output_suffix "low_load_high_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 --output_layout scenario+flat  --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 --output_suffix "high_load_low_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus_test/IEEE13_test.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus_test/IEEE13_test.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 --output_layout scenario+flat  --load_l 0.2 --load_u 0.4 --der_frac_l 0.7 --der_frac_u 1.0 --output_suffix "low_load_high_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus_test/IEEE13_test.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 --output_layout scenario+flat  --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 --output_suffix "high_load_low_der"



# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/123_bus/IEEE123Master.dss --num_scenarios 10 --feasible_samples_per_scenario 10 --max_attempts_multiplier 1

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/123_bus/IEEE123Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 1 --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 --output_suffix "high_load_low_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/123_bus/IEEE123Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 1 --load_l 0.2 --load_u 0.5 --der_frac_l 0.6 --der_frac_u 1.0 --output_suffix "low_load_high_der"


# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/240_bus/240bus_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 5 --output_layout scenario+flat --load_l 0.2 --load_u 1.5

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/240_bus/240bus_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 5 --output_layout scenario+flat --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 --output_suffix "high_load_low_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/240_bus/240bus_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 5 --output_layout scenario+flat --load_l 0.2 --load_u 0.5 --der_frac_l 0.6 --der_frac_u 1.0 --output_suffix "low_load_high_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/906_bus/906bus_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 5 --output_layout scenario+flat --load_l 0.2 --load_u 1.5

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/906_bus/906bus_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 5 --output_layout scenario+flat --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 --output_suffix "high_load_low_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/906_bus/906bus_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 5 --output_layout scenario+flat --load_l 0.2 --load_u 0.5 --der_frac_l 0.6 --der_frac_u 1.0 --output_suffix "low_load_high_der"


# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/IEEE8500/8500_Master.dss --num_scenarios 5 --feasible_samples_per_scenario 10 --max_attempts_multiplier 5 --output_layout flat_only

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/IEEE8500/8500_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 10 --max_attempts_multiplier 5 --output_layout flat_only --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 --output_suffix "high_load_low_der" --save_config false

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/IEEE8500/8500_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 10 --max_attempts_multiplier 5 --output_layout flat_only --load_l 0.2 --load_u 0.5 --der_frac_l 0.6 --der_frac_u 1.0 --output_suffix "low_load_high_der" --save_config false


const DEFAULT_TESTING_CONFIG = Dict(
    "load" => Dict(
        "noise_type" => "ScaledUniform",
        "l" => 0.2,
        "u" => 1.5,
        "sigma" => 0.30,
    ),
    "obj" => Dict{String, Any}(
        "wmin" => [0.5, 0.5],
        "wmax" => [1.0, 10.0],
    ),
    "penalty" => deepcopy(DEFAULT_PENALTY_SAMPLING),
    "der" => Dict(
        "frac_l" => 0.0,
        "frac_u" => 1.0,
        "sigma" => DEFAULT_DER_SIGMA,
    ),
    "topology" => Dict{String, Any}(
        "flip_probability" => 0.5,
        "radial_ratio" => 0.5,
    ),
    "randomness" => Dict(
        "seed_base" => 100000,
        "smin" => 2_000_001,
    ),
    "scenario" => Dict(
        "num_scenarios" => 10,
        "feasible_samples_per_scenario" => 100,
        "max_attempts_multiplier" => 5,
    ),
    "shift" => Dict(
        "enabled" => false,
        "noise_type" => "gaussian",    # gaussian | additive
        "sigma_rel" => 0.2,
        "der_sigma" => 0.2,
        "mu_rel" => 0.1,
        "der_mu" => 0.2,
        "scale_factor" => 1.0,
        "random_sign" => true,
    ),
    "save_config" => true,
    "output_suffix" => "",
)

const VALID_VARIANTS = ("base", "shifted")

# `--variant` wins; otherwise the config's `shift.enabled` selects base/shifted.
function parse_variant(flags::Dict{String, String}, config::AbstractDict)
    config_variant = config["shift"]["enabled"] ? "shifted" : "base"
    variant = lowercase(strip(get(flags, "variant", config_variant)))
    variant in VALID_VARIANTS || error("Invalid variant '$variant'. Use one of: base, shifted.")
    return variant
end


function generate_testing_dataset!(args::Vector{String}=String[]; datagen_dir::AbstractString=normpath(joinpath(@__DIR__, "..", "..")))
    root_dir = datagen_dir
    config = deepcopy(DEFAULT_TESTING_CONFIG)
    positional, flags = parse_cli_tokens(String.(args))
    if haskey(flags, "config")
        config = deep_merge_config(config, load_json_config(flags["config"]))
    end
    config["obj"] = require_obj_block(config)

    grid_dir = length(positional) >= 1 ? positional[1] : joinpath(root_dir, "grid_data", "13_bus", "IEEE13Nodeckt.dss")

    variant = parse_variant(flags, config)
    seed_base = parse(Int, get(flags, "seed_base", string(config["randomness"]["seed_base"])))
    seed_start = parse(Int, get(flags, "seed_start", string(config["randomness"]["smin"])))
    num_scenarios = parse(Int, get(flags, "num_scenarios", string(config["scenario"]["num_scenarios"])))
    feasible_samples_per_scenario = parse(Int, get(flags, "feasible_samples_per_scenario", string(config["scenario"]["feasible_samples_per_scenario"])))
    max_attempts_multiplier = parse(Int, get(flags, "max_attempts_multiplier", string(config["scenario"]["max_attempts_multiplier"])))
    scenario_mode = parse_bool(get(flags, "scenario_mode", "true"))
    output_suffix = get(flags, "output_suffix", string(config["output_suffix"]))
    config["save_config"] = parse_bool(get(flags, "save_config", string(config["save_config"])))
    shift_mu = parse(Float64, get(flags, "shift_mu", string(config["shift"]["mu_rel"])))
    shift_der_mu = parse(Float64, get(flags, "shift_der_mu", string(config["shift"]["der_mu"])))

    if !scenario_mode
        error("scenario_mode=false is no longer supported by this generator.")
    end

    if length(positional) >= 3
        legacy_smin = parse(Int, positional[2])
        legacy_smax = parse(Int, positional[3])
        seed_start = legacy_smin
        if !haskey(flags, "num_scenarios") && !haskey(flags, "feasible_samples_per_scenario")
            num_scenarios = 1
            feasible_samples_per_scenario = max(legacy_smax - legacy_smin + 1, 1)
            @warn "Detected legacy positional range [$legacy_smin, $legacy_smax]. Mapping to num_scenarios=1 and feasible_samples_per_scenario=$feasible_samples_per_scenario."
        end
    end

    if variant == "shifted"
        if length(positional) >= 4 && !haskey(flags, "shift_mu")
            shift_mu = parse(Float64, positional[4])
        end
        if length(positional) >= 5 && !haskey(flags, "shift_der_mu")
            shift_der_mu = parse(Float64, positional[5])
        end
        if length(positional) >= 6 && !haskey(flags, "output_suffix")
            output_suffix = positional[6]
        end
    end

    config["shift"]["enabled"] = variant == "shifted"
    config["shift"]["mu_rel"] = shift_mu
    config["shift"]["der_mu"] = shift_der_mu
    config["shift"]["random_sign"] = parse_bool(
        get(flags, "shift_random_sign", string(get(config["shift"], "random_sign", true))),
    )
    apply_common_sampling_flags!(config, flags)
    config["topology"]["case_path"] = grid_dir

    network = create_network_data(grid_dir)

    base_branch_status = copy(network.branch_status)

    float_type = resolve_float_type(config, DEFAULT_FLOAT_PRECISION)
    @info "Floating-point data will be exported in `$(float_type)`"

    ls = LoadScaler(network, config["load"])
    ds = DERScaler(network, config["der"])
    os = ObjScaler(network, config["obj"], config["penalty"])
    ts = TopologySampler(network, config["topology"])

    case_name = network.case
    suffix = output_suffix == "" ? "" : "_" * output_suffix
    if variant == "shifted"
        variant_label = output_suffix == "" ? "shifted" : "shifted_" * output_suffix
        infeasible_filename = case_name * "_test_infeasible_shifted" * suffix * ".h5"
        feasible_filename = case_name * "_test_feasible_shifted" * suffix * ".h5"
        load_cap_ratio = shift_load_cap_ratio(config["load"])
        transform_profile! = (pd, qd, pder_avai, seed) -> apply_shift!(
            pd, qd, pder_avai, seed, config["shift"];
            pd_ref=ls.pd_ref, qd_ref=ls.qd_ref, der_rated=ds.rated, load_cap_ratio=load_cap_ratio,
        )
    else
        variant_label = output_suffix == "" ? "base" : "base_" * output_suffix
        infeasible_filename = case_name * "_test_infeasible" * suffix * ".h5"
        feasible_filename = case_name * "_test_feasible" * suffix * ".h5"
        transform_profile! = (pd, qd, pder_avai, seed) -> nothing
    end

    @info "Generating $(variant) TEST dataset for case $(case_name)"
    @info "Num_scenarios=$num_scenarios, feasible_samples_per_scenario=$feasible_samples_per_scenario, max_attempts_multiplier=$max_attempts_multiplier"
    @info "seed_base=$seed_base, seed_start=$seed_start, output_suffix='$output_suffix', save_config=$(config["save_config"])"
    if variant == "shifted"
        @info "shift enabled=$(config["shift"]["enabled"]), noise_type=$(config["shift"]["noise_type"]), mu_rel=$(config["shift"]["mu_rel"]), sigma_rel=$(config["shift"]["sigma_rel"]), der_mu=$(config["shift"]["der_mu"]), der_sigma=$(config["shift"]["der_sigma"]), random_sign=$(config["shift"]["random_sign"])"
    end
    @info "Load scale [$(config["load"]["l"]), $(config["load"]["u"])], load sigma $(config["load"]["sigma"]), DER availability factor [$(config["der"]["frac_l"]), $(config["der"]["frac_u"])], DER sigma $(get(config["der"], "sigma", DEFAULT_DER_SIGMA))"
    @info "Topology radial_ratio=$(config["topology"]["radial_ratio"]), flip_probability=$(config["topology"]["flip_probability"])"

    summary = run_scenario_generation!(;
        network=network,
        ts=ts,
        ls=ls,
        ds=ds,
        os=os,
        float_type=float_type,
        root_dir=root_dir,
        case_name=case_name,
        topology_cfg=config["topology"],
        num_scenarios=num_scenarios,
        feasible_samples_per_scenario=feasible_samples_per_scenario,
        max_attempts_multiplier=max_attempts_multiplier,
        seed_base=seed_base,
        seed_start=seed_start,
        variant_label=variant_label,
        infeasible_filename=infeasible_filename,
        feasible_filename=feasible_filename,
        transform_profile! = transform_profile!,
    )

    network_config_name = output_suffix == "" ? "network_config.json" : "network_config_" * output_suffix * ".json"
    save_path_json = joinpath(root_dir, "datasets", case_name, network_config_name)
    if config["save_config"]
        save_network_config!(network, ts, save_path_json, base_branch_status)
        @info "Network config saved to $save_path_json"
    end

    @info "Feasible file: $(feasible_filename)"
    @info "Infeasible file: $(infeasible_filename)"
    @info "Summary: $summary"
    @info "All data saved successfully. Done!"

    return nothing
end
