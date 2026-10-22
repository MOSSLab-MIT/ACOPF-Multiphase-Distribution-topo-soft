# Generate soft-constraint OPF test datasets (feasible/infeasible HDF5 + network_config.json).
# Usage and flags: datagen_topo/gen_scripts/USAGE.md. Run from the repository root.

include(joinpath(@__DIR__, "..", "datagen_topo.jl"))
include(joinpath(@__DIR__, "..", "src", "generation", "testing.jl"))

generate_testing_dataset!(String.(ARGS))

# Examples (run from the repository root):
# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5  --load_l 0.2 --load_u 1.5

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5  --load_l 0.2 --load_u 0.4 --der_frac_l 0.7 --der_frac_u 1.0 --output_suffix "low_load_high_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5  --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 --output_suffix "high_load_low_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus_test/IEEE13_test.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus_test/IEEE13_test.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5  --load_l 0.2 --load_u 0.4 --der_frac_l 0.7 --der_frac_u 1.0 --output_suffix "low_load_high_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus_test/IEEE13_test.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5  --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 --output_suffix "high_load_low_der"



# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/123_bus/IEEE123Master.dss --num_scenarios 10 --feasible_samples_per_scenario 10 --max_attempts_multiplier 1

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/123_bus/IEEE123Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 1 --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 --output_suffix "high_load_low_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/123_bus/IEEE123Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 1 --load_l 0.2 --load_u 0.5 --der_frac_l 0.6 --der_frac_u 1.0 --output_suffix "low_load_high_der"


# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/240_bus/240bus_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 5 --load_l 0.2 --load_u 1.5

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/240_bus/240bus_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 5 --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 --output_suffix "high_load_low_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/240_bus/240bus_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 5 --load_l 0.2 --load_u 0.5 --der_frac_l 0.6 --der_frac_u 1.0 --output_suffix "low_load_high_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/906_bus/906bus_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 5 --load_l 0.2 --load_u 1.5

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/906_bus/906bus_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 5 --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 --output_suffix "high_load_low_der"

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/906_bus/906bus_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 50 --max_attempts_multiplier 5 --load_l 0.2 --load_u 0.5 --der_frac_l 0.6 --der_frac_u 1.0 --output_suffix "low_load_high_der"


# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/IEEE8500/8500_Master.dss --num_scenarios 5 --feasible_samples_per_scenario 10 --max_attempts_multiplier 5

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/IEEE8500/8500_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 10 --max_attempts_multiplier 5 --load_l 0.7 --load_u 1.0 --der_frac_l 0.0 --der_frac_u 0.3 --output_suffix "high_load_low_der" --save_config false

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/IEEE8500/8500_Master.dss --num_scenarios 10 --feasible_samples_per_scenario 10 --max_attempts_multiplier 5 --load_l 0.2 --load_u 0.5 --der_frac_l 0.6 --der_frac_u 1.0 --output_suffix "low_load_high_der" --save_config false


# shifted
# Examples:
# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 --load_l 0.2 --load_u 1.5 --variant shifted --shift_mu 0.1 --shift_der_mu 0.1 --output_suffix "m01_std01" --save_config false

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 --load_l 0.2 --load_u 1.5 --variant shifted --shift_mu 0.2 --shift_der_mu 0.2 --output_suffix "m02_std02" --save_config false

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 --load_l 0.2 --load_u 1.5 --variant shifted --shift_mu 0.3 --shift_der_mu 0.3 --output_suffix "m03_std03" --save_config false

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 --load_l 0.2 --load_u 1.5 --variant shifted --shift_mu 0.4 --shift_der_mu 0.4 --output_suffix "m04_std04" --save_config false

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --num_scenarios 10 --feasible_samples_per_scenario 100 --max_attempts_multiplier 5 --load_l 0.2 --load_u 1.5 --variant shifted --shift_mu 0.5 --shift_der_mu 0.5 --output_suffix "m05_std05" --save_config false


# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/906_bus/906bus_Master.dss --num_scenarios 2 --feasible_samples_per_scenario 2 --max_attempts_multiplier 3 --load_l 0.2 --load_u 1.5 --variant shifted --shift_mu 0.1 --shift_der_mu 0.1 --output_suffix m01_std01 --save_config false

# out-of-training-load-range (load scale [0.5, 2.0], extends above the training range [0.2, 1.5])
# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --config datagen_topo/gen_scripts/configs/testing_oor_load.json --output_suffix oor_load --save_config false