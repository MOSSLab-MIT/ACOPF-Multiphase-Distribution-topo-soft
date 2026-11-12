include(joinpath(@__DIR__, "..", "datagen_topo.jl"))
include(joinpath(@__DIR__, "..", "src", "generation", "training.jl"))

generate_training_dataset!(String.(ARGS))

# Training load/DER regimes: edit weights and bounds below.

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_training_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --smin 1 --smax 50000

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_training_dataset.jl datagen_topo/grid_data/13_bus_test/IEEE13_test.dss --smin 1 --smax 50000

# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_training_dataset.jl datagen_topo/grid_data/123_bus/IEEE123Master.dss --smin 1 --smax 50000
# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_training_dataset.jl datagen_topo/grid_data/240_bus/240bus_Master.dss --smin 1 --smax 50000
# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_training_dataset.jl datagen_topo/grid_data/906_bus/906bus_Master.dss --smin 1 --smax 50000
# Legacy positional: julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_training_dataset.jl "datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss" 1 50000
# Smoke (regime mix): julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_training_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --smin 1 --smax 300