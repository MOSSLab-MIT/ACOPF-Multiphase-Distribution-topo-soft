include(joinpath(@__DIR__, "..", "datagen_topo.jl"))
include(joinpath(@__DIR__, "..", "src", "generation", "testing.jl"))

generate_testing_dataset!(String.(ARGS))

# Examples: see datagen_topo/gen_scripts/USAGE.md and the header of src/generation/testing.jl.
# julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/create_testing_dataset.jl datagen_topo/grid_data/13_bus/IEEE13Nodeckt.dss --config datagen_topo/gen_scripts/configs/testing_defaults.yaml
