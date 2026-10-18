# Shared include entrypoint for datagen_topo.

if !isdefined(@__MODULE__, :DATAGEN_TOPO_INCLUDED)
    const DATAGEN_TOPO_INCLUDED = true

    include(joinpath(@__DIR__, "src", "opf", "data.jl"))
    include(joinpath(@__DIR__, "src", "opf", "topology_admittance.jl"))
    include(joinpath(@__DIR__, "src", "opf", "solve.jl"))
    include(joinpath(@__DIR__, "src", "opf", "model.jl"))

    include(joinpath(@__DIR__, "src", "topology", "policy.jl"))
    include(joinpath(@__DIR__, "src", "topology", "graph.jl"))

    include(joinpath(@__DIR__, "src", "sampling", "profile.jl"))
    include(joinpath(@__DIR__, "src", "sampling", "topology.jl"))
    include(joinpath(@__DIR__, "src", "topology", "log.jl"))
    include(joinpath(@__DIR__, "src", "sampling", "mutations.jl"))
    include(joinpath(@__DIR__, "src", "sampling", "shifts.jl"))

    include(joinpath(@__DIR__, "src", "io", "json_utils.jl"))
    include(joinpath(@__DIR__, "src", "io", "h5_schema.jl"))

    include(joinpath(@__DIR__, "src", "cli", "parse.jl"))
    include(joinpath(@__DIR__, "src", "cli", "defaults.jl"))

    include(joinpath(@__DIR__, "src", "generation", "topology_plan.jl"))
    include(joinpath(@__DIR__, "src", "generation", "scenario_runner.jl"))
end
