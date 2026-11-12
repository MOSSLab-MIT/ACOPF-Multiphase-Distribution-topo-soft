# Economic objective weights live in the YAML configs, and config merge must keep floats.
# Run: julia --project=datagen_topo/FeederFlow.jl tests/test_config_obj_weights.jl
using Test

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const DATAGEN_DIR = joinpath(ROOT_DIR, "datagen_topo")

include(joinpath(DATAGEN_DIR, "datagen_topo.jl"))
include(joinpath(DATAGEN_DIR, "src", "generation", "training.jl"))
include(joinpath(DATAGEN_DIR, "src", "generation", "testing.jl"))

const CONFIG_DIR = joinpath(DATAGEN_DIR, "gen_scripts", "configs")
const WMIN = [0.5, 0.5]
const WMAX = [1.0, 10.0]

@testset "config obj weight ranges" begin
    paths = filter(p -> endswith(p, ".yaml"), readdir(CONFIG_DIR; join=true))
    @test !isempty(paths)
    for path in paths
        obj = require_obj_block(load_config(path))
        @test Float64.(obj["wmin"]) == WMIN
        @test Float64.(obj["wmax"]) == WMAX
    end
end

@testset "code defaults use the config ranges" begin
    for default in (DEFAULT_TRAINING_CONFIG, DEFAULT_TESTING_CONFIG)
        obj = require_obj_block(default)
        @test Float64.(obj["wmin"]) == WMIN
        @test Float64.(obj["wmax"]) == WMAX
        @test eltype(obj["wmin"]) == Float64
        @test eltype(obj["wmax"]) == Float64
    end
    @test_throws ErrorException require_obj_block(Dict{String, Any}())
    # A config that omits obj keeps the default range.
    bare = Dict{String, Any}("topology" => Dict{String, Any}("flip_probability" => 0.1))
    merged = deep_merge_config(DEFAULT_TRAINING_CONFIG, bare)
    @test Float64.(require_obj_block(merged)["wmin"]) == WMIN
    @test Float64.(require_obj_block(merged)["wmax"]) == WMAX
end

@testset "deep_merge keeps float overrides of integer vectors" begin
    # The old default inferred Dict{String, Vector{Int64}} and threw Int64(0.5).
    base = Dict("wmin" => [0, 0], "wmax" => [1, 10])
    @test base isa Dict{String, Vector{Int64}}
    override = Dict{String, Any}("wmin" => Any[0.5, 0.5], "wmax" => Any[1.0, 10.0])
    merged = deep_merge_config(base, override)
    @test Float64.(merged["wmin"]) == WMIN
    @test Float64.(merged["wmax"]) == WMAX

    for path in filter(p -> endswith(p, ".yaml"), readdir(CONFIG_DIR; join=true))
        merged_cfg = deep_merge_config(DEFAULT_TESTING_CONFIG, load_config(path))
        obj = require_obj_block(merged_cfg)
        @test Float64.(obj["wmin"]) == WMIN
        @test Float64.(obj["wmax"]) == WMAX
    end
end

@testset "load_config reads YAML with JSON-equivalent types" begin
    mktempdir() do dir
        path = joinpath(dir, "cfg.yaml")
        write(path, """
        # comment
        obj:
          wmin: [0.5, 0.5]
          wmax: [1.0, 10.0]
        penalty: {rho_min: [1, 1, 1, 100]}
        shift: {enabled: true}
        floating_point_type: Float64
        regimes:
          - {name: base, weight: 1.0}
        """)
        cfg = load_config(path)
        @test cfg isa Dict{String, Any}
        @test cfg["obj"] isa Dict{String, Any}
        @test cfg["obj"]["wmax"] == [1.0, 10.0] && all(x -> x isa Float64, cfg["obj"]["wmax"])
        @test all(x -> x isa Int, cfg["penalty"]["rho_min"])
        @test cfg["shift"]["enabled"] === true
        @test cfg["floating_point_type"] == "Float64"
        @test cfg["regimes"][1] isa Dict{String, Any} && cfg["regimes"][1]["weight"] === 1.0
        # Containers are untyped, as with JSON: a float can replace an integer entry.
        @test cfg["penalty"]["rho_min"] isa Vector{Any} && cfg["regimes"] isa Vector{Any}
        cfg["penalty"]["rho_min"][1] = 0.5
        @test cfg["penalty"]["rho_min"][1] === 0.5
        @test_throws ErrorException load_config(joinpath(dir, "cfg.json"))
        list_path = joinpath(dir, "list.yaml")
        write(list_path, "- 1\n- 2\n")
        @test_throws ErrorException load_config(list_path)
    end
end
