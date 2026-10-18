using HDF5

abstract type DatagenH5Schema end

struct ScenarioH5Schema <: DatagenH5Schema
    metadata_extra::Dict{String, Any}
end

struct TrainingH5Schema <: DatagenH5Schema
    base_regime::Dict
    regimes_json::String
    regime_counts
end

function tensorize(V::Vector{T}) where {T}
    length(V) > 0 || error("Trying to tensorize an empty collection")
    return copy(V)
end

function tensorize(V::Vector{Array{T,N}}) where {T,N}
    length(V) > 0 || error("Trying to tensorize an empty collection")
    return stack(V)
end

"""Pass through arrays that are already batched (e.g. preallocated sample matrices)."""
function maybe_tensorize(V)
    if V isa AbstractArray && ndims(V) > 1
        return V
    elseif V isa Vector
        return tensorize(V)
    else
        return V
    end
end

function convert_float_data(D, F)
    F <: AbstractFloat || error("`F` must be a subtype of `AbstractFloat`")
    keytype(D) <: AbstractString || error("Only `AbstractString`-like keys are supported.")

    d = Dict{String, Any}()

    for (k, v) in D
        if v isa Dict
            d[k] = convert_float_data(v, F)
        elseif v isa AbstractFloat
            d[k] = convert(F, v)
        elseif v isa (Array{T,N} where {T <: AbstractFloat, N})
            d[k] = convert(Array{F,ndims(v)}, v)
        else
            d[k] = v
        end
    end

    return d
end

function init_dataset(float_type)
    return Dict{String, Any}(
        "input" => Dict{String, Any}(
            "seed" => Int[],
            "scenario_index" => Int[],
            "pd" => Vector{Float64}[],
            "qd" => Vector{Float64}[],
            "pder_avai" => Vector{Float64}[],
            "changing_switch_status" => Vector{Bool}[],
            "is_radial" => Bool[],
            "obj_weights" => Vector{Float64}[],
        ),
        "primal" => Dict{String, Any}(
            "vr" => Vector{float_type}[],
            "vi" => Vector{float_type}[],
            "psub" => float_type[],
            "qsub" => float_type[],
            "psub_phase" => Vector{float_type}[],
            "qsub_phase" => Vector{float_type}[],
            "pder" => Vector{float_type}[],
            "qder" => Vector{float_type}[],
        ),
        "meta" => Dict{String, Any}(
            "termination_status" => String[],
            "primal_status" => String[],
            "dual_status" => String[],
            "solve_time" => Float32[],
            "primal_objective_value" => float_type[],
            "dual_objective_value" => float_type[],
            "build_time" => Float32[],
            "update_time" => Float32[],
            "extract_time" => Float32[],
        ),
    )
end

function append_sample!(
    D::Dict{String, Any},
    scenario_idx::Int,
    seed::Int,
    pd,
    qd,
    pder_avai,
    changing_switch_status,
    is_radial,
    obj_weights,
    res::Dict{String, Any},
)
    push!(D["input"]["seed"], seed)
    push!(D["input"]["scenario_index"], scenario_idx)
    push!(D["input"]["pd"], pd)
    push!(D["input"]["qd"], qd)
    push!(D["input"]["pder_avai"], pder_avai)
    push!(D["input"]["changing_switch_status"], changing_switch_status)
    push!(D["input"]["is_radial"], is_radial)
    push!(D["input"]["obj_weights"], obj_weights)

    for (k, v) in res["primal"]
        push!(D["primal"][k], v)
    end

    for (k, v) in res["meta"]
        push!(D["meta"][k], v)
    end

    return nothing
end

function _tensorize_dataset(D::Dict{String, Any})
    D_tensorized = deepcopy(D)
    for (section, dict) in D_tensorized
        for (k2, v2) in dict
            D_tensorized[section][k2] = tensorize(v2)
        end
    end
    return D_tensorized
end

function _write_h5_section!(file, section::AbstractString, data::Dict)
    for (key, val) in data
        val_safe = val isa BitArray ? Array(val) : val
        write(file, "$section/$key", val_safe)
    end
    return nothing
end

function _write_common_metadata!(file, network, ts::TopologySampler)
    write(file, "metadata/n_net", network.n_net)
    write(file, "metadata/n_slk", network.n_slk)
    write(file, "metadata/n_total", network.n_total)
    write(file, "metadata/n_der_unit", network.n_der_unit)
    write(file, "metadata/n_branch", length(network.branch_status))
    write(file, "metadata/n_switch", length(network.switch_idx))
    write(file, "metadata/changing_switches", changing_switches(ts))
    write(file, "metadata/topology_flip_probability", ts.flip_probability)
    write(file, "metadata/topology_radial_ratio", ts.radial_ratio)
    write(file, "metadata/topology_status_is_compact", true)
    write(file, "metadata/topology_perturb_sw", ts.policy.perturb_sw)
    write(file, "metadata/topology_repair_nc", ts.policy.repair_nc)
    return nothing
end

function _write_schema_metadata!(file, schema::ScenarioH5Schema)
    for (k, v) in schema.metadata_extra
        v === nothing && continue
        v_safe = v isa BitArray ? Array(v) : v
        write(file, "metadata/$k", v_safe)
    end
    return nothing
end

function _write_schema_metadata!(file, schema::TrainingH5Schema)
    base_regime = schema.base_regime
    write(file, "metadata/load_l", base_regime["load"]["l"])
    write(file, "metadata/load_u", base_regime["load"]["u"])
    write(file, "metadata/der_frac_l", base_regime["der"]["frac_l"])
    write(file, "metadata/der_frac_u", base_regime["der"]["frac_u"])
    write(file, "metadata/regimes_json", schema.regimes_json)
    write(file, "metadata/regime_counts", schema.regime_counts)
    return nothing
end

function save_dataset_to_h5(
    D::Dict{String, Any},
    save_path::AbstractString;
    float_type,
    network,
    ts::TopologySampler,
    topology_cfg::Dict,
    metadata_extra::Dict{String, Any}=Dict{String, Any}(),
)
    if isempty(D["input"]["seed"])
        @warn "Skipping save to $save_path because dataset is empty."
        return false
    end

    D_float_type = convert_float_data(_tensorize_dataset(D), float_type)
    schema = ScenarioH5Schema(metadata_extra)
    mkpath(dirname(save_path))

    h5open(save_path, "w") do file
        _write_h5_section!(file, "input", D_float_type["input"])
        _write_h5_section!(file, "primal", D_float_type["primal"])
        _write_h5_section!(file, "meta", D_float_type["meta"])
        _write_common_metadata!(file, network, ts)
        _write_schema_metadata!(file, schema)
    end

    return true
end

function save_training_dataset_to_h5(
    D::Dict{String, Any},
    save_path::AbstractString;
    float_type,
    network,
    ts::TopologySampler,
    base_regime::Dict,
    regimes_json::AbstractString,
    regime_counts,
)
    schema = TrainingH5Schema(base_regime, String(regimes_json), regime_counts)
    D_tensorized = deepcopy(D)
    for (section, dict) in D_tensorized
        for (key, value) in dict
            D_tensorized[section][key] = maybe_tensorize(value)
        end
    end
    D_float_type = convert_float_data(D_tensorized, float_type)
    mkpath(dirname(save_path))

    h5open(save_path, "w") do file
        _write_h5_section!(file, "input", D_float_type["input"])
        _write_common_metadata!(file, network, ts)
        _write_schema_metadata!(file, schema)
    end

    return true
end
