# Shared CLI/config helpers for datagen_topo/gen_scripts/*.jl.

using JSON

const LOAD_DER_CLI_FLAGS = ("load_l", "load_u", "der_frac_l", "der_frac_u")

function deep_merge_config(base::AbstractDict, override::AbstractDict)
    # Dict{String, Any}: a concretely typed base (e.g. Dict{String, Vector{Int64}})
    # would convert the override and throw InexactError on a float such as 0.5.
    merged = Dict{String, Any}()
    for (key, value) in base
        merged[String(key)] = deepcopy(value)
    end
    for (key, value) in override
        key_s = String(key)
        if haskey(merged, key_s) && merged[key_s] isa AbstractDict && value isa AbstractDict
            merged[key_s] = deep_merge_config(merged[key_s], value)
        else
            merged[key_s] = deepcopy(value)
        end
    end
    return merged
end

"""
    require_obj_block(config) -> Dict

Economic weights `obj.wmin` / `obj.wmax` (`w_cost`, `w_curtail`). The code default is `[0.5, 0.5]`–`[1.0, 10.0]`.
"""
function require_obj_block(config::AbstractDict)
    haskey(config, "obj") ||
        error("Config must include \"obj\" with \"wmin\" and \"wmax\".")
    obj = config["obj"]
    obj isa AbstractDict || error("Config \"obj\" must be an object.")
    for bound in ("wmin", "wmax")
        haskey(obj, bound) || error("Config obj is missing \"$bound\".")
        vec = obj[bound]
        vec isa AbstractVector || error("Config obj.$bound must be a vector.")
        length(vec) == N_ECON_WEIGHTS ||
            error("Config obj.$bound must have length $N_ECON_WEIGHTS; got $(length(vec)).")
    end
    return obj
end

function load_json_config(path::AbstractString)
    return JSON.parsefile(path; dicttype=Dict{String, Any})
end

"""
    apply_common_sampling_flags!(config, flags)

Update `config` topology/floating_point_type from CLI flags. Updates `config["load"]`/`config["der"]`
only when those keys exist (testing scripts); training uses `apply_regime_cli_overrides!` instead.
"""
function apply_common_sampling_flags!(config::Dict, flags::Dict{String, String})
    if haskey(config, "load")
        if !haskey(config, "der")
            config["der"] = Dict{String, Any}("frac_l" => 0.0, "frac_u" => 1.0)
        end
        apply_load_der_cli_flags!(config["load"], config["der"], flags)
    end
    config["topology"]["flip_probability"] = parse(
        Float64,
        get(flags, "flip_probability", string(config["topology"]["flip_probability"])),
    )
    config["topology"]["radial_ratio"] = parse(
        Float64,
        get(flags, "radial_ratio", string(config["topology"]["radial_ratio"])),
    )
    if haskey(flags, "floating_point_type")
        config["floating_point_type"] = flags["floating_point_type"]
    end
    return config
end

"""
    resolve_training_seed_range(positional, flags, config) -> (smin, smax)

Prefer `--smin` / `--smax`; fall back to legacy positional `[grid, smin, smax]`; else config defaults.
"""
function resolve_training_seed_range(
    positional::Vector{String},
    flags::Dict{String, String},
    config::Dict,
)
    rand_cfg = config["randomness"]
    smin = parse(Int, get(flags, "smin", string(rand_cfg["smin"])))
    smax = parse(Int, get(flags, "smax", string(get(rand_cfg, "smax", smin))))

    if length(positional) >= 3
        legacy_smin = parse(Int, positional[2])
        legacy_smax = parse(Int, positional[3])
        if !haskey(flags, "smin")
            smin = legacy_smin
        end
        if !haskey(flags, "smax")
            smax = legacy_smax
        end
    elseif length(positional) >= 2 && !haskey(flags, "smin")
        smin = parse(Int, positional[2])
    end

    smin <= smax || error("Invalid seed range: smin=$smin > smax=$smax")
    return smin, smax
end

function resolve_grid_path(positional::Vector{String}, root_dir::AbstractString, default_rel::AbstractString)
    if length(positional) >= 1
        return positional[1]
    end
    return joinpath(root_dir, default_rel)
end

function resolve_float_type(config::Dict, default_fp::Type)
    fp_string = get(config, "floating_point_type", string(default_fp))
    if lowercase(fp_string) == "float32"
        return Float32
    elseif lowercase(fp_string) == "float64"
        return Float64
    else
        error("Invalid floating-point type: $(fp_string); only `Float32` and `Float64` are supported.")
    end
end

function _validate_load_bounds!(load::Dict)
    l = Float64(get(load, "l", NaN))
    u = Float64(get(load, "u", NaN))
    isfinite(l) && isfinite(u) || error("Regime load bounds must be finite; got l=$l, u=$u")
    l <= u || error("Invalid regime load bounds: l=$l > u=$u")
    return nothing
end

function apply_load_der_cli_flags!(load::Dict, der::Dict, flags::Dict{String, String})
    if haskey(flags, "load_l")
        load["l"] = parse(Float64, flags["load_l"])
    end
    if haskey(flags, "load_u")
        load["u"] = parse(Float64, flags["load_u"])
    end
    if haskey(flags, "der_frac_l")
        der["frac_l"] = parse(Float64, flags["der_frac_l"])
    end
    if haskey(flags, "der_frac_u")
        der["frac_u"] = parse(Float64, flags["der_frac_u"])
    end
    return nothing
end

"""
    validate_regimes!(regimes, load_defaults)

Normalize training regimes: unique names, non-negative weights (at least one > 0),
merged load dicts, validated DER fractions. Mutates `regimes` in place.
"""
function validate_regimes!(regimes::Vector{<:Dict}, load_defaults::Dict)
    isempty(regimes) && error("TRAINING_REGIMES must contain at least one regime.")
    seen_names = Set{String}()
    any_positive = false
    for (idx, regime) in enumerate(regimes)
        name = String(get(regime, "name", ""))
        isempty(name) && error("Regime at index $idx is missing a non-empty \"name\".")
        name in seen_names && error("Duplicate regime name: $name")
        push!(seen_names, name)

        weight = Float64(get(regime, "weight", NaN))
        isfinite(weight) && weight >= 0 ||
            error("Regime \"$name\" must have a finite weight >= 0; got weight=$weight")
        weight > 0 && (any_positive = true)

        load_partial = get(regime, "load", Dict{String, Any}())
        regime["load"] = merge(copy(load_defaults), load_partial)
        _validate_load_bounds!(regime["load"])

        der_partial = get(regime, "der", Dict{String, Any}())
        frac_l, frac_u = _parse_der_frac_bounds(der_partial)
        regime["der"] = Dict{String, Any}("frac_l" => frac_l, "frac_u" => frac_u)
    end
    any_positive || error("At least one training regime must have weight > 0.")
    return regimes
end

"""
    apply_regime_cli_overrides!(regimes, flags)

Apply `--load_l`, `--load_u`, `--der_frac_l`, `--der_frac_u` uniformly to every training regime.
"""
function apply_regime_cli_overrides!(regimes::Vector{<:Dict}, flags::Dict{String, String})
    if any(haskey(flags, k) for k in LOAD_DER_CLI_FLAGS)
        @warn "CLI load/der flags override all training regimes uniformly."
        for regime in regimes
            apply_load_der_cli_flags!(regime["load"], regime["der"], flags)
            _validate_load_bounds!(regime["load"])
            frac_l, frac_u = _parse_der_frac_bounds(regime["der"])
            regime["der"] = Dict{String, Any}("frac_l" => frac_l, "frac_u" => frac_u)
        end
    end
    return regimes
end

"""
    allocate_regime_counts(n_samples, weights) -> Vector{Int}

Largest-remainder allocation of integer sample counts per regime (sum equals `n_samples`).
"""
function allocate_regime_counts(n_samples::Int, weights::AbstractVector{<:Real})
    n_samples >= 0 || error("n_samples must be non-negative; got $n_samples")
    isempty(weights) && error("weights must be non-empty")
    all(isfinite, weights) || error("weights must be finite")
    all(>=(0), weights) || error("weights must be non-negative")

    w = Float64.(weights)
    w_sum = sum(w)
    w_sum > 0 || error("sum(weights) must be > 0; got $w_sum")

    counts = zeros(Int, length(w))
    active = w .> 0
    w_active = w[active]
    w_active ./= sum(w_active)
    raw = zeros(Float64, length(w))
    raw[active] = w_active .* n_samples
    counts[active] = floor.(Int, raw[active])
    remainder = n_samples - sum(counts)
    if remainder > 0
        order = sortperm(raw[active] .- counts[active], rev=true)
        active_indices = findall(active)
        for i in 1:remainder
            counts[active_indices[order[i]]] += 1
        end
    end
    @assert sum(counts) == n_samples
    return counts
end

"""
    round_robin_regime_schedule(counts) -> Vector{Int}

Expand per-regime counts into a 1-based regime index schedule, interleaved round-robin.
"""
function round_robin_regime_schedule(counts::AbstractVector{<:Integer})
    n = sum(Int, counts)
    schedule = Vector{Int}(undef, n)
    remaining = collect(Int, counts)
    R = length(remaining)
    col = 1
    start = 1
    while col <= n
        placed = false
        for offset in 0:(R - 1)
            idx = ((start - 1 + offset) % R) + 1
            if remaining[idx] > 0
                schedule[col] = idx
                remaining[idx] -= 1
                col += 1
                start = (idx % R) + 1
                placed = true
                break
            end
        end
        placed || error("round_robin_regime_schedule: internal error (remaining=$remaining)")
    end
    return schedule
end

"""
    regime_counts_from_schedule(schedule, n_regimes) -> Vector{Int}
"""
function regime_counts_from_schedule(schedule::AbstractVector{<:Integer}, n_regimes::Int)
    counts = zeros(Int, n_regimes)
    for idx in schedule
        1 <= idx <= n_regimes || error("regime index $idx out of range 1:$n_regimes")
        counts[idx] += 1
    end
    return counts
end

"""
    find_base_regime_index(regimes) -> Int

Index of the regime named `"base"`, or `1` if none.
"""
function find_base_regime_index(regimes::Vector{<:Dict})
    idx = findfirst(r -> get(r, "name", "") == "base", regimes)
    return something(idx, 1)
end

"""
    build_regime_scalers(network, regimes) -> (load_scalers, der_scalers)

One `LoadScaler` and `DERScaler` per training regime.
"""
function build_regime_scalers(network, regimes::Vector{<:Dict})
    load_scalers = [LoadScaler(network, r["load"]) for r in regimes]
    der_scalers = [DERScaler(network, r["der"]) for r in regimes]
    return load_scalers, der_scalers
end
