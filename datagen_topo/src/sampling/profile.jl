using Distributions
using Random
using Random: AbstractRNG
using Distributions: length, eltype, _rand!, ContinuousUnivariateDistribution, ContinuousMultivariateDistribution

const UvDist = Distributions.ContinuousUnivariateDistribution
const MvDist = Distributions.ContinuousMultivariateDistribution

"""
    Glocal{G,L}

A glocal distribution with global/local factors `α::G` and  `η::L`.

This distribution represents a random variable of the form `ϵ = α×η`, where
* `α` is a _scalar_ random variable, with distribution `d_α::G`
* `η` is a _vector_ random variable, with distribution `d_η::L`
* `α` and `η` are independent random variables
"""
struct Glocal{G<:UvDist,L<:MvDist} <: MvDist
    d_α::G
    d_η::L
end

Distributions.length(d::Glocal) = length(d.d_η)
Distributions.eltype(::Glocal) = Float64

# custom samplers for Array-shaped data
function Distributions._rand!(::AbstractRNG, ::Glocal, ::AbstractArray)
    throw(DimensionMismatch(
        "Inconsistent argument dimensions: only vector and matrix-shaped `x` is supported."
    ))
end
function Distributions._rand!(rng::AbstractRNG, d::Glocal, x::AbstractVector)
    length(x) == length(d) || throw(DimensionMismatch("Inconsistent argument dimensions."))

    α = rand(rng, d.d_α)
    η = rand(rng, d.d_η)

    x .= α .* η

    return x
end
function Distributions._rand!(rng::AbstractRNG, d::Glocal, x::AbstractMatrix)
    size(x, 1) == length(d) || throw(DimensionMismatch("Inconsistent argument dimensions."))
    n = size(x, 2)

    α = rand(rng, d.d_α, n)
    η = rand(rng, d.d_η, n)

    mul!(x, η, Diagonal(α))  # re-scales every column `η[:, j]` by `α[j]`

    return x
end

"""
    ScaledLogNormal(l, u, σs)

Generate a [`Glocal`](@ref) distribution `ϵ = α×η` where `α~U[l,u]` and `ηᵢ ~ LogNormal(-σᵢ²/2, σᵢ)`.
"""
function ScaledLogNormal(l::Float64, u::Float64, σs::Vector{Float64})
    l <= u || error("Invalid bounds: l > u")
    all(σs .>= 0) || error("Invalid input: σs must be non-negative")

    # Uniform distribution
    d_α = Uniform(l, u)
    
    # Generate LogNormal distributions
    σs2 = σs .^ 2
    d_η = MvLogNormal(-σs2 ./ 2, Diagonal(σs2))

    return Glocal(d_α, d_η)
end

"""
    ScaledUniform(l, u, σs)

Generate a [`Glocal`](@ref) distribution `ϵ = α×η` where `α ~ U[l,u]` and `ηᵢ ~ U[1-σᵢ, 1+σᵢ]`.
"""
function ScaledUniform(l::Float64, u::Float64, σs::Vector{Float64})
    l <= u || error("Invalid bounds: l > u")
    all(σs .>= 0) || error("Invalid input: σs must be non-negative")

    # Bounds are validated above. `check_args=false` admits the degenerate l == u and σ == 0, which
    # `Uniform` otherwise rejects; draws are unchanged for non-degenerate ranges.
    d_α = Uniform(l, u; check_args=false)

    # Load-level uniform distributions
    d_η = Distributions.product_distribution([Uniform(1 - σ, 1 + σ; check_args=false) for σ in σs])

    return Glocal(d_α, d_η)
end


abstract type AbstractLoadSampler end

"""
    LoadScaler{D}

Scales loads with multiplicative noise sampled from `d::D`.

The distribution `d::D` is a `2*L`-dimensional distribution.
The sampled active (resp reactive) demand for load ``i`` is denoted by ``\\tilde{p}_{i}`` (resp. ``\\tilde{q}_{i}``) and has the form
```math
\\tilde{p}_{i} = \\epsilon_{i} \\bar{p}_{i}, \\quad
\\tilde{q}_{i} = \\epsilon_{i+L} \\bar{q}_{i},
```
where ``\\bar{p}_{i}, \\bar{q}_{i}`` are the reference active/reactive demand for load ``i``,
and ``\\epsilon \\in \\mathbb{R}^{2L}`` is multiplicative noise sampled from distribution `d::D`.
"""
struct LoadScaler{D} <: AbstractLoadSampler
    d::D    # distribution of multiplicative noise
    pd_ref::Vector{Float64}  # reference active load
    qd_ref::Vector{Float64}  # reference reactive load
end

# struct DERScaler{D} <: AbstractLoadSampler
#     d::D    # distribution of multiplicative noise
#     pder_ref::Vector{Float64}  # reference active load
#     qder_ref::Vector{Float64}  # reference reactive load
# end

function Random.rand(rng::AbstractRNG, ls::LoadScaler)
    L = length(ls.pd_ref)
    ϵ = rand(rng, ls.d)  # sample multiplicative noise

    pd = ϵ[1:L] .* ls.pd_ref
    qd = ϵ[(L+1):(2*L)] .* ls.qd_ref

    return pd, qd
end

function LoadScaler(data::OPFData, options::Dict)
    L = data.n_net
    pd = copy(data.p_load_ref)
    qd = copy(data.q_load_ref)

    # Noise distribution
    # TODO: modularize this
    noise_type = get(options, "noise_type", "")
    if noise_type ∉ ["ScaledLogNormal", "ScaledUniform"]
        error("""Invalid noise type for load: $(noise_type). Supported values are:
        * \"ScaledLogNormal\"
        * \"ScaledUniform\"""")
    end

    # Grab noise parameters, and perform sanity checks
    l = get(options, "l", NaN)
    u = get(options, "u", NaN)
    if !(isa(l, Real) && isfinite(l))
        error("Missing or invalid input data: \"l\"")
    end
    if !(isa(u, Real) && isfinite(u))
        error("Missing or invalid input data: \"u\"")
    end
    if !(l <= u)
        error("Invalid global scaling parameters: [$l, $u]")
    end

    σs = zeros(Float64, 2*L)
    σ = get(options, "sigma", NaN)
    if isa(σ, AbstractVector{<:Real})
        length(σ) == L || error("Invalid sigma length")
        # We use the same noise level for active/reactive load
        σs[1:L] .= σ
        σs[(L+1):(2*L)] .= σ
    elseif isa(σ, Real) && isfinite(σ)
        σs .= σ
    else
        error("Invalid input data: sigma. Must be a finite real number or a real-valued vector.")
    end

    # Noise distribution
    d = if noise_type == "ScaledLogNormal"
        ScaledLogNormal(l, u, σs)
    elseif noise_type == "ScaledUniform"
        ScaledUniform(l, u, σs)
    end

    return LoadScaler(d, pd, qd)
end


# Per-unit spread of DER availability around the feeder-wide factor (see `DERScaler`).
const DEFAULT_DER_SIGMA = 0.5

"""
    DERScaler{D}

Samples correlated per-unit DER availability
```math
\\tilde{p}_{i} = \\min(\\alpha \\eta_{i}, 1) \\bar{p}_{i}, \\quad
\\alpha \\sim U[l, u], \\quad \\eta_{i} \\sim U[1-\\sigma, 1+\\sigma],
```
where `α` is one feeder-wide factor per sample and ``\\bar{p}_{i}`` is the rated availability.

`frac_l`/`frac_u` bound `α`, not each unit: a unit can exceed `frac_u` by up to `(1+σ)`, and only
the physical rating clips. `d` is `nothing` for a feeder without DER units.
"""
struct DERScaler{D}
    d::D  # distribution of the multiplicative factor α×η
    rated::Vector{Float64}  # rated availability, snapshotted: `_set_DERs!` overwrites the network's copy
end

function Random.rand(rng::AbstractRNG, ds::DERScaler)
    isempty(ds.rated) && return Float64[]
    ϵ = rand(rng, ds.d)
    return min.(ϵ, 1.0) .* ds.rated
end

function _parse_der_frac_bounds(options::Dict)
    frac_l = Float64(get(options, "frac_l", 0.0))
    frac_u = Float64(get(options, "frac_u", 1.0))
    isfinite(frac_l) && isfinite(frac_u) ||
        error("DER fraction bounds must be finite reals; got frac_l=$frac_l, frac_u=$frac_u")
    frac_l >= 0.0 || error("Invalid der frac_l=$frac_l: must be >= 0")
    frac_u >= 0.0 || error("Invalid der frac_u=$frac_u: must be >= 0")
    frac_l <= frac_u || error("Invalid DER fraction bounds: frac_l=$frac_l > frac_u=$frac_u")
    frac_u <= 1.0 || error("Invalid der frac_u=$frac_u: must be <= 1 (fraction of rated availability)")
    return frac_l, frac_u
end

function _parse_der_sigma(options::Dict)
    σ = Float64(get(options, "sigma", DEFAULT_DER_SIGMA))
    isfinite(σ) && 0.0 <= σ <= 1.0 || error("Invalid der sigma=$σ: must be in [0, 1]")
    return σ
end

"""
    normalize_der_options(options) -> Dict

Validated DER sampling options with exactly the keys `frac_l`, `frac_u`, `sigma`.
"""
function normalize_der_options(options::Dict)
    frac_l, frac_u = _parse_der_frac_bounds(options)
    return Dict{String, Any}("frac_l" => frac_l, "frac_u" => frac_u, "sigma" => _parse_der_sigma(options))
end

function DERScaler(data::OPFData, options::Dict)
    D = data.n_der_unit
    frac_l, frac_u = _parse_der_frac_bounds(options)
    σ = _parse_der_sigma(options)
    d = D == 0 ? nothing : ScaledUniform(frac_l, frac_u, fill(σ, D))
    return DERScaler(d, copy(data.der_pavail_ref))
end

DERScaler(data::OPFData) = DERScaler(data, Dict{String, Any}())

# Default soft-limit penalty sampling: ρ_V, ρ_I, ρ_θ, ρ_sub ~ log-uniform on [rho_min, rho_max].
const DEFAULT_PENALTY_SAMPLING = Dict{String, Any}(
    "rho_min" => [100.0, 100.0, 100.0, 100.0],
    "rho_max" => [1000.0, 1000.0, 1000.0, 200.0],
)

"""
    ObjScaler

Sampler for `obj_weights = [w_cost, w_curtail, ρ_V, ρ_I, ρ_θ, ρ_sub]`. With `ϵ ~ U(0, 1)` per entry:
- economic weights (`log_scale = false`): `w = wmin + ϵ (wmax - wmin)`;
- penalty weights (`log_scale = true`): `log w = log wmin + ϵ (log wmax - log wmin)` (log-uniform),
  since penalty weights are meaningful across orders of magnitude.
"""
struct ObjScaler
    d::Vector{Uniform{Float64}}  # distribution of the per-entry uniform noise
    wmin::Vector{Float64}  # minimum weights
    wmax::Vector{Float64}  # maximum weights
    log_scale::Vector{Bool}  # true: sample log-uniformly
end

function Random.rand(rng::AbstractRNG, obj_weights::ObjScaler)
    ϵ = rand.(rng, obj_weights.d)  # sample uniform noise

    return [
        ls ? exp(log(lo) + e * (log(hi) - log(lo))) : lo + e * (hi - lo)
        for (e, lo, hi, ls) in zip(ϵ, obj_weights.wmin, obj_weights.wmax, obj_weights.log_scale)
    ]
end

function _parse_weight_bounds(options::Dict, key_min::String, key_max::String, dim::Int, block::String)
    lo_raw = get(options, key_min, nothing)
    hi_raw = get(options, key_max, nothing)
    lo_raw isa AbstractVector || error("Missing or invalid $block.$key_min: expected a vector of length $dim.")
    hi_raw isa AbstractVector || error("Missing or invalid $block.$key_max: expected a vector of length $dim.")
    length(lo_raw) == dim || error("Invalid $block.$key_min length $(length(lo_raw)); expected $dim.")
    length(hi_raw) == dim || error("Invalid $block.$key_max length $(length(hi_raw)); expected $dim.")
    lo = Float64.(lo_raw)
    hi = Float64.(hi_raw)
    all(isfinite, lo) && all(isfinite, hi) || error("Invalid $block bounds: entries must be finite.")
    all(lo .<= hi) || error("Invalid $block bounds: every $key_min entry must be <= $key_max.")
    return lo, hi
end

function ObjScaler(data::OPFData, obj_options::Dict, penalty_options::Dict=DEFAULT_PENALTY_SAMPLING)
    length(data.obj_weights) == N_OBJ_WEIGHTS ||
        error("data.obj_weights has length $(length(data.obj_weights)); expected $N_OBJ_WEIGHTS.")
    n_pen = N_OBJ_WEIGHTS - N_ECON_WEIGHTS

    wmin, wmax = _parse_weight_bounds(obj_options, "wmin", "wmax", N_ECON_WEIGHTS, "obj")
    all(wmin .>= 0.0) || error("Invalid obj.wmin: weights must be nonnegative.")
    rho_min, rho_max = _parse_weight_bounds(penalty_options, "rho_min", "rho_max", n_pen, "penalty")
    all(rho_min .> 0.0) || error("Invalid penalty.rho_min: log-uniform sampling needs rho_min > 0.")

    d = [Uniform(0.0, 1.0) for _ in 1:N_OBJ_WEIGHTS]
    log_scale = vcat(falses(N_ECON_WEIGHTS), trues(n_pen))
    return ObjScaler(d, vcat(wmin, rho_min), vcat(wmax, rho_max), log_scale)
end
