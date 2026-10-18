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

    # Uniform distribution
    d_α = Uniform(l, u)
    
    # Load-level uniform distributions
    d_η = Distributions.product_distribution([Uniform(1 - σ, 1 + σ) for σ in σs])

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


struct DERScaler
    d::Vector{Uniform{Float64}}  # distribution of multiplicative noise
    der_pavail_min::Vector{Float64}  # minimum available active DER power
    der_pavail_max::Vector{Float64}  # maximum available active DER power
end

function Random.rand(rng::AbstractRNG, ds::DERScaler)
    D = length(ds.der_pavail_min)
    ϵ = rand.(rng, ds.d)  # sample multiplicative noise

    pder = ϵ .* (ds.der_pavail_max - ds.der_pavail_min ) + ds.der_pavail_min

    return pder
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

function DERScaler(data::OPFData, options::Dict)
    D = data.n_der_unit
    frac_l, frac_u = _parse_der_frac_bounds(options)
    ref = data.der_pavail_ref
    der_pavail_min = frac_l .* ref
    der_pavail_max = frac_u .* ref
    d = [Uniform(0.0, 1.0) for _ in 1:D]
    return DERScaler(d, der_pavail_min, der_pavail_max)
end

DERScaler(data::OPFData) = DERScaler(data, Dict{String, Any}())

struct ObjScaler
    d::Vector{Uniform{Float64}}  # distribution of multiplicative noise
    wmin::Vector{Float64}  # minimum weights
    wmax::Vector{Float64}  # maximum weights
end

function Random.rand(rng::AbstractRNG, obj_weights::ObjScaler)
    D = length(obj_weights.wmin)
    ϵ = rand.(rng, obj_weights.d)  # sample multiplicative noise

    weights = ϵ .* (obj_weights.wmax - obj_weights.wmin ) + obj_weights.wmin

    return weights
end

function ObjScaler(data::OPFData, options::Dict)
    dim = length(data.obj_weights)
    wmin_raw = get(options, "wmin", nothing)
    wmax_raw = get(options, "wmax", nothing)

    wmin_raw isa AbstractVector || error("Missing or invalid obj.wmin: expected a vector of length $dim.")
    wmax_raw isa AbstractVector || error("Missing or invalid obj.wmax: expected a vector of length $dim.")
    length(wmin_raw) == dim || error("Invalid obj.wmin length $(length(wmin_raw)); expected $dim.")
    length(wmax_raw) == dim || error("Invalid obj.wmax length $(length(wmax_raw)); expected $dim.")

    wmin = Float64.(wmin_raw)
    wmax = Float64.(wmax_raw)
    all(wmin .<= wmax) || error("Invalid obj weight bounds: every wmin entry must be <= wmax.")

    # Noise distribution
    d = [Uniform(0.0, 1.0) for _ in 1:dim]

    return ObjScaler(d, wmin, wmax)
end
