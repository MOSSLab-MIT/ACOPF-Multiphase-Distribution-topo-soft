"""
    _relative_shift_noise(rng, n, mu, sigma, random_sign)

Sample relative shift noise for one vector.

When `random_sign=true`, each element is `sign * (mu + sigma * |Z|)` with independent
uniform `sign ∈ {-1, +1}`. When `random_sign=false`, legacy signed Gaussian
`mu + sigma * Z` is used.
"""
function _relative_shift_noise(rng::AbstractRNG, n::Int, mu::Real, sigma::Real, random_sign::Bool)
    if random_sign
        signs = ifelse.(rand(rng, n) .< 0.5, -1.0, 1.0)
        return signs .* (mu .+ sigma .* abs.(randn(rng, n)))
    end
    return mu .+ sigma .* randn(rng, n)
end

"""
    apply_shift!(pd, qd, pder_avai, seed, shift_cfg)

Apply distributional shift in-place to load and DER availability samples.

`shift_cfg` keys: `enabled`, `noise_type` (`gaussian` | `additive`), `sigma_rel`,
`der_sigma`, `mu_rel`, `der_mu`, `scale_factor`, `random_sign` (default `true`).
"""
function apply_shift!(
    pd::Vector,
    qd::Vector,
    pder_avai::Vector,
    seed::Int,
    shift_cfg::Dict,
)
    if !get(shift_cfg, "enabled", false)
        return nothing
    end

    rng = MersenneTwister(seed + 1000)
    nt = get(shift_cfg, "noise_type", "gaussian")
    sigma_rel = Float64(get(shift_cfg, "sigma_rel", 0.25))
    der_sigma = Float64(get(shift_cfg, "der_sigma", 0.30))
    mu_rel = Float64(get(shift_cfg, "mu_rel", 0.0))
    der_mu = Float64(get(shift_cfg, "der_mu", 0.0))
    random_sign = parse_bool(string(get(shift_cfg, "random_sign", true)))

    if nt == "gaussian"
        pd .*= (1 .+ _relative_shift_noise(rng, length(pd), mu_rel, sigma_rel, random_sign))
        qd .*= (1 .+ _relative_shift_noise(rng, length(qd), mu_rel, sigma_rel, random_sign))
        pder_avai .*= (1 .+ _relative_shift_noise(rng, length(pder_avai), der_mu, der_sigma, random_sign))
    elseif nt == "additive"
        pd .+= _relative_shift_noise(rng, length(pd), mu_rel, sigma_rel, random_sign)
        qd .+= _relative_shift_noise(rng, length(qd), mu_rel, sigma_rel, random_sign)
        pder_avai .+= _relative_shift_noise(rng, length(pder_avai), der_mu, der_sigma, random_sign)
    else
        error("Unsupported shift noise_type '$nt'. Use one of: gaussian, additive.")
    end

    sf = Float64(get(shift_cfg, "scale_factor", 1.0))
    if sf != 1.0
        pd .*= sf
        qd .*= sf
    end

    pd .= clamp.(pd, 0.0, 1.5) # clamp load to 0.0-1.5
    qd .= clamp.(qd, 0.0, 1.5) # clamp reactive load to 0.0-1.5
    pder_avai .= clamp.(pder_avai, 0.0, 1.0) # clamp DER availability to 0.0-1.0
    return nothing
end
