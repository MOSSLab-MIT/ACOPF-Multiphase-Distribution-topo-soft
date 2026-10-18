"""
    apply_switch_stamp!(Y_scaled, from_idx, to_idx, valid, y_series, y_shunt, v_scale_full, stamp_sign)

Add or remove one switch's π-model contribution to the scaled Y-bus.

``stamp_sign = +1`` adds the stamp (switch closed relative to base); ``-1`` removes it (opened).
Self and mutual entries are scaled by ``v_scale_full[i] * v_scale_full[j]`` for local-pu stamping.
"""
function apply_switch_stamp!(
    Y_scaled::AbstractMatrix{ComplexF64},
    from_idx::Vector{Int},
    to_idx::Vector{Int},
    valid::Vector{Bool},
    y_series::Matrix{ComplexF64},
    y_shunt::Matrix{ComplexF64},
    v_scale_full::AbstractVector{<:Real},
    stamp_sign::Integer,
)
    @inbounds for r in eachindex(from_idx)
        i_from = from_idx[r]
        i_to = to_idx[r]
        (!valid[r] || i_from == 0 || i_to == 0) && continue

        for c in eachindex(from_idx)
            j_from = from_idx[c]
            j_to = to_idx[c]
            (!valid[c] || j_from == 0 || j_to == 0) && continue

            # π-model stamp: diagonal += y_series + y_shunt/2, off-diagonal += -y_series.
            self_val = y_series[r, c] + 0.5 * y_shunt[r, c]
            mutual_val = -y_series[r, c]

            scale_ff = v_scale_full[i_from] * v_scale_full[j_from]
            scale_tt = v_scale_full[i_to] * v_scale_full[j_to]
            scale_ft = v_scale_full[i_from] * v_scale_full[j_to]
            scale_tf = v_scale_full[i_to] * v_scale_full[j_from]

            Y_scaled[i_from, j_from] += stamp_sign * self_val * scale_ff
            Y_scaled[i_to, j_to] += stamp_sign * self_val * scale_tt
            Y_scaled[i_from, j_to] += stamp_sign * mutual_val * scale_ft
            Y_scaled[i_to, j_from] += stamp_sign * mutual_val * scale_tf
        end
    end

    return Y_scaled
end

"""Copy network–network and network–slack blocks from ``Y_scaled`` into ``Y_nn``, ``Y_ns``."""
function refresh_warm_start_blocks!(data::OPFData)
    n_net = data.n_net
    data.Y_nn .= data.Y_scaled[1:n_net, 1:n_net]
    data.Y_ns .= data.Y_scaled[1:n_net, (n_net + 1):data.n_total]
    return nothing
end

"""
    patch_switch_admittance!(data::OPFData)

Rebuild ``Y_scaled`` from ``Y_scaled_base`` by applying switch stamps for every switch whose
``switch_is_closed`` differs from ``switch_is_closed_base``. Syncs ``branch_status`` and
``switch_states``, then refreshes warm-start blocks.
"""
function patch_switch_admittance!(data::OPFData)
    Y_scaled = copy(data.Y_scaled_base)

    for s in eachindex(data.switch_from_idx)
        data.switch_is_closed[s] == data.switch_is_closed_base[s] && continue
        stamp_sign = data.switch_is_closed[s] ? 1 : -1
        apply_switch_stamp!(
            Y_scaled,
            data.switch_from_idx[s],
            data.switch_to_idx[s],
            data.switch_valid[s],
            data.switch_y_series[s],
            data.switch_y_shunt[s],
            data.v_scale_full,
            stamp_sign,
        )
    end

    data.Y_scaled = Y_scaled
    refresh_warm_start_blocks!(data)

    for (s, line_idx) in enumerate(data.switch_idx)
        data.branch_status[line_idx] = data.switch_is_closed[s]
    end

    for (name, idx) in data.switch_name_to_idx
        data.switch_states[name] = data.switch_is_closed[idx]
    end

    return data.Y_scaled
end

"""Return a copy of switch name → closed (``true``) / open (``false``) from current ``OPFData``."""
function query_switch_states(data::OPFData)
    states = Dict{String, Bool}()
    for (name, idx) in data.switch_name_to_idx
        states[name] = data.switch_is_closed[idx]
    end
    return states
end

"""
    perturb_switch_states!(data::OPFData; rng, flip_probability=0.2)

Random topology perturbation for datagen: each switch independently flips from its base state
with probability ``flip_probability``, then [`patch_switch_admittance!`](@ref) is applied.

Returns the sampled ``Dict{String, Bool}`` of switch states.
"""
function perturb_switch_states!(data::OPFData; rng=Random.default_rng(), flip_probability=0.2)
    0.0 <= flip_probability <= 1.0 || throw(ArgumentError("flip_probability must be in [0, 1]"))

    sampled_states = Dict{String, Bool}()
    for (name, idx) in data.switch_name_to_idx
        base_closed = data.switch_is_closed_base[idx]
        flip = rand(rng) < flip_probability
        new_closed = flip ? !base_closed : base_closed
        data.switch_is_closed[idx] = new_closed
        sampled_states[name] = new_closed
    end

    patch_switch_admittance!(data)
    return sampled_states
end



"""
    update_branch_status!(data::OPFData, new_status::Vector{Bool})

Apply a full ``branch_status`` vector (one entry per ``network.lines``), update switch states,
and rebuild ``Y_scaled`` via [`patch_switch_admittance!`](@ref).

Only switch lines may differ from the previous status; non-switch changes raise `ArgumentError`.
"""
function update_branch_status!(data::OPFData, new_status::Vector{Bool})
    n_branch = length(data.branch_status)
    length(new_status) == n_branch || throw(DimensionMismatch("new_status length must equal number of branches"))

    switch_mask = falses(n_branch)
    switch_mask[data.switch_idx] .= true

    for i in 1:n_branch
        if !switch_mask[i] && new_status[i] != data.branch_status[i]
            throw(ArgumentError("Only switch lines can change status; attempted non-switch change at line index $i"))
        end
    end

    data.branch_status .= new_status
    for (s, line_idx) in enumerate(data.switch_idx)
        data.switch_is_closed[s] = new_status[line_idx]
    end

    patch_switch_admittance!(data)
    return data
end

"""
    compute_warm_start(data::OPFData)

Linear warm start for non-slack voltages: solve ``Y_nn * u_net = -Y_ns * u_slack`` with fixed
slack voltages ``u_slack = v_sub_ref ./ v_scale_slack`` (local per-unit). Refreshes Y from
current switch states first.
"""
function compute_warm_start(data::OPFData)
    patch_switch_admittance!(data)

    v_slack_local = data.v_sub_ref ./ data.v_scale_slack
    rhs = -(data.Y_ns * v_slack_local)
    u_start = data.Y_nn \ rhs

    return u_start
end
