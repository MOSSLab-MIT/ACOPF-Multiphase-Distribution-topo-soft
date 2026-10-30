"""
    _build_opf_cache(data, optimizer; warm_start, v_slack_local_override)

Internal: assemble the non-convex AC-OPF JuMP model in rectangular coordinates (bus-injection
nodal balance, DER limits) with SOFT voltage, branch-current, angle-difference and per-phase
substation apparent-power limits.

Soft limits use nonnegative slacks `s` in the constraint's raw units (pu² of the hard form) and an
L1 penalty `ρ·Σs` in the objective (exact penalty: for ρ above the hard-OPF multiplier the soft optimum
equals the hard optimum). Power balance and DER limits stay hard.

Uses scaled Y-bus `Y_scaled` fixed at build time for the current topology. Sample-varying
loads, DER availability and objective/penalty weights enter via JuMP `Parameter` variables.
"""
function _build_opf_cache(
    data::OPFData,
    optimizer;
    warm_start::Union{Nothing, AbstractVector{<:Complex}}=nothing,
    v_slack_local_override::Union{Nothing, AbstractVector{<:Complex}}=nothing,
)
    n_net = data.n_net
    n_slk = data.n_slk
    n_der_phase = length(data.der_node)
    n_der_unit = data.n_der_unit

    v_scale_full = data.v_scale_full
    Y_scaled = data.Y_scaled

    v_slack_local = if v_slack_local_override === nothing
        data.v_sub_ref ./ data.v_scale_slack
    else
        length(v_slack_local_override) == n_slk || throw(DimensionMismatch("v_slack_local_override length must equal n_slk"))
        ComplexF64.(v_slack_local_override)
    end

    u_start = if warm_start === nothing
        compute_warm_start(data)
    else
        length(warm_start) == n_net || throw(DimensionMismatch("warm_start length must equal n_net"))
        ComplexF64.(warm_start)
    end

    model = JuMP.Model(optimizer)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-6)
    set_optimizer_attribute(model, "max_iter", 500)
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    # No bound relaxation: Ipopt's default (1e-8) lets every slack sit at s ≈ -1e-8, and ρ·Σs then biases
    # the objective downward in proportion to ρ and the slack count (0.2% at ρ = 1e3 on IEEE13).
    set_optimizer_attribute(model, "bound_relax_factor", 0.0)
    # set_optimizer_attribute(model, "linear_solver", "mumps")
    set_attribute(model, "hsllib", HSL_PATH) # Remove this if you don't have HSL
    set_attribute(model, "linear_solver", "ma97") #ma97 might be better for IEEE8500; ma57 is good for smaller networks
    #ma97 is best for 240 bus

    # Sample-varying profile data as JuMP parameters (updated via sync_params!).
    @variable(model, p_load_param[i=1:n_net] in Parameter(0.0))
    @variable(model, q_load_param[i=1:n_net] in Parameter(0.0))
    @variable(model, der_pavail_param[g=1:n_der_unit] in Parameter(0.0))
    @variable(model, w_cost in Parameter(1.0))
    @variable(model, w_curtail in Parameter(1.0))
    @variable(model, rho_v in Parameter(PENALTY_WEIGHT_DEFAULT))
    @variable(model, rho_i in Parameter(PENALTY_WEIGHT_DEFAULT))
    @variable(model, rho_theta in Parameter(PENALTY_WEIGHT_DEFAULT))
    @variable(model, rho_sub in Parameter(PENALTY_WEIGHT_DEFAULT))

    # Local-pu voltage variables on non-slack nodes.
    @variable(model, vr[1:n_net])
    @variable(model, vi[1:n_net])

    @variable(model, pder[k=1:n_der_phase], lower_bound=0.0, upper_bound=data.der_pmax[k])
    @variable(model, qder[k=1:n_der_phase], lower_bound=data.der_qmin[k], upper_bound=data.der_qmax[k])

    @variable(model, psub)
    @variable(model, qsub)
    # No ±smax boxes on p_sub_k / q_sub_k: they are implied by the hard disk but would keep the soft one hard.
    @variable(model, p_sub_k[k=1:n_slk])
    @variable(model, q_sub_k[k=1:n_slk])

    # Soft voltage magnitude limits on non-slack nodes (local pu²):
    #   vmin² - s_vmin <= |u|² <= vmax² + s_vmax,   s_vmin, s_vmax >= 0.
    @variable(model, s_vmin[1:n_net] >= 0.0)
    @variable(model, s_vmax[1:n_net] >= 0.0)
    @constraint(model, c_vmin[i=1:n_net], vr[i]^2 + vi[i]^2 >= data.vmin[i]^2 - s_vmin[i])
    @constraint(model, c_vmax[i=1:n_net], vr[i]^2 + vi[i]^2 <= data.vmax[i]^2 + s_vmax[i])

    # Per-phase apparent limits and per-unit aggregate active availability.
    @constraint(model, [k=1:n_der_phase], pder[k]^2 + qder[k]^2 <= data.der_smax[k]^2)
    @constraint(model, [g=1:n_der_unit], sum(pder[k] for k in data.der_unit_to_phase[g]; init=0.0) <= der_pavail_param[g])

    # Nodal power balance in local-pu coordinates.
    # Current I = Y*u with u = vr + j*vi; injection S = u * conj(I) => P_net, Q_net below.
    ders_at_node = data.node_to_der_phase

    @expression(model, I_re[i=1:n_net],
        sum(real(Y_scaled[i, j]) * vr[j] - imag(Y_scaled[i, j]) * vi[j] for j in 1:n_net) +
        sum(real(Y_scaled[i, n_net + k]) * real(v_slack_local[k]) - imag(Y_scaled[i, n_net + k]) * imag(v_slack_local[k]) for k in 1:n_slk),
    )
    @expression(model, I_im[i=1:n_net],
        sum(real(Y_scaled[i, j]) * vi[j] + imag(Y_scaled[i, j]) * vr[j] for j in 1:n_net) +
        sum(real(Y_scaled[i, n_net + k]) * imag(v_slack_local[k]) + imag(Y_scaled[i, n_net + k]) * real(v_slack_local[k]) for k in 1:n_slk),
    )

    @expression(model, P_net[i=1:n_net], vr[i] * I_re[i] + vi[i] * I_im[i])
    @expression(model, Q_net[i=1:n_net], vi[i] * I_re[i] - vr[i] * I_im[i])

    @constraint(model, [i=1:n_net], sum(pder[k] for k in ders_at_node[i]; init=0.0) - p_load_param[i] == P_net[i])
    @constraint(model, [i=1:n_net], sum(qder[k] for k in ders_at_node[i]; init=0.0) - q_load_param[i] == Q_net[i])

    # Per-phase substation P/Q: S_slk = u_slk * conj(I_slk) with fixed slack voltages u_slk.
    @expression(model, I_re_slk[k=1:n_slk],
        sum(real(Y_scaled[n_net + k, m]) * vr[m] - imag(Y_scaled[n_net + k, m]) * vi[m] for m in 1:n_net) +
        sum(real(Y_scaled[n_net + k, n_net + s]) * real(v_slack_local[s]) - imag(Y_scaled[n_net + k, n_net + s]) * imag(v_slack_local[s]) for s in 1:n_slk),
    )
    @expression(model, I_im_slk[k=1:n_slk],
        sum(real(Y_scaled[n_net + k, m]) * vi[m] + imag(Y_scaled[n_net + k, m]) * vr[m] for m in 1:n_net) +
        sum(real(Y_scaled[n_net + k, n_net + s]) * imag(v_slack_local[s]) + imag(Y_scaled[n_net + k, n_net + s]) * real(v_slack_local[s]) for s in 1:n_slk),
    )
    @constraint(model, [k=1:n_slk], p_sub_k[k] == real(v_slack_local[k]) * I_re_slk[k] + imag(v_slack_local[k]) * I_im_slk[k])
    @constraint(model, [k=1:n_slk], q_sub_k[k] == imag(v_slack_local[k]) * I_re_slk[k] - real(v_slack_local[k]) * I_im_slk[k])
    # Soft per-phase substation apparent-power limit (local pu²): |S_k|² <= smax_k² + s_sub_k, s_sub_k >= 0.
    @variable(model, s_sub[1:n_slk] >= 0.0)
    @constraint(model, c_sub[k=1:n_slk], p_sub_k[k]^2 + q_sub_k[k]^2 <= data.sub_smax_phase[k]^2 + s_sub[k])

    @constraint(model, psub == sum(p_sub_k))
    @constraint(model, qsub == sum(q_sub_k))

    # Branch constraints: voltage angle difference (tan limit) and SOC current on each phase pair.
    function local_voltage_parts(i)
        if !data.all_is_slack[i]
            return vr[i], vi[i]
        end
        slack_i = data.all_to_slack_local[i]
        return real(v_slack_local[slack_i]), imag(v_slack_local[slack_i])
    end

    function system_voltage_parts(i)
        vr_i, vi_i = local_voltage_parts(i)
        return v_scale_full[i] * vr_i, v_scale_full[i] * vi_i
    end

    function system_voltage_drop_parts(i_f, i_t)
        vr_f, vi_f = system_voltage_parts(i_f)
        vr_t, vi_t = system_voltage_parts(i_t)
        return vr_f - vr_t, vi_f - vi_t
    end

    function line_current_parts(y_series, r, terminal_cache)
        from_idx = terminal_cache.from_idx
        to_idx = terminal_cache.to_idx
        valid = terminal_cache.valid

        ir = @expression(model, sum(
            begin
                if !valid[c]
                    0.0
                else
                    i_f = from_idx[c]
                    i_t = to_idx[c]
                    y = y_series[r, c]
                    dv_re, dv_im = system_voltage_drop_parts(i_f, i_t)
                    real(y) * dv_re - imag(y) * dv_im
                end
            end for c in eachindex(from_idx)
        ))

        ii = @expression(model, sum(
            begin
                if !valid[c]
                    0.0
                else
                    i_f = from_idx[c]
                    i_t = to_idx[c]
                    y = y_series[r, c]
                    dv_re, dv_im = system_voltage_drop_parts(i_f, i_t)
                    imag(y) * dv_re + real(y) * dv_im
                end
            end for c in eachindex(from_idx)
        ))

        return ir, ii
    end

    # One slack per valid (line, phase-row) for current and for each sign of the angle limit.
    n_branch_rows = sum(count(v) for v in data.branch_valid; init=0)
    @variable(model, s_imax[1:n_branch_rows] >= 0.0)
    @variable(model, s_ang_pos[1:n_branch_rows] >= 0.0)
    @variable(model, s_ang_neg[1:n_branch_rows] >= 0.0)
    c_imax = Vector{ConstraintRef}(undef, n_branch_rows)
    c_ang_pos = Vector{ConstraintRef}(undef, n_branch_rows)
    c_ang_neg = Vector{ConstraintRef}(undef, n_branch_rows)
    row = 0

    for line_idx in eachindex(data.branch_from_idx)
        terminal_cache = (
            from_idx = data.branch_from_idx[line_idx],
            to_idx = data.branch_to_idx[line_idx],
            valid = data.branch_valid[line_idx],
        )
        y_series = data.branch_y_series[line_idx]
        imax = data.branch_imax[line_idx]

        for r in eachindex(terminal_cache.from_idx)
            terminal_cache.valid[r] || continue
            row += 1

            i_f = terminal_cache.from_idx[r]
            i_t = terminal_cache.to_idx[r]

            vr_f, vi_f = local_voltage_parts(i_f)
            vr_t, vi_t = local_voltage_parts(i_t)

            angle_dot = @expression(model, vr_f * vr_t + vi_f * vi_t)
            # Soft angle bound: |Im(u_f * conj(u_t))| <= dvamax * Re(u_f * conj(u_t)) + s  (60 deg default).
            angle_cross = @expression(model, vr_f * vi_t - vi_f * vr_t)

            c_ang_pos[row] = @constraint(model, angle_cross <= data.dvamax * angle_dot + s_ang_pos[row])
            c_ang_neg[row] = @constraint(model, -angle_cross <= data.dvamax * angle_dot + s_ang_neg[row])

            # Soft current bound in system pu²: |I|² <= imax² + s.
            ir_line, ii_line = line_current_parts(y_series, r, terminal_cache)
            c_imax[row] = @constraint(model, ir_line^2 + ii_line^2 <= imax^2 + s_imax[row])
        end
    end
    row == n_branch_rows || error("internal: branch slack count mismatch ($row != $n_branch_rows)")
    # Register soft branch constraints (row-aligned with their slacks) for dual / KKT access.
    model[:c_imax] = c_imax
    model[:c_ang_pos] = c_ang_pos
    model[:c_ang_neg] = c_ang_neg

    length(data.obj_weights) == N_OBJ_WEIGHTS || throw(DimensionMismatch("obj_weights must have length $N_OBJ_WEIGHTS"))
    sub_lin, sub_quad = _linear_quadratic_terms(data.sub_cost)
    der_lin = [(_linear_quadratic_terms(c))[1] for c in data.der_cost]
    der_quad = [(_linear_quadratic_terms(c))[2] for c in data.der_cost]

    @expression(model, obj_cost,
        sum(der_lin[k] * pder[k] + der_quad[k] * pder[k]^2 for k in 1:n_der_phase) +
        sub_lin * psub + sub_quad * psub^2,
    )

    @expression(model, obj_curtailment,
        sum(der_pavail_param[g] - sum(pder[k] for k in data.der_unit_to_phase[g]; init=0.0) for g in 1:n_der_unit),
    )

    # L1 penalties on soft-limit slacks (raw pu² units).
    @expression(model, pen_v, sum(s_vmin) + sum(s_vmax))
    @expression(model, pen_i, sum(s_imax))
    @expression(model, pen_theta, sum(s_ang_pos) + sum(s_ang_neg))
    @expression(model, pen_sub, sum(s_sub))

    # Weighted sum: economic cost (DER + substation quadratic/linear) + curtailment + soft-limit penalties.
    @objective(model, Min,
        w_cost * obj_cost + w_curtail * obj_curtailment +
        rho_v * pen_v + rho_i * pen_i + rho_theta * pen_theta + rho_sub * pen_sub,
    )

    return OPFSolveCache(
        model,
        data,
        copy(data.branch_status),
        [p_load_param[i] for i in 1:n_net],
        [q_load_param[i] for i in 1:n_net],
        [der_pavail_param[g] for g in 1:n_der_unit],
        w_cost,
        w_curtail,
        [rho_v, rho_i, rho_theta, rho_sub],
        [vr[i] for i in 1:n_net],
        [vi[i] for i in 1:n_net],
        u_start,
        0.0,
        true,
    )
end
