# Generate `pv_systems.dss` for a feeder: PVs spread evenly over load buses plus a few
# three-phase PVs on no-load MV (primary) buses, sized to a target feeder penetration.
#
# Usage:
#   julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/generate_pv_systems.jl \
#       <master.dss> <n_pv> [--penetration 0.7] [--mv_frac 0.1] [--mv_share 0.3] [--out <path>]
#
#   julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/generate_pv_systems.jl datagen_topo/grid_data/123_bus/IEEE123Master.dss 50
#   julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/generate_pv_systems.jl datagen_topo/grid_data/240_bus/240bus_Master.dss 100
#   julia --project=datagen_topo/FeederFlow.jl datagen_topo/gen_scripts/generate_pv_systems.jl datagen_topo/grid_data/IEEE8500/8500_Master.dss 1000 --mv_frac 0.05
#
# Method (deterministic):
#   1. Order buses by iterative DFS from the slack bus over lines (open tie switches excluded),
#      transformers and regulators. Feeder order = DFS discovery order.
#   2. n_mv = round(mv_frac * n_pv) three-phase PVs on no-load MV buses
#      (3 modeled phases, 1 kV <= V_LN <= 20 kV, not slack); n_lv = n_pv - n_mv on load buses
#      (buses with positive nominal load kW).
#      Each group is picked at evenly spaced positions floor((k - 1/2) * N / M) + 1 in feeder order.
#   3. Total pmpp = penetration * total nominal load kW. MV units share mv_share of it equally;
#      load-bus units share the rest in proportion to their local nominal load kW.
#   4. Load-bus PV phases = union of the load phases at that bus (wye); kV = load kV.
#      kVA = pmpp (the OPF uses only pmpp; der_smax = der_pmax per phase).
#
# The master file redirects the existing pv_systems.dss; existing PVs do not affect the result
# (only buses, lines, transformers and loads are used).

using FeederFlow
using Printf

const VLN_MV_MIN = 1.0e3
const VLN_MV_MAX = 20.0e3

function parse_args(args)
    positional = String[]
    flags = Dict{String,String}()
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            i < length(args) || error("missing value for $(args[i])")
            flags[args[i][3:end]] = args[i + 1]
            i += 2
        else
            push!(positional, args[i])
            i += 1
        end
    end
    length(positional) == 2 || error("usage: generate_pv_systems.jl <master.dss> <n_pv> [--penetration p] [--mv_frac f] [--mv_share s] [--out path]")
    unknown = setdiff(keys(flags), ["penetration", "mv_frac", "mv_share", "out"])
    isempty(unknown) || error("unknown flags: $(collect(unknown))")
    master = positional[1]
    return (
        master = master,
        n_pv = parse(Int, positional[2]),
        penetration = parse(Float64, get(flags, "penetration", "0.7")),
        mv_frac = parse(Float64, get(flags, "mv_frac", "0.1")),
        mv_share = parse(Float64, get(flags, "mv_share", "0.3")),
        out = get(flags, "out", joinpath(dirname(master), "pv_systems.dss")),
    )
end

modeled(phases) = sort(unique(filter(p -> 1 <= p <= 3, phases)))

"""DFS discovery order of buses from the slack over closed lines, transformers and regulators."""
function feeder_order(net)
    adj = Dict{String,Vector{String}}()
    function connect!(a, b)
        a == b && return
        push!(get!(adj, a, String[]), b)
        push!(get!(adj, b, String[]), a)
    end
    for line in net.lines
        line.is_switch && !line.is_closed_base && continue
        connect!(line.from.bus, line.to.bus)
    end
    for xf in Iterators.flatten((net.transformers, net.regulators))
        buses = [w.bus.bus for w in xf.windings]
        for b in buses[2:end]
            connect!(buses[1], b)
        end
    end
    order = Dict{String,Int}()
    stack = [net.slack_bus]
    while !isempty(stack)
        bus = pop!(stack)
        haskey(order, bus) && continue
        order[bus] = length(order) + 1
        for nb in reverse(get(adj, bus, String[]))
            haskey(order, nb) || push!(stack, nb)
        end
    end
    return order
end

evenly_spaced(N, M) = [floor(Int, (k - 0.5) * N / M) + 1 for k in 1:M]

function main(args)
    opt = parse_args(args)
    opt.penetration > 0.0 || error("penetration must be positive")
    (0.0 <= opt.mv_frac <= 1.0 && 0.0 <= opt.mv_share <= 1.0) || error("mv_frac and mv_share must be in [0, 1]")

    net = FeederFlow.parse_file(opt.master)
    sbase_kw = net.base.Sbase / 1e3
    order = feeder_order(net)

    # Aggregate loads per bus.
    load_kw = Dict{String,Float64}()
    load_ph = Dict{String,Vector{Int}}()
    load_kv = Dict{String,Float64}()
    for l in net.loads
        b = l.bus.bus
        load_kw[b] = get(load_kw, b, 0.0) + l.p_pu * sbase_kw
        load_ph[b] = modeled(vcat(get(load_ph, b, Int[]), l.bus.phases))
        load_kv[b] = max(get(load_kv, b, 0.0), l.kv)
    end
    unreached = [b for b in keys(load_kw) if !haskey(order, b)]
    isempty(unreached) || error("load buses not reachable from slack: $(first(unreached, 5))")
    total_load_kw = sum(values(load_kw))

    # Zero-kW loads (e.g. 240-bus Load_2028) would give zero-capacity PVs; treat them as no-load buses.
    load_buses = sort([b for (b, p) in load_kw if p > 0.0]; by = b -> order[b])
    mv_buses = sort([bus.name for bus in net.buses if haskey(order, bus.name) && bus.name != net.slack_bus &&
                     !haskey(load_kw, bus.name) && length(modeled(bus.phases)) == 3 &&
                     VLN_MV_MIN <= bus.vbase <= VLN_MV_MAX]; by = b -> order[b])

    n_mv = round(Int, opt.mv_frac * opt.n_pv)
    n_lv = opt.n_pv - n_mv
    n_lv <= length(load_buses) || error("n_lv=$n_lv exceeds the $(length(load_buses)) load buses")
    n_mv <= length(mv_buses) || error("n_mv=$n_mv exceeds the $(length(mv_buses)) MV candidate buses")

    lv_sel = load_buses[evenly_spaced(length(load_buses), n_lv)]
    mv_sel = mv_buses[evenly_spaced(length(mv_buses), n_mv)]

    total_pv_kw = opt.penetration * total_load_kw
    mv_kw_total = n_mv > 0 ? opt.mv_share * total_pv_kw : 0.0
    lv_kw_total = total_pv_kw - mv_kw_total
    lv_sel_load = sum(load_kw[b] for b in lv_sel; init = 0.0)

    # (bus, phases, kV, pmpp kW, comment)
    entries = Tuple{String,Vector{Int},Float64,Float64,String}[]
    for b in mv_sel
        push!(entries, (b, [1, 2, 3], net.buses[b].vbase * sqrt(3) / 1e3, mv_kw_total / n_mv, "MV no-load bus"))
    end
    for b in lv_sel
        pmpp = lv_kw_total * load_kw[b] / lv_sel_load
        push!(entries, (b, load_ph[b], load_kv[b], pmpp, @sprintf("local load %.2f kW", load_kw[b])))
    end
    sort!(entries; by = e -> order[e[1]])

    written_kw = sum(round(e[4]; digits = 3) for e in entries)
    open(opt.out, "w") do io
        println(io, "! PV Systems for $(splitext(basename(opt.master))[1])")
        println(io, "! Generated by datagen_topo/gen_scripts/generate_pv_systems.jl")
        println(io, "!   args: $(basename(opt.master)) $(opt.n_pv) --penetration $(opt.penetration) --mv_frac $(opt.mv_frac) --mv_share $(opt.mv_share)")
        @printf(io, "! %d PV units: %d on load buses (of %d), %d three-phase on no-load MV buses (of %d candidates)\n",
                opt.n_pv, n_lv, length(load_buses), n_mv, length(mv_buses))
        @printf(io, "! Total PV pmpp: %.1f kW on %.1f kW nominal load (%.1f%% penetration)\n",
                written_kw, total_load_kw, 100 * written_kw / total_load_kw)
        @printf(io, "!   MV units: %.1f kW total; load-bus units: %.1f kW total (%.2f x their local load)\n",
                mv_kw_total, lv_kw_total, lv_sel_load > 0 ? lv_kw_total / lv_sel_load : 0.0)
        println(io, "! Entries are listed in feeder (DFS-from-substation) order.")
        println(io)
        for (b, ph, kv, pmpp, note) in entries
            p = round(pmpp; digits = 3)
            @printf(io, "New PVSystem.PV_%s  bus1=%s.%s  phases=%d  kV=%.4g  kVA=%.3f  pmpp=%.3f  irradiance=1.0  pf=1.0  %%cutin=0.1  %%cutout=0.1  ! %s\n",
                    b, b, join(ph, "."), length(ph), kv, p, p, note)
        end
    end
    @printf("Wrote %s: %d PVs (%d LV, %d MV), %.1f kW / %.1f kW load = %.1f%%\n",
            opt.out, opt.n_pv, n_lv, n_mv, written_kw, total_load_kw, 100 * written_kw / total_load_kw)
end

main(ARGS)
