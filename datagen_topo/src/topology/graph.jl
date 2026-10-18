using FeederFlow

"""
    BusTopologyContext

Phase-aware graph metadata for connectivity / radiality checks, aligned with the OPF
`(bus, phase)` node model. Bus-level string fields are retained for logging and legacy tests.
"""
struct BusTopologyContext
    slack_bus::String
    slack_seed_buses::Vector{String}
    load_buses::Set{String}
    line_bus_fr::Vector{String}
    line_bus_to::Vector{String}
    transformer_edges::Vector{Tuple{String, String}}
    bus_to_idx::Dict{String, Int}
    n_bus::Int
    # Bus-level graph (phase-aggregated radiality)
    slack_seed_bus_idx::Vector{Int}
    load_bus_idx::Vector{Int}
    line_fr_idx::Vector{Int}
    line_to_idx::Vector{Int}
    xf_fr_idx::Vector{Int}
    xf_to_idx::Vector{Int}
    # Phase-aware graph (integer indices into node_keys)
    node_keys::Vector{Tuple{String, Int}}
    n_phase::Int
    slack_seed_idx::Vector{Int}
    required_node_idx::Vector{Int}
    line_fr_node_idx::Vector{Vector{Int}}
    line_to_node_idx::Vector{Vector{Int}}
    xf_phase_edges::Vector{Tuple{Int, Int}}
end

"""Alias used by [`TopologySampler`](@ref)."""
const TopologyGraphCache = BusTopologyContext

"""Strip DSS phase/node suffixes: `47.1.2.3` -> `47`, `9r.1` -> `9r`."""
function _base_bus_name(name::AbstractString)
    s = lowercase(strip(String(name)))
    dot = findfirst('.', s)
    return dot === nothing ? s : s[1:prevind(s, first(dot))]
end

"""Normalize DSS bus names to lowercase base bus strings."""
function _normalize_bus_name(endpoint)
    name = endpoint isa AbstractString ? endpoint : endpoint.bus
    return _base_bus_name(String(name))
end

function _excluded_buses(network::NetworkModel)
    excluded = Set{String}()
    for bus in network.buses
        endswith(bus.name, "_open") && push!(excluded, bus.name)
    end
    return excluded
end

"""Transformers and regulators both couple connectivity (e.g. 150 <-> 150r)."""
function _coupling_devices(network::NetworkModel)
    return vcat(collect(network.transformers), collect(network.regulators))
end

"""Bus-level edges for every transformer/regulator winding pair (substation-seed BFS)."""
function _build_transformer_edges(network::NetworkModel)
    transformer_edges = Tuple{String, String}[]
    for tr in _coupling_devices(network)
        buses = unique(_normalize_bus_name(w.bus) for w in tr.windings)
        for i in 1:length(buses), j in (i + 1):length(buses)
            push!(transformer_edges, (buses[i], buses[j]))
        end
    end
    return transformer_edges
end

"""
    _substation_seed_buses(network, transformer_edges) -> Vector{String}

Buses reachable from `network.slack_bus` through transformer/regulator coupling only
(substation stack — not distribution transformers).
"""
function _substation_seed_buses(network::NetworkModel, transformer_edges::Vector{Tuple{String, String}})
    slack_bus = _normalize_bus_name(network.slack_bus)
    adj = Dict{String, Set{String}}()
    for (a, b) in transformer_edges
        haskey(adj, a) || (adj[a] = Set{String}())
        haskey(adj, b) || (adj[b] = Set{String}())
        push!(adj[a], b)
        push!(adj[b], a)
    end
    haskey(adj, slack_bus) || (adj[slack_bus] = Set{String}())

    visited = Set{String}()
    queue = [slack_bus]
    while !isempty(queue)
        bus = popfirst!(queue)
        bus in visited && continue
        push!(visited, bus)
        for nbr in get(adj, bus, Set{String}())
            nbr in visited && continue
            push!(queue, nbr)
        end
    end
    return collect(visited)
end

function _register_bus!(bus_to_idx::Dict{String, Int}, name::AbstractString)
    key = _normalize_bus_name(name)
    haskey(bus_to_idx, key) || (bus_to_idx[key] = length(bus_to_idx) + 1)
    return bus_to_idx[key]
end

function _register_phase_node!(phase_to_idx::Dict{Tuple{String, Int}, Int}, bus::AbstractString, phase::Int)
    key = (_normalize_bus_name(bus), phase)
    haskey(phase_to_idx, key) || (phase_to_idx[key] = length(phase_to_idx) + 1)
    return phase_to_idx[key]
end

"""
Phase edges between transformer/regulator windings.

When windings share a phase index, connect matching phases. Otherwise (open-delta /
split-phase banks) connect all modeled terminal pairs across the winding pair.
"""
function _build_transformer_phase_edges!(phase_to_idx::Dict{Tuple{String, Int}, Int}, network::NetworkModel)
    edges = Tuple{Int, Int}[]
    for tr in _coupling_devices(network)
        windings = tr.windings
        length(windings) < 2 && continue
        for i in 1:length(windings), j in (i + 1):length(windings)
            wi, wj = windings[i], windings[j]
            phases_i = FeederFlow.modeled_phases(wi.bus.phases; preserve_order=true)
            phases_j = FeederFlow.modeled_phases(wj.bus.phases; preserve_order=true)
            isempty(phases_i) && continue
            isempty(phases_j) && continue
            common = intersect(Set(phases_i), Set(phases_j))
            if !isempty(common)
                for ph in common
                    u = _register_phase_node!(phase_to_idx, wi.bus.bus, ph)
                    v = _register_phase_node!(phase_to_idx, wj.bus.bus, ph)
                    push!(edges, (u, v))
                end
            else
                for ph_i in phases_i, ph_j in phases_j
                    u = _register_phase_node!(phase_to_idx, wi.bus.bus, ph_i)
                    v = _register_phase_node!(phase_to_idx, wj.bus.bus, ph_j)
                    push!(edges, (u, v))
                end
            end
        end
    end
    return edges
end

"""Non-slack bus-phase nodes in FeederFlow `network_order` (matches OPF `n_net` nodes)."""
function _build_required_phase_nodes!(
    phase_to_idx::Dict{Tuple{String, Int}, Int},
    network::NetworkModel,
)
    ybus = FeederFlow.build_y(network; regulator_model=:nonideal)
    required = Int[]
    for bp in ybus.network_order
        idx = _register_phase_node!(phase_to_idx, bp.bus, bp.phase)
        push!(required, idx)
    end
    return sort(unique(required))
end

function build_bus_topology_context(case_path::AbstractString)
    network = FeederFlow.parse_file(case_path)

    line_bus_fr = String[]
    line_bus_to = String[]
    line_fr_node_idx = Vector{Vector{Int}}()
    line_to_node_idx = Vector{Vector{Int}}()

    phase_to_idx = Dict{Tuple{String, Int}, Int}()
    bus_to_idx = Dict{String, Int}()

    for line in network.lines
        push!(line_bus_fr, _normalize_bus_name(line.from.bus))
        push!(line_bus_to, _normalize_bus_name(line.to.bus))

        fr_nodes = Int[]
        to_nodes = Int[]
        line_phases = FeederFlow.modeled_phases(line.phases; preserve_order=true)
        for ph in line_phases
            push!(fr_nodes, _register_phase_node!(phase_to_idx, line.from.bus, ph))
            push!(to_nodes, _register_phase_node!(phase_to_idx, line.to.bus, ph))
        end
        push!(line_fr_node_idx, fr_nodes)
        push!(line_to_node_idx, to_nodes)

        _register_bus!(bus_to_idx, line.from.bus)
        _register_bus!(bus_to_idx, line.to.bus)
    end

    load_buses = Set(_normalize_bus_name(load.bus.bus) for load in network.loads)
    slack_bus = _normalize_bus_name(network.slack_bus)

    transformer_edges = _build_transformer_edges(network)
    slack_seed_buses = _substation_seed_buses(network, transformer_edges)

    for name in (line_bus_fr..., line_bus_to..., load_buses..., slack_seed_buses...)
        _register_bus!(bus_to_idx, name)
    end
    for (a, b) in transformer_edges
        _register_bus!(bus_to_idx, a)
        _register_bus!(bus_to_idx, b)
    end

    xf_phase_edges = _build_transformer_phase_edges!(phase_to_idx, network)

    for load in network.loads
        for ph in load.bus.phases
            _register_phase_node!(phase_to_idx, load.bus.bus, ph)
        end
        _register_bus!(bus_to_idx, load.bus.bus)
    end
    for bus in network.buses
        for ph in bus.phases
            _register_phase_node!(phase_to_idx, bus.name, ph)
        end
    end

    required_node_idx = _build_required_phase_nodes!(phase_to_idx, network)

    n_phase = length(phase_to_idx)
    node_keys = Vector{Tuple{String, Int}}(undef, n_phase)
    for (key, idx) in phase_to_idx
        node_keys[idx] = key
    end

    slack_seed_set = Set(_normalize_bus_name(b) for b in slack_seed_buses)
    slack_seed_idx = Int[]
    for idx in 1:n_phase
        bus, _ = node_keys[idx]
        bus in slack_seed_set && push!(slack_seed_idx, idx)
    end
    sort!(slack_seed_idx)

    n_bus = length(bus_to_idx)
    line_fr_idx = [bus_to_idx[b] for b in line_bus_fr]
    line_to_idx = [bus_to_idx[b] for b in line_bus_to]
    xf_fr_idx = Int[]
    xf_to_idx = Int[]
    sizehint!(xf_fr_idx, length(transformer_edges))
    sizehint!(xf_to_idx, length(transformer_edges))
    for (a, b) in transformer_edges
        push!(xf_fr_idx, bus_to_idx[a])
        push!(xf_to_idx, bus_to_idx[b])
    end
    slack_seed_bus_idx = sort(unique(bus_to_idx[b] for b in slack_seed_buses))
    load_bus_idx = sort(unique(bus_to_idx[b] for b in load_buses))

    return BusTopologyContext(
        slack_bus,
        slack_seed_buses,
        load_buses,
        line_bus_fr,
        line_bus_to,
        transformer_edges,
        bus_to_idx,
        n_bus,
        slack_seed_bus_idx,
        load_bus_idx,
        line_fr_idx,
        line_to_idx,
        xf_fr_idx,
        xf_to_idx,
        node_keys,
        n_phase,
        slack_seed_idx,
        required_node_idx,
        line_fr_node_idx,
        line_to_node_idx,
        xf_phase_edges,
    )
end

build_topology_graph_cache(case_path::AbstractString) = build_bus_topology_context(case_path)

function _build_bus_adjacency(status::AbstractVector{Bool}, ctx::BusTopologyContext)
    n = ctx.n_bus
    adj = [Int[] for _ in 1:n]

    n_lines = length(ctx.line_fr_idx)
    length(status) >= n_lines ||
        throw(ArgumentError("status length $(length(status)) < number of lines ($n_lines)"))

    for line_idx in 1:n_lines
        status[line_idx] || continue
        u = ctx.line_fr_idx[line_idx]
        v = ctx.line_to_idx[line_idx]
        push!(adj[u], v)
        push!(adj[v], u)
    end

    for k in 1:length(ctx.xf_fr_idx)
        u = ctx.xf_fr_idx[k]
        v = ctx.xf_to_idx[k]
        push!(adj[u], v)
        push!(adj[v], u)
    end

    return adj
end

function _build_phase_adjacency(status::AbstractVector{Bool}, ctx::BusTopologyContext)
    n = ctx.n_phase
    adj = [Int[] for _ in 1:n]

    n_lines = length(ctx.line_fr_node_idx)
    length(status) >= n_lines ||
        throw(ArgumentError("status length $(length(status)) < number of lines ($n_lines)"))

    for line_idx in 1:n_lines
        status[line_idx] || continue
        fr_nodes = ctx.line_fr_node_idx[line_idx]
        to_nodes = ctx.line_to_node_idx[line_idx]
        length(fr_nodes) == length(to_nodes) || continue
        for k in eachindex(fr_nodes)
            u = fr_nodes[k]
            v = to_nodes[k]
            push!(adj[u], v)
            push!(adj[v], u)
        end
    end

    for (u, v) in ctx.xf_phase_edges
        push!(adj[u], v)
        push!(adj[v], u)
    end

    return adj
end

function _bfs_reachable_idx!(adj::Vector{Vector{Int}}, seeds::AbstractVector{Int}, visited::AbstractVector{Bool})
    queue = Int[]
    sizehint!(queue, length(seeds))
    for s in seeds
        if !visited[s]
            visited[s] = true
            push!(queue, s)
        end
    end
    head = 1
    while head <= length(queue)
        u = queue[head]
        head += 1
        for v in adj[u]
            if !visited[v]
                visited[v] = true
                push!(queue, v)
            end
        end
    end
    return visited
end

function _reachable_edge_count_idx(adj::Vector{Vector{Int}}, visited::AbstractVector{Bool})
    seen = Set{Tuple{Int, Int}}()
    for u in eachindex(adj)
        visited[u] || continue
        for v in adj[u]
            visited[v] || continue
            edge = u < v ? (u, v) : (v, u)
            push!(seen, edge)
        end
    end
    return length(seen)
end

"""Phase connectivity: every required OPF bus-phase reachable from substation phase seeds."""
function _phase_connectivity_status(status::AbstractVector{Bool}, ctx::BusTopologyContext)
    adj = _build_phase_adjacency(status, ctx)
    has_edges = any(!isempty, adj)

    if isempty(ctx.required_node_idx)
        return has_edges
    end

    visited = Vector{Bool}(undef, ctx.n_phase)
    fill!(visited, false)
    _bfs_reachable_idx!(adj, ctx.slack_seed_idx, visited)
    return all(visited[i] for i in ctx.required_node_idx)
end

"""Bus-level radiality on the substation-fed subgraph (phase-aggregated lines and XFs)."""
function _bus_radial_status(status::AbstractVector{Bool}, ctx::BusTopologyContext)
    adj = _build_bus_adjacency(status, ctx)
    has_edges = any(!isempty, adj)

    if isempty(ctx.load_bus_idx)
        return has_edges
    end

    visited = Vector{Bool}(undef, ctx.n_bus)
    fill!(visited, false)
    _bfs_reachable_idx!(adj, ctx.slack_seed_bus_idx, visited)

    loads_connected = all(visited[i] for i in ctx.load_bus_idx)
    loads_connected || return false

    n_reachable = count(visited)
    n_reachable <= 1 && return true

    n_edges = _reachable_edge_count_idx(adj, visited)
    return n_edges == n_reachable - 1
end

"""
    _feeder_topology_status(status, ctx) -> (phase_connected::Bool, is_radial::Bool)

Phase connectivity plus bus-level structural radiality (matches OPF energization + feeder tree).
"""
function _feeder_topology_status(status::AbstractVector{Bool}, ctx::BusTopologyContext)
    phase_connected = _phase_connectivity_status(status, ctx)
    phase_connected || return false, false
    bus_radial = _bus_radial_status(status, ctx)
    return true, bus_radial
end

"""
    is_connected(status, ctx) -> Bool

True when every required non-slack bus-phase node is reachable from the substation
phase seeds through closed lines and transformer phase edges.
"""
function is_connected(status::AbstractVector{Bool}, ctx::BusTopologyContext)
    return _phase_connectivity_status(status, ctx)
end

is_loads_connected(ctx::BusTopologyContext, status::AbstractVector{Bool}) = is_connected(status, ctx)

"""
    is_radial(status, ctx) -> Bool

Radial when all required bus-phase nodes are energized and the bus-level subgraph is a tree.
"""
function is_radial(status::AbstractVector{Bool}, ctx::BusTopologyContext)
    phase_connected, is_radial_flag = _feeder_topology_status(status, ctx)
    return phase_connected && is_radial_flag
end

is_radial_feeder(ctx::BusTopologyContext, status::AbstractVector{Bool}) = is_radial(status, ctx)

function _repair_topology_ok(status::AbstractVector{Bool}, ctx::BusTopologyContext)
    phase_connected, is_radial_flag = _feeder_topology_status(status, ctx)
    return phase_connected && is_radial_flag
end

"""
    _minimal_nc_repair_core(status, repair_line_idx, perturb_line_idx, base_status, ctx; ...)

Open NC repair switches to restore phase-aware radiality after perturbation flips.
"""
function _minimal_nc_repair_core(
    status::Vector{Bool},
    repair_line_idx::Vector{Int},
    perturb_line_idx::Vector{Int},
    base_status::Vector{Bool},
    ctx::BusTopologyContext;
    repair_hints::Dict{String, String}=Dict{String, String}(),
    line_to_switch_name::Dict{Int, String}=Dict{Int, String}(),
    switch_name_to_line::Dict{String, Int}=Dict{String, Int}(),
)
    status = copy(status)
    opened = Int[]
    opened_via_hint = Int[]
    opened_via_greedy = Int[]
    repair_set = Set(repair_line_idx)

    for line in perturb_line_idx
        status[line] && !base_status[line] || continue
        sw_name = get(line_to_switch_name, line, "")
        isempty(sw_name) && continue
        hint_name = get(repair_hints, sw_name, "")
        isempty(hint_name) && continue
        repair_line = get(switch_name_to_line, hint_name, 0)
        repair_line == 0 && continue
        repair_line in repair_set || continue
        status[repair_line] || continue
        status[repair_line] = false
        push!(opened, repair_line)
        push!(opened_via_hint, repair_line)
    end

    if _repair_topology_ok(status, ctx)
        return status, opened, opened_via_hint, opened_via_greedy
    end

    for line in repair_line_idx
        _repair_topology_ok(status, ctx) && break
        line in opened && continue
        status[line] || continue
        trial = copy(status)
        trial[line] = false
        if _repair_topology_ok(trial, ctx)
            status = trial
            push!(opened, line)
            push!(opened_via_greedy, line)
        end
    end

    if _repair_topology_ok(status, ctx)
        return status, opened, opened_via_hint, opened_via_greedy
    end
    return nothing
end

"""
    minimal_nc_repair(status, ctx, repair_line_idx; kwargs...) -> Union{Nothing, Tuple}
"""
function minimal_nc_repair(
    status::Vector{Bool},
    ctx::BusTopologyContext,
    repair_line_idx::Vector{Int};
    repair_hints::Dict{String, String}=Dict{String, String}(),
    line_idx_to_name::Dict{Int, String}=Dict{Int, String}(),
    perturb_line_idx::Vector{Int}=Int[],
    base_status::Vector{Bool}=copy(status),
)
    switch_name_to_line = Dict{String, Int}()
    for (line, name) in line_idx_to_name
        switch_name_to_line[name] = line
    end

    result = _minimal_nc_repair_core(
        status,
        repair_line_idx,
        perturb_line_idx,
        base_status,
        ctx;
        repair_hints=repair_hints,
        line_to_switch_name=line_idx_to_name,
        switch_name_to_line=switch_name_to_line,
    )
    result === nothing && return nothing
    repaired_status, opened, opened_via_hint, opened_via_greedy = result
    return repaired_status, opened, opened_via_hint, opened_via_greedy
end
