using Random
using StatsFuns

export triple_edges, prune_edges, compute_flows, node_to_flows,
prune_edges_threshold, prune_edges_topk

function triple_edges(circuit::ProbCircuit; 
            double_fakesum=false, 
            normalize=true, 
            std=0.1, 
            min_random_noise=0.1)

    f_input(n) = begin
        if isliteral(n)
            return (n, n)
        else
            return (n, deepcopy(n))
        end
    end

    f_mul(n, cn) = begin # multiply does not create cross edges
        (multiply(first.(cn)), multiply(last.(cn)))
    end

    f_sum(n, cn) = begin
        if !double_fakesum && num_children(n) == 1
            # do not add cross edges
            (summate(first.(cn)), summate(last.(cn)))
        else
            # literal gates are shared
            islit = isliteral.(inputs(n))
            lits = inputs(n)[islit]
            new_children = [lits; first.(cn)[.!islit]; last.(cn)[.!islit]]
            new_sum = summate(new_children)
            old_sum = summate(new_children)
            
            # copy parameterss
            noise1 = randn(sum(islit)) * std .+ 1
            noise2 = randn(sum(.!islit)) * std .+ 1
            noise1[noise1 .< 0 ] .= min_random_noise
            noise2[noise2 .< 0 ] .= min_random_noise
            params = [n.params[islit] .* noise1; n.params[.!islit] .* noise2; n.params[.!islit] ./ noise2]
            if normalize
                params .-= logsumexp(params)
            end
            old_sum.params .= params
            new_sum.params .= params

            (old_sum, new_sum)
        end
    end

    new_r, old_r = foldup_aggregate(circuit, f_input, f_mul, f_sum, Tuple{ProbCircuit, ProbCircuit})
    new_r
    
end


function prune_edges(pc, train_x; by, heuristic, threshold)
    if by == "topk"
        pc, _ = prune_edges_topk(pc, heuristic, train_x; k=threshold)
    elseif by == "threshold"
        pc = prune_edges_threshold(pc, heuristic, train_x; threshold)
    end
    pc 
end


function prune_edges_func(circuit::ProbCircuit, func; debug=false)
    f_input(n) = (n, BitSet(literal(n)))

    f_mul(n, cs) = begin
        cn, clits = first.(cs), last.(cs)
        new_n = multiply(cn)

        lits = reduce(union, clits)
        (new_n, lits)
    end

    f_sum(n, cs) = begin
        cn, clits = first.(cs), last.(cs)

        if num_inputs(n) == 1
            return (summate(cn), clits[1])
        end

        # only prune (approximately) non-deterministic nodo
        remain = .!func(n, n.params)
        lits = reduce(union, clits)
        if !all(length(lits) .== length.(clits))
            @assert all(remain)
        end

        if !any(remain)
            remain[1] = true
        end
        new_n = summate(cn[remain])

        # copy parameters and normalize
        if sum(remain) == 1
            new_n.params .= 0.0
        else
            new_n.params .= n.params[remain]
            new_n.params .-= logsumexp(new_n.params)
        end

        (new_n, lits)

    end

    foldup_aggregate(circuit, f_input, f_mul, f_sum, Tuple{PlainProbCircuit, BitSet})[1]
end


function non_deterministic_nodes(circuit) 
    # approximately non-deterministic

    non_deterministic = Dict()
    sizehint!(non_deterministic, num_nodes(circuit))

    f_input(n) = BitSet(literal(n))

    f_mul(n, cs) = reduce(union, cs)

    f_sum(n, cs) = begin
        reduced = reduce(union, cs)
        non_deterministic[n] = num_inputs(n) > 1 && all(length(reduced) .== length.(cs))
        reduced
    end

    foldup_aggregate(circuit, f_input, f_mul, f_sum, BitSet)

    [x for x in sumnodes(circuit) if non_deterministic[x]]
end


function prune_edges_topk(circuit::ProbCircuit, heuristic, train_x=nothing; 
        k=nothing, cutnum_edge=nothing)
    @assert !isnothing(cutnum_edge) || (k >= 0.0 && k <= 1.0) "`K` has to be in the range [0, 1], or specify the number of edges to cut"
    
    node2flows = nothing

    if heuristic == :flows
        node2flows = node_to_flows(circuit, train_x; heuristic)
    end

    candidate_nodes =  non_deterministic_nodes(circuit)
    criterion = nothing

    if heuristic == :random
        criterion = vcat((map(x -> rand(length(x.params)), candidate_nodes))...)  
    elseif heuristic == :params
        criterion = vcat((map(x -> x.params, candidate_nodes))...) 
    elseif heuristic == :flows
        criterion = vcat((map(x -> node2flows[x], candidate_nodes))...) 
    else
          error("Heuristic $heuristic is not defined.")
    end
    
    begin
        inversep = sortperm(sortperm(criterion))
        flags_flat = falses(length(criterion))
        if isnothing(cutnum_edge)
            cut_off = Int(floor(length(criterion) * k))
        else
            cut_off = cutnum_edge
        end
        flags_flat[1:cut_off] .= true
        flags_flat = flags_flat[inversep]

        node2pruneflag = Dict()
        sizehint!(node2pruneflag, length(candidate_nodes))
        ind_begin=1
        for n in candidate_nodes
            node2pruneflag[n] = flags_flat[ind_begin:ind_begin+num_inputs(n)-1]
            ind_begin += num_inputs(n)
        end
        @assert ind_begin == length(criterion) + 1 "$ind_begin != $(length(params))"
        
        func(node, params) = begin
            get(node2pruneflag, node, falses(num_children(node)))
        end
    end
    prune_edges_func(circuit, func), sort(criterion)[1:cut_off]
end


function prune_edges_threshold(circuit::ProbCircuit, heuristic, train_x=nothing; 
            threshold=nothing)
    func = nothing
    node2flow = nothing
    if heuristic ∈ [:flows]
        node2flows = node_to_flows(circuit, train_x)
    end
    func_random(node, params) = rand(length(params)) .< threshold 
    func_params(node, params) = params .< log(threshold)
    func_flow(node, params) = begin
        @assert length(node2flows[node]) == length(params) "$(node2flows[node])"
        node2flows[node] .< threshold
    end
    
    if heuristic == :random
        func = func_random
    
    elseif heuristic == :params
        func = func_params
    
    elseif heuristic == :flows
        func = func_flow

    else
        error("Heuristic $heuristic is not defined.")
    end
    prune_edges_func(circuit, func)
end


using CUDA
using ProbabilisticCircuits: prep_memory, clear_input_node_mem, probs_flows_circuit
function compute_flows(bpc::CuBitsProbCircuit, data::CuArray; 
                       heuristic=nothing, flowflag=nothing, batch_size=1024)

    num_examples = size(data)[1]
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    
    marginals = prep_memory(nothing, (batch_size, num_nodes), (false, true))
    flows = prep_memory(nothing, (batch_size, num_nodes), (false, true))
    edge_aggr = prep_memory(nothing, (num_edges,))
    heuristic_aggr = prep_memory(nothing, (num_edges,))
    node_aggr = prep_memory(nothing, (num_nodes,))


    edge_aggr .= zero(Float32)
    heuristic_aggr .= zero(Float32)
    clear_input_node_mem(bpc; rate = 0)
    batch_index = 0

    for batch_start = 1:batch_size:num_examples
        batch_end = min(batch_start+batch_size-1, num_examples)
        batch = batch_start:batch_end
        batch_index += 1

        probs_flows_circuit(flows, marginals, edge_aggr, bpc, data, batch; 
                            mine=2, maxe=32, debug=false)
        @debug size(marginals), size(edge_aggr), batch
    end
    aggr_node_flows(node_aggr, bpc, edge_aggr)
    edge_aggr_cpu = Array(edge_aggr)
    node_aggr_cpu = Array(node_aggr)

    CUDA.unsafe_free!(edge_aggr)
    CUDA.unsafe_free!(node_aggr)
    CUDA.unsafe_free!(marginals)
    CUDA.unsafe_free!(flows)

    edge_aggr_cpu, node_aggr_cpu
end


function node_to_flows(pc, data; heuristic=nothing, debug=false, flowflag=0)
    bpc = BitsProbCircuit(pc)
    cubpc = CuBitsProbCircuit(bpc)

    node2flows = Dict()
    sizehint!(node2flows, length(bpc.nodes))

    edge_aggr = compute_flows(cubpc, data; heuristic, flowflag)[1]
    perm = sortperm(bpc.down2upedge) 

    if debug
        for (x1, x2) in zip(bpc.edge_layers_down.vectors[perm], bpc.edge_layers_up.vectors)
            if x2 isa ProbabilisticCircuits.SumEdge
                @assert x1.logp == x2.logp
            end
        end
    end
    edge_aggr = edge_aggr[perm]

    for (nodeid, bits_node) in enumerate(bpc.nodes)
        pc_node = bpc.nodes_map[nodeid]
        edge_ids = bpc.node_begin_end[nodeid]
        f = edge_aggr[edge_ids[1]:edge_ids[2]]
        if length(f) < 1 && debug
            @debug edge_ids[1], edge_ids[2]
        end
        node2flows[pc_node] = f
    end
    node2flows
end