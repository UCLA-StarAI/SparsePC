using Logging

using ProbabilisticCircuits: num_parameters_node
export literal, isliteral, isinput, truncate, bits_per_dim, 
data_summary, report_ll, evaluate_pc, evaluateln_pc, mkdir_safe,
debug!, no_debug!

isliteral(n::PlainProbCircuit) = false
isliteral(n::PlainInputNode{Literal}) = true
isinput(n::PlainProbCircuit) = false
isinput(n::PlainInputNode) = true

import LogicCircuits: literal, variable
literal(n::PlainInputNode{Literal}) = begin
    if n.dist.value
        return collect(n.randvars)[1]
    else
        return - collect(n.randvars)[1]
    end
end

literal(n::PlainInputNode) = begin
    return collect(randvars(n))[1]
end


import ProbabilisticCircuits: loglikelihood
function loglikelihood(pc::ProbCircuit, data; batch_size=512)
    loglikelihood(CuBitsProbCircuit(pc), data; batch_size)
end


function truncate(data::Matrix; bits)
    data .÷ 2^bits
end


bits_per_dim(ll, data) = begin
    -(ll  / size(data, 2)) / log(2)
end


function data_summary(name, datas...)
    println("Dataset $(name) summary: ")
    num_cats = maximum(datas[1]) + 1
    println(" - Number of variables: $(size(datas[1], 2))")
    for data in datas
        if issomething(data)
            println(" - Number of examples: $(size(data, 1))")
        end
    end
    println(" - Number of categories: $(num_cats)")
end


function report_ll(bpc, train_x, valid_x, test_x; batch_size) 
    for (str, data) in zip(["train", "valid", "test"], [train_x, valid_x, test_x])
        if !isnothing(data)
            ll = loglikelihood(bpc, data; batch_size)
            println("  $str LL: $(ll), $(bits_per_dim(ll, data))")
        else
            println("  $str LL: nothing")
        end
    end
end


function evaluateln_pc(pc, train_x, valid_x, test_x;
                       latents=nothing, batch_size=512, prefix="", other=true)
        evaluate_pc(pc, train_x, valid_x, test_x; 
            latents, batch_size, prefix, other, ln=true)
end


function evaluate_pc(pc, train_x, valid_x, test_x; 
                     latents, batch_size, prefix="", other=true, ln=false)
    bpc = CuBitsProbCircuit(pc)
    ll1 = loglikelihood(bpc, train_x; batch_size)
    ll2, ll3 = nothing, nothing
    if valid_x!== nothing
        ll2 = loglikelihood(bpc, valid_x; batch_size)
    end
    ll3 = loglikelihood(bpc, test_x; batch_size)
    if other
        println("$prefix; # Latents: $latents")
        println("$prefix; # parameters: $(num_parameters(pc))")
        sum_p = (sum(n -> num_parameters_node(n, true), sumnodes(pc)))
        println("$prefix; # sum parameters: $sum_p")
    end
    println("$prefix; train ll: $ll1")
    if valid_x!== nothing
        println("$prefix; valid ll: $ll2")
    end
    println("$prefix; test ll: $ll3")
    ln && println()
    return ll1, ll2, ll3
end


function mkdir_safe(dir)
    if !isdir(dir)
        mkdir(dir)
    end
end