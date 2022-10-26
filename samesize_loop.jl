using ArgParse
using Random
using CUDA
using JLD
using SparsePC

using LogicCircuits
using ProbabilisticCircuits
using ProbabilisticCircuits: read_fast


function samesize_loop_main(;seed, cuda_id, dir, # config
                datasetname, latents, by, heuristic, threshold, std, maxiter, noprune, nogrow, max_param, # alg
                verbose, batch_size, pseudocount, softness, # EM 
                num_epochs1, num_epochs2, num_epochs3, num_epochs4, # EM
                param_inertia1, param_inertia2, param_inertia3, param_inertia4,# EM
                logiter, patience, warmup, outpatience) # Early stop in em

    # seed, dir and device
    Random.seed!(seed)
    device!(cuda_id)
    if !isdir(dir)
        mkdir(dir)
    end
    if !isdir("$dir/ckpt")
        mkdir("$dir/ckpt")
    end
    
    # filenamestr
    filenamestr = "$dir/ckpt/$(datasetname)_z=$(latents)_pse=$(pseudocount)_std=$(std)_thre=$(threshold)_bz=$(batch_size)_" * 
        "p=$(max_param)_$(param_inertia1)_$(param_inertia2)_$(param_inertia3)_$(param_inertia4)_" * 
        "n=$(maxiter)_$(num_epochs1)_$(num_epochs2)_$(num_epochs3)_$(num_epochs4)"
    
    # load data and init pc
    train_x, valid_x, test_x = data_gpu(datasetname)
    data_summary(datasetname, train_x, valid_x, test_x)
    file = "circuits/$datasetname/$datasetname.z=$latents.jpc"
    if !isfile(file)
        file = file * ".gz"
    end

    @info "Load PC from $file"
    pc = read(file, ProbCircuit)
    ll1, ll2, ll3 = evaluateln_pc(pc, train_x, valid_x, test_x; latents, batch_size=512, prefix="Init-PC")

    # if distill
    if nogrow # distillation experiments
        minimal_ll = ll2 * 1.01
    else
        minimal_ll = -Inf
    end
    @info "Min Valid LL is $minimal_ll"

    pc = train(pc, train_x, valid_x, test_x, latents, filenamestr; by, heuristic, threshold, std, maxiter, noprune, nogrow,
                       verbose, batch_size, pseudocount, softness, # EM 
                       num_epochs1, num_epochs2, num_epochs3, num_epochs4, # EM
                       max_param, param_inertia1, param_inertia2, param_inertia3, param_inertia4,# EM
                       logiter, patience, warmup, outpatience, 
                       distill=nogrow, minimal_ll) # Early stop in em
    evaluateln_pc(pc, train_x, valid_x, test_x; latents, batch_size=512, prefix="Last-PC")
end

function train(pc, train_x, valid_x, test_x, latents, filenamestr; by, heuristic, threshold, std, maxiter, noprune, nogrow, # alg
                verbose, batch_size, pseudocount, softness, # EM 
                num_epochs1, num_epochs2, num_epochs3, num_epochs4, # EM
                max_param, param_inertia1, param_inertia2, param_inertia3, param_inertia4,# EM
                logiter, patience, warmup, outpatience,
                distill, minimal_ll) # Early stop in em



    valid_lls = []
    @time for i in 1:maxiter
        if !noprune
            @info "Pruning heuristic = $heuristic, threshold = $threshold, by = $by"
            pc = prune_edges(pc, train_x; by, heuristic, threshold)
            ll1, ll2, ll3 = evaluateln_pc(pc, train_x, valid_x, test_x; latents, batch_size, prefix="After-Prune")
        end
        
        if !nogrow
            @info "Growing std = $std"
            pc = triple_edges(pc; std)
            ll1, ll2, ll3 = evaluateln_pc(pc, train_x, valid_x, test_x; latents, batch_size, prefix="After-Grow")
        end

        if num_epochs1 != 0
            param_inertia1 = max_param
        end
        @info "Running EM... param_inertia = $param_inertia1 $param_inertia2 $param_inertia3 $param_inertia4"
       

        @time run_em(pc, train_x, valid_x, test_x; 
                verbose, batch_size, pseudocount, softness,
                num_epochs1, num_epochs2, num_epochs3, num_epochs4, 
                param_inertia1, param_inertia2, param_inertia3, param_inertia4,
                logiter, patience, warmup)
        
        ll1, ll2, ll3 = evaluateln_pc(pc, train_x, valid_x, test_x; latents, batch_size, prefix="After-EM")
        
        filename = filenamestr * "_iter=$i.jpc.gz"
        @info "Save ckpt to $filename"
        write(filename, pc)
        println()
        

        push!(valid_lls, ll2)
        idx = findmax(valid_lls)[2]
        if distill && ll2 < minimal_ll || (!distill && length(valid_lls) > idx + outpatience)
            if distill 
                idx = length(valid_lls)
            end
            @info "Early stopping loop. $(length(valid_lls)), $idx, $outpatience"
            cp(filenamestr * "_iter=$idx.jpc.gz", filenamestr * ".jpc.gz"; force=true)
            pc = read(filenamestr * "_iter=$idx.jpc.gz", ProbCircuit)
            return pc
        end
        println()
    end
    return pc
end

function run()
    samesize_loop_main(seed=1337, cuda_id=0, dir="exp", # config
        datasetname="mnist", latents=16, by="topk", heuristic=:flows, 
        threshold=0.75, std=0.5, maxiter=20, noprune=false, nogrow=false, # alg
        verbose=true, batch_size=512, pseudocount=0.01, softness=0.0, # EM 
        num_epochs1=50, num_epochs2=50, num_epochs3=50, num_epochs4=0, # EM
        max_param=0.7, param_inertia1=0.0, param_inertia2=0.9,
        param_inertia3=0.99, param_inertia4=0.999, # EM
        logiter=5, patience=20, warmup=0, outpatience=1) # Early stop in em
end

function distill()
    n=10
    samesize_loop_main(seed=1337, cuda_id=1, dir="exp", # config
        datasetname="mnist", latents=8, by="topk", heuristic=:flows, 
        threshold=0.01, std=0.5, maxiter=20, noprune=false, nogrow=true, # alg
        verbose=true, batch_size=512, pseudocount=0.01, softness=0.0, # EM 
        num_epochs1=n, num_epochs2=n, num_epochs3=n, num_epochs4=0, # EM
        max_param=0.7, param_inertia1=0.0, param_inertia2=0.9,
        param_inertia3=0.99, param_inertia4=0.999, # EM
        logiter=5, patience=20, warmup=0, outpatience=1) # Early stop in em
end

function read_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--datasetname"
            help = "Dataset name"
            required = true
        "--latents"
            help = "Nmber of hidden variables"
            arg_type = Int64
            default = 64
        "--by"
            help = ""
            arg_type = String
            default = "threshold"
        "--heuristic"
            help = "heuristic"
            arg_type = Symbol
            default = :flows
        "--threshold"
            arg_type = Float64
            default = 0.15
        "--maxiter", "--n"
            help = "Number of iterations."
            arg_type = Int64
            default = 10
        "--std"
            arg_type = Float64
            default = 0.1
        "--noprune"
            action = :store_true
        "--nogrow"
            action = :store_true


        "--seed"
            arg_type = Int
            default = 1337
        "--cuda_id"
            arg_type = Int
            default = 0
        "--dir"
            arg_type = String
            default = "exp"

        # EM hyper
        "--verbose"
            help = "Verbose"
            arg_type = Bool
            default = true
        "--batch_size"
            help = "batch_size"
            arg_type = Int64
            default = 512
        "--pseudocount"
            help = "pseudocount"
            arg_type = Float64
            default = 0.01
        "--softness"
            help = "softness"
            arg_type = Float64
            default = 0.
        "--num_epochs1", "--n1"
            help = "Number of iterations."
            arg_type = Int64
            default = 50#200
        "--num_epochs2", "--n2"
            help = "Number of iterations."
            arg_type = Int64
            default = 50#200
        "--num_epochs3", "--n3"
            help = "Number of iterations."
            arg_type = Int64
            default = 50#100
        "--num_epochs4", "--n4"
            help = "Number of iterations."
            arg_type = Int64
            default = 100
        "--param_inertia1", "--p1"
            arg_type = Float64
            default = 0.0
        "--param_inertia2", "--p2"
            arg_type = Float64
            default = 0.9
        "--param_inertia3", "--p3"
            arg_type = Float64
            default = 0.99
        "--param_inertia4", "--p4"
            arg_type = Float64
            default = 0.999
        "--max_param", "--pm"
            arg_type = Float64
            default = 1.0

        "--logiter"
            arg_type = Int
            default = 5
        "--patience"
            arg_type = Int
            default = 20
        "--warmup"
            arg_type = Int
            default = 50
        "--outpatience"
            arg_type = Int
            default = 0
        end
    return s
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_args(ARGS, read_args())
    @info args
    kwargs = Dict([(Symbol(k), v) for (k, v) in args])
    samesize_loop_main(;kwargs...)
end
