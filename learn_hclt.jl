using ArgParse
using SparsePC
using Random

if abspath(PROGRAM_FILE) == @__FILE__
    s = ArgParseSettings()
    @add_arg_table s begin
        "--datasetname"
            help = "Dataset name"
            required = true
        "--nosplit"
            help = "Whether split train data into train and valid"
            action = :store_true
        "--k_fold"
            help = "Cross validation id"
            arg_type = Int64
            default = nothing

        "--latents"
            help = "Nmber of hidden variables"
            arg_type = Int64
            default = 16

        "--structure"
            help = "Structure of circuit"
            arg_type = String
            default = "hclt"
        "--save_circuit"
            action = :store_true

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
            default = 0.005
        "--softness"
            help = "softness"
            arg_type = Float64
            default = 0.
        "--num_epochs1", "--n1"
            help = "Number of iterations."
            arg_type = Int64
            default = 200
        "--num_epochs2", "--n2"
            help = "Number of iterations."
            arg_type = Int64
            default = 200
        "--num_epochs3", "--n3"
            help = "Number of iterations."
            arg_type = Int64
            default = 100
        "--num_epochs4", "--n4"
            help = "Number of iterations."
            arg_type = Int64
            default = 100
        "--param_inertia1", "--p1"
            arg_type = Float64
            default = 0.2
        "--param_inertia2", "--p2"
            arg_type = Float64
            default = 0.9
        "--param_inertia3", "--p3"
            arg_type = Float64
            default = 0.95
        "--param_inertia4", "--p4"
            arg_type = Float64
            default = 0.999
        
        # other
        "--seed"
            help = "Random seed"
            arg_type = Int64
            default = 1337
        "--dir"
            help = "Random seed"
            arg_type = String
            default = "exp"
        
        # early stopping
        "--logiter"
            arg_type = Int
            default = 5
        "--patience"
            arg_type = Int
            default = 20
        "--warmup"
            arg_type = Int
            default = 50
        "--is_earlystop"
            action = :store_true
        end
    args = parse_args(ARGS, s)
    println(args)
    
    kwargs = Dict([(Symbol(k), v) for (k, v) in args])
    Random.seed!(args["seed"])
    pc = nothing
    
    # filename
    filename = "$(args["dir"])/$(args["datasetname"])"
    if !isnothing(args["k_fold"])
        filename = filename * "_$(args["k_fold"])"
    end
    filename = filename * "_$(args["latents"])_pse=$(args["pseudocount"])" * 
        "_soft=$(args["softness"])_bz=$(args["batch_size"])" * 
        "_p=$(args["param_inertia1"])_$(args["param_inertia2"])_$(args["param_inertia3"])_$(args["param_inertia4"])" * 
        "_n=$(args["num_epochs1"])_$(args["num_epochs2"])_$(args["num_epochs3"])_$(args["num_epochs4"]).jpc.gz"
    
    println("Save file to $filename")
    @time pc = train_hclt(;kwargs...)
    if args["save_circuit"]
        
        write(filename, pc)
    end
end
