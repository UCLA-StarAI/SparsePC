using ArgParse
using SparsePC
using ProbabilisticCircuits: ProbCircuit
using Random

if abspath(PROGRAM_FILE) == @__FILE__
    indir = "exp/hclt"
    outdir = "circuits/hclt"
    results = Dict()
    for datasetname in ["mnist", "emnist_mnist", "emnist_letters", 
        "emnist_balanced", "emnist_byclass", "fashionmnist", "ptbchar_288"]
        train_x, valid_x, test_x = data_gpu(datasetname)
        mkdir_safe(outdir *  "/" * datasetname)
        for latents in [8, 16, 32, 64, 96, 128]
            println("$datasetname $latents")
            best_ll, best_file = (-Inf, -Inf, -Inf), nothing
            for filename in readdir("$indir/$datasetname/$latents/")
                if endswith(filename, "jpc.gz")
                    pathname = "$indir/$datasetname/$latents/$filename"
                    println("Reading $pathname")
                    pc = read(pathname, ProbCircuit)
                    ll1, ll2, ll3 = evaluate_pc(pc, train_x, valid_x, test_x; latents, batch_size=512)
                    if ll2 > best_ll[2]
                        best_ll = (ll1, ll2, ll3)
                        best_file = pathname
                    end
                end
            end
            bpp = bits_per_dim(best_ll[1], train_x), bits_per_dim(best_ll[2], train_x), bits_per_dim(best_ll[3], train_x)
            results["$(datasetname)_z=$latents"] = bpp, best_file
            
            println("$latents\t$(bpp[1])\t$(bpp[2])\t$(bpp[3])\t$(best_file)\n")
            if !isnothing(best_file)
                cp(best_file, "$outdir/$datasetname/$datasetname.z=$latents.jpc.gz",force=true)
            end
        end
    end


    for datasetname in ["mnist", "emnist_mnist", "emnist_letters", 
        "emnist_balanced", "emnist_byclass", "fashionmnist", "ptbchar_288"]
        println(datasetname)
        for latents in [8, 16, 32, 64, 96, 128]
            run_r = results["$(datasetname)_z=$latents"]
            println("$latents\t$(run_r[1][1])\t$(run_r[1][2])\t$(run_r[1][3])\t$(run_r[2])\n")
        end
    end
end
