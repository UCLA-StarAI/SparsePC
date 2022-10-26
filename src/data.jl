using MLDatasets, CUDA, DataFrames, CSV, Random, MLBase

export data_cpu, data_gpu, DATASETS

TWENTY_DATASETS_NAMES = [
        "nltcs", "msnbc", "kdd", "plants", "baudio", 
        "jester", "bnetflix", "accidents", "tretail", "pumsb_star",
        "dna", "kosarek", "msweb", "book", "tmovie", 
        "cwebkb", "cr52", "c20ng", "bbc", "ad", 
        "binarized_mnist"]

ML_DATASETS = [["mnist", "fashionmnist"]; 
                "emnist_" .* ["mnist", "letters", "balanced", "byclass", "bymerge", "digits"];
               ["cifar10", "svhn2"]]

TEXT_DATASETS = ["ptbchar_288"]

DATASETS = [TWENTY_DATASETS_NAMES;["bits_mnist"]; 
            ML_DATASETS; TEXT_DATASETS;]


function data_cpu(name; split=true, data_dir="./datasets", kwargs...)

    if name in TWENTY_DATASETS_NAMES
        return Matrix.(twenty_datasets(name))
    
    elseif name == "bits_mnist"
        return bits_mnist_cpu()
    
    elseif name in ML_DATASETS
        return ml_datasets(name; split)

    elseif name in TEXT_DATASETS
        return  load_local_data(name; data_dir)

    else
        return load_local_data(name; data_dir)
    end
end


function data_gpu(name; kwargs...)
    cu.(data_cpu(name; kwargs...))
end


function bits_mnist_cpu()
    train_int = transpose(reshape(MNIST.traintensor(UInt8), 28*28, :));
    test_int = transpose(reshape(MNIST.testtensor(UInt8), 28*28, :));

    function bitsfeatures(data_int)
        data_bits = zeros(Bool, size(data_int,1), 28*28*8)
        for ex = 1:size(data_int,1), pix = 1:size(data_int,2)
            x = data_int[ex,pix]
            for b = 0:7
                if (x & (one(UInt8) << b)) != zero(UInt8)
                    data_bits[ex, (pix-1)*8+b+1] = true
                end
            end
        end
        data_bits
    end

    train_cpu = bitsfeatures(train_int);
    test_cpu = bitsfeatures(test_int);

    train_cpu, test_cpu
end


function ml_datasets(name; split=true)
    train_cpu, test_cpu = nothing, nothing
    if name == "mnist"
        train_cpu, test_cpu = map([:train, :test]) do x
            data = MNIST(Tx=UInt8, split=x).features
            @assert size(data)[1] == 28
            @assert size(data)[2] == 28
            collect(reshape(permutedims(data, (3, 1, 2)), :, 28*28))
        end
    elseif name == "fashionmnist"
        train_cpu, test_cpu = map([:train, :test]) do x
            data = FashionMNIST(Tx=UInt8, split=x).features
            @assert size(data)[1] == 28
            @assert size(data)[2] == 28
            collect(reshape(permutedims(data, (3, 2, 1)), :, 28*28))
        end
    elseif startswith(name, "emnist")
        subname = Base.split(name,"_")[2]
        train_cpu, test_cpu = map([:train, :test]) do x
            data = EMNIST(Symbol(subname); Tx=UInt8, split=x).features
            @assert size(data)[1] == 28
            @assert size(data)[2] == 28
            collect(reshape(permutedims(data, (3, 2, 1)), :, 28*28))
        end
    elseif name == "cifar10"
        train_cpu, test_cpu = map([:train, :test]) do x
            data = CIFAR10(Tx=UInt8, split=x).features
            @assert size(data)[1] == 32
            @assert size(data)[2] == 32
            @assert size(data)[3] == 3
            collect(reshape(permutedims(data, (4, 1, 2, 3)), :, 32*32*3))
        end
    elseif name == "svhn2"
        train_cpu, test_cpu = map([:train, :test]) do x
            data = SVHN2(Tx=UInt8, split=x).features
            @assert size(data)[1] == 32
            @assert size(data)[2] == 32
            @assert size(data)[3] == 3
            collect(reshape(permutedims(data, (4, 3, 2, 1)), :, 32*32*3))
        end
    else
        error("Dataset $name not found.")
    end
    if split
        split_valid(train_cpu)..., test_cpu
    else
        @warn "$name does not have a validation set"
        train_cpu, nothing, test_cpu
    end
end


function split_valid(data, prop=0.05)
    indices = collect(1:size(data, 1))
    randperm!(indices)
    num_train = Int(floor(size(data, 1) * (1 - prop)))
    data[indices[1:num_train], :], data[indices[num_train+1:end], :]
end


function load_local_data(name; data_dir="./datasets")
    function load(type)
        dataframe = CSV.read(data_dir*"/$name/$name.$type.data", DataFrame; 
            header=false, types=UInt8)
        Tables.matrix(dataframe)
    end
    train = load("train")
    valid = load("valid")
    test = load("test")
    return train, valid, test
end