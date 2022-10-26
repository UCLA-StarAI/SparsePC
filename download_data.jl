using MLDatasets

MLDatasets.MNIST()
for name in [:balanced, :byclass, :bymerge, :digits, :letters, :mnist]
    MLDatasets.EMNIST(name)
end
MLDatasets.FashionMNIST()