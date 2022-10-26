using Pkg
Pkg.activate("$(@__DIR__)")
Pkg.add(url = "https://github.com/Juice-jl/ProbabilisticCircuits.jl.git", rev = "master")
Pkg.update()
Pkg.build()
Pkg.precompile()