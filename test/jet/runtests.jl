using Pkg
Pkg.develop(path = joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using JET
using EpiBranch

JET.test_package(EpiBranch; target_modules = (EpiBranch,))
