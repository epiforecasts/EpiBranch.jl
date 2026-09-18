include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "continuous_actions.jl"))
test_continuous_vaccine_actions(k -> NetworkProcess(
    [[j for j in 1:4 if j != i]
     for i in 1:4], k))
