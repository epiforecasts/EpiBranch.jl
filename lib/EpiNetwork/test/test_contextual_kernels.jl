include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "contextual_kernels.jl"))
test_contextual_simulation(k -> NetworkProcess([[2, 3], [1, 3], [1, 2]], k), network_infections)
