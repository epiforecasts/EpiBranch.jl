include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "stateful_kernels.jl"))
test_stateful_simulation(k -> NetworkProcess([[2, 3], [1, 3], [1, 2]], k), network_infections)
