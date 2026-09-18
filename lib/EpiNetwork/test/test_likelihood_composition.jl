include(joinpath(
    @__DIR__, "..", "..", "..", "test", "testutils", "structured_likelihood.jl"))
test_structured_composition(k -> NetworkProcess([[2], [1]], k), network_infections)
