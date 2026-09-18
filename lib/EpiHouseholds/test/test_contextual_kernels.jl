include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "contextual_kernels.jl"))
test_contextual_simulation(k -> HouseholdProcess([3], k), household_infections)
