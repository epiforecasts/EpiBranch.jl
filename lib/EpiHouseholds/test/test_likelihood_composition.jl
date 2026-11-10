include(joinpath(@__DIR__, "testutils", "structured_likelihood.jl"))
test_structured_composition(k -> HouseholdProcess([2], k), household_infections)
