include(joinpath(
    @__DIR__, "..", "..", "..", "test", "testutils", "structured_likelihood.jl"))
test_structured_composition(k -> HouseholdProcess([2], k), household_infections)
