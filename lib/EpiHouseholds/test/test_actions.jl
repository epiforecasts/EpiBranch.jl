include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "continuous_actions.jl"))
test_continuous_vaccine_actions(k -> HouseholdProcess([4], k))
