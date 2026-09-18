include(joinpath(@__DIR__, "..", "..", "..", "test", "testutils", "calendar_kernels.jl"))
test_calendar_simulation(k -> HouseholdProcess([3], k), household_infections)
