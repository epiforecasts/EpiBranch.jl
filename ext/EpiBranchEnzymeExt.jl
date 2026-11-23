module EpiBranchEnzymeExt

# Enzyme cannot compile these checks and lookups in the `ModelSpec`
# constructor. They return nothing or a symbol and have no derivative.

using EpiBranch: EpiBranch
using Enzyme: Enzyme
using Enzyme.EnzymeRules: EnzymeRules

EnzymeRules.inactive(::typeof(EpiBranch.validate_process_windows), args...) = nothing

for f in (
        :_warn_incomplete_terminal_coverage, :_validate_dose_schedule,
        :_validate_test_triggers, :_validate_testing_sensitivity,
        :_warn_unmet_trigger_requirements, :_warn_inert_interventions,
    )
    @eval EnzymeRules.inactive(::typeof(EpiBranch.$f), args...) = nothing
end

EnzymeRules.inactive(::typeof(EpiBranch._resolved_key), args...) = nothing

end
