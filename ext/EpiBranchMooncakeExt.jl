module EpiBranchMooncakeExt

using EpiBranch: EpiBranch
using Mooncake: Mooncake, CoDual

# Zero adjoints for functions Mooncake cannot differentiate and whose results
# do not depend on the parameters.
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch._resolved_key), Symbol}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch._validate_model_spec), Any, Any, Any}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch._reporting_closed_form_readable), Any, Any}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch.has_single_type_law), EpiBranch.TransmissionModel}
for f in (
        :validate_process_windows, :_validate_test_triggers, :_validate_testing_sensitivity,
        :_warn_unmet_trigger_requirements,
    )
    @eval Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch.$f), Any, Any}
end
for f in (
        :_warn_incomplete_terminal_coverage, :_validate_dose_schedule, :_store_traits!,
        :_warn_inert_interventions,
    )
    @eval Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch.$f), Any}
end

const _SimulateNCall = Tuple{
    typeof(Core.kwcall), NamedTuple, typeof(EpiBranch._simulate_n), Any, Type,
    EpiBranch.TransmissionModel, Int, EpiBranch.SimOpts,
}

Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode _SimulateNCall

function Mooncake.rrule!!(::CoDual{typeof(Core.kwcall)}, args::Vararg{CoDual, 7})
    throw(
        ArgumentError(
            "Mooncake cannot differentiate a simulation. A `loglikelihood` that has to " *
                "simulate (a model with interventions, say) is a noisy estimate with no " *
                "useful gradient: fit it with a sampler that needs only likelihood values, " *
                "such as `MH()`."
        )
    )
end

end
