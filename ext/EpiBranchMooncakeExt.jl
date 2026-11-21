module EpiBranchMooncakeExt

# A likelihood estimated by simulation is a noisy, piecewise-constant function
# of the model's parameters, so a reverse-mode gradient through it carries no
# information. Mooncake would otherwise fail inside the simulator with an error
# about an unsupported `ccall`; this rule says what is wrong instead.

using EpiBranch: EpiBranch
using Mooncake: Mooncake, CoDual

# A state key is a name, with nothing to differentiate, and the checks a
# `ModelSpec` runs on construction only read the model and return `nothing`.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(EpiBranch._resolved_key), Symbol}
for f in (
        :validate_process_windows, :_validate_test_triggers, :_validate_testing_sensitivity,
        :_warn_unmet_trigger_requirements,
    )
    @eval Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(EpiBranch.$f), Any, Any}
end
for f in (
        :_warn_incomplete_terminal_coverage, :_validate_dose_schedule, :_store_traits!,
        :_warn_inert_interventions,
    )
    @eval Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(EpiBranch.$f), Any}
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
