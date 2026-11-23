module EpiBranchMooncakeExt

using EpiBranch: EpiBranch
using Mooncake: Mooncake, CoDual

# Zero adjoints for functions that Mooncake cannot differentiate and that
# return results independent of the parameters.
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch._resolved_key), Symbol}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch._reporting_closed_form_readable), Any, Any}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch.has_single_type_law), Any}

for f in EpiBranch._MODEL_CHECKS
    @eval Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof($f), Vararg{Any}}
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
