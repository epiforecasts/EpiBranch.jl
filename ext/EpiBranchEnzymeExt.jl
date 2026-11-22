module EpiBranchEnzymeExt

# Enzyme compiles the simulation fallback even when it is never called, and
# cannot differentiate it. These rules stop it being compiled. A call that does
# reach it throws an error instead of returning a zero gradient.

using EpiBranch: EpiBranch
using Enzyme: Enzyme
using Enzyme.EnzymeRules: EnzymeRules

function _not_differentiable()
    throw(
        ArgumentError(
            "this model has no closed-form chain-size likelihood, so it is " *
                "estimated by simulation, which Enzyme cannot differentiate"
        )
    )
end

# The window check only warns and returns nothing, so there is nothing to
# differentiate. Enzyme fails to compile its loops on some Julia versions.
EnzymeRules.inactive(::typeof(EpiBranch.validate_process_windows), args...) = nothing

function EnzymeRules.augmented_primal(
        config, func::Enzyme.Const{typeof(EpiBranch._simulated_chain_size_loglik)},
        ::Type{RT}, args...
    ) where {RT}
    return _not_differentiable()
end

function EnzymeRules.reverse(
        config, func::Enzyme.Const{typeof(EpiBranch._simulated_chain_size_loglik)},
        dret, tape, args...
    )
    return _not_differentiable()
end

end
