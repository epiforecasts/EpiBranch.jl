module EpiBranchEnzymeCoreExt

using EpiBranch: EpiBranch
using EnzymeCore: EnzymeCore

function EnzymeCore.EnzymeRules.augmented_primal(
        ::EnzymeCore.EnzymeRules.RevConfig,
        ::EnzymeCore.Const{typeof(EpiBranch._simulated_chain_size_loglik)},
        ::Type{<:EnzymeCore.Annotation}, args::EnzymeCore.Annotation...
    )
    throw(
        ArgumentError(
            "Enzyme cannot differentiate a likelihood estimated by simulation " *
                "(interventions, a multi-type model or an offspring law without " *
                "a closed form). Use ForwardDiff, or a model with a closed form."
        )
    )
end

function EnzymeCore.EnzymeRules.reverse(
        ::EnzymeCore.EnzymeRules.RevConfig,
        ::EnzymeCore.Const{typeof(EpiBranch._simulated_chain_size_loglik)},
        dret, tape, args::EnzymeCore.Annotation...
    )
    return ntuple(_ -> nothing, length(args))
end

end
