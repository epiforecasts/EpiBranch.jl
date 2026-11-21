module EpiBranchEnzymeExt

using EpiBranch: EpiBranch
using Enzyme: Annotation, Const, EnzymeRules

for f in EpiBranch._MODEL_CHECKS
    @eval EnzymeRules.inactive(::typeof($f), args...) = nothing
end

# Symbol and Bool lookups with no derivative. Left active, Enzyme on
# Julia 1.10 reports their constant result as stored into a differentiable value.
EnzymeRules.inactive(::typeof(EpiBranch._resolved_key), args...) = nothing
EnzymeRules.inactive(::typeof(EpiBranch._closed_form_reportable), args...) = nothing
EnzymeRules.inactive(::typeof(EpiBranch.has_single_type_law), args...) = nothing

function _refuse_simulation()
    throw(
        ArgumentError(
            "this likelihood is estimated by simulation, which Enzyme cannot " *
                "differentiate in reverse mode; use a model with a closed-form " *
                "likelihood, or forward mode"
        )
    )
end

for f in (EpiBranch._simulated_chain_size_loglik, EpiBranch._simulated_chain_length_loglik)
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfig, ::Const{typeof($f)}, ::Type, args::Annotation...
            )
            return _refuse_simulation()
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfig, ::Const{typeof($f)}, dret, tape, args::Annotation...
            )
            return _refuse_simulation()
        end
    end
end

end
