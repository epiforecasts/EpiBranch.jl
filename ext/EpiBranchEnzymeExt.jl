module EpiBranchEnzymeExt

# Enzyme compiles the simulation fallback even when it is never called, and
# cannot differentiate it.

using EpiBranch: EpiBranch
using Enzyme: Enzyme

Enzyme.EnzymeRules.inactive(::typeof(EpiBranch._simulated_chain_size_loglik), args...; kwargs...) = nothing

end
