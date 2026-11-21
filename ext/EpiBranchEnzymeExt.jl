module EpiBranchEnzymeExt

# The closed-form chain-size likelihood differentiates under Enzyme, but the
# simulation fallback sitting beside it in the same function does not, and
# Enzyme compiles both. The fallback is a random estimate, so it is marked as
# carrying no derivative and left out of compilation.

using EpiBranch: EpiBranch
using Enzyme: Enzyme

Enzyme.EnzymeRules.inactive(::typeof(EpiBranch._simulated_chain_size_loglik), args...; kwargs...) = nothing

end
