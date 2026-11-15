module EpiBranchForwardDiffExt

# The incidence likelihood gets the mean and covariance of the step counts from
# derivatives of a cumulant generating function. `ForwardDiff` supplies them
# here, so users who only simulate do not need it.

using EpiBranch: EpiBranch
using ForwardDiff: ForwardDiff

# Typed on `u` so it specialises the untyped fallback in core instead of
# overwriting it, which precompilation forbids.
function EpiBranch._gradient_and_hessian(K, u::AbstractVector{<:Real})
    return ForwardDiff.gradient(K, u), ForwardDiff.hessian(K, u)
end

# A value stripped of every dual level, for the ends of a bracket.
EpiBranch._plain(x::ForwardDiff.Dual) = EpiBranch._plain(ForwardDiff.value(x))

end
