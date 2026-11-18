module EpiBranchForwardDiffExt

# `incidence_moments` gets the mean and covariance of the step counts from
# derivatives of a cumulant generating function. `ForwardDiff` supplies them
# here, so users who only simulate do not need it.

using EpiBranch: EpiBranch
using ForwardDiff: ForwardDiff

# Typed on `u` to specialise the untyped fallback in core. An untyped method
# would overwrite the fallback, and precompilation forbids that.
function EpiBranch._gradient_and_hessian(K, u::AbstractVector{<:Real})
    return ForwardDiff.gradient(K, u), ForwardDiff.hessian(K, u)
end

# A value stripped of every dual level, for the ends of a bracket.
EpiBranch._plain(x::ForwardDiff.Dual) = EpiBranch._plain(ForwardDiff.value(x))

end
