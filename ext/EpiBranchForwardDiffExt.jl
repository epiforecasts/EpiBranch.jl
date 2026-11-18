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

# A fixed point has converged only once its derivatives have as well.
function EpiBranch._fixed_point_gap(a::ForwardDiff.Dual, b::ForwardDiff.Dual)
    gap = EpiBranch._fixed_point_gap(ForwardDiff.value(a), ForwardDiff.value(b))
    for (pa, pb) in zip(ForwardDiff.partials(a), ForwardDiff.partials(b))
        gap = max(gap, EpiBranch._fixed_point_gap(pa, pb))
    end
    return gap
end

end
