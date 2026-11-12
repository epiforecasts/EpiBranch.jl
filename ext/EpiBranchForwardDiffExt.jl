module EpiBranchForwardDiffExt

# The incidence likelihood reads the mean and covariance of the step counts off
# the derivatives of a cumulant generating function, which `ForwardDiff` takes.
# Loading it activates those derivatives, keeping it optional for users who
# only simulate.

using EpiBranch: EpiBranch
using ForwardDiff: ForwardDiff

function EpiBranch._gradient_and_hessian(K, u)
    return ForwardDiff.gradient(K, u), ForwardDiff.hessian(K, u)
end

# A value stripped of every dual level, for the ends of a bracket.
EpiBranch._plain(x::ForwardDiff.Dual) = EpiBranch._plain(ForwardDiff.value(x))

end
