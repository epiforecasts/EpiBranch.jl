module EpiBranchMooncakeExt

# Zero adjoints for functions Mooncake cannot differentiate and whose results
# do not depend on the parameters.

using EpiBranch: EpiBranch
using Mooncake: Mooncake

Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch._resolved_key), Symbol}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch._reporting_closed_form_readable), Any, Any}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch.has_single_type_law), Any}

for f in EpiBranch._MODEL_CHECKS
    @eval Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof($f), Vararg{Any}}
end

end
