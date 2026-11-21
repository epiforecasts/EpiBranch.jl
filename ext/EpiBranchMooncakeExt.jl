module EpiBranchMooncakeExt

# Building a model resolves state names into dictionary keys, which builds
# symbols through a `ccall` that Mooncake has no rule for. The keys carry no
# derivative, so they are marked as such. So are the checks that run when a
# model is built or a likelihood is formed (they return nothing or a flag, and
# warn through `try` blocks that Mooncake cannot differentiate).

using EpiBranch: EpiBranch
using Mooncake: Mooncake

Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch._resolved_key), Symbol}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch._closed_form_reportable), Any, Any}

for f in EpiBranch._MODEL_CHECKS
    @eval Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof($f), Vararg{Any}}
end

end
