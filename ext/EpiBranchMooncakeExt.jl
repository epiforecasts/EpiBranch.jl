module EpiBranchMooncakeExt

# Mooncake cannot differentiate through `try` blocks or `Symbol` to `String`
# conversions. These functions return only a name or a check result that does
# not depend on the parameters, so reverse mode skips them. That lets a model
# be built inside the function being differentiated.

using EpiBranch: EpiBranch
using Mooncake: Mooncake

Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch._resolved_key), Symbol}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch._validate_model_spec), Any, Any, Any}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(EpiBranch._reporting_closed_form_readable), Any, Any}

end
