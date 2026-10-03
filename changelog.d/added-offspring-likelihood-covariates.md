`loglikelihood(::OffspringCounts, ::AbstractVector{<:Distribution})` scores
each secondary case count against its own offspring distribution, for
case-level covariates such as `NegBin.(exp.(X * β), k)`. Composes with
`Distributions.truncated` for zero-truncated data and with
`Distributions.product_distribution` for direct use on the right-hand side
of Turing's `~`.
