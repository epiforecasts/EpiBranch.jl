# ── Internal likelihood implementations ──────────────────────────────
# These are called by the unified loglikelihood() interface in fitting.jl.

"""
    _chain_length_ll_pgf(data, G, ::Type{Tel}; n_initial = 1)

Analytical chain-length log-likelihood for a branching process whose
offspring PGF is `G`. Iterating `q[n] = G(q[n-1])` from `q[1] = G(0)`
gives `q[len + 1] = P(length ≤ len)` for a single seed. A chain with
`n_initial` independent seeds has the length of the longest of `n_initial`
single-seed chains: its CDF is `q[len + 1]^n_initial`, and `P(length =
len)` is the successive difference of that. Element-type generic in `Tel`
(the offspring parameter type) so `ForwardDiff` gradients flow through.
"""
function _chain_length_ll_pgf(data, G, ::Type{Tel}; n_initial::Int = 1) where {Tel}
    n_initial >= 1 || throw(ArgumentError("n_initial must be ≥ 1"))

    max_len = maximum(data)

    q = Vector{Tel}(undef, max_len + 1)
    q[1] = G(zero(Tel))
    for n in 2:(max_len + 1)
        q[n] = G(q[n - 1])
    end

    # `len` ranges over `data`, so `len ≤ max_len` always holds and every
    # index below is in bounds.
    ll = zero(Tel)
    for len in data
        if len == 0
            ll += n_initial * log(q[1])
        else
            prob = q[len + 1]^n_initial - q[len]^n_initial
            prob <= zero(prob) && return oftype(ll, -Inf)
            ll += log(prob)
        end
    end
    return ll
end

"""Analytical chain length likelihood for NegBin offspring."""
function _chain_length_ll_negbin(data, offspring::NegativeBinomial; n_initial::Int = 1)
    k = offspring.r
    R = mean(offspring)
    R < 1.0 ||
        throw(ArgumentError("chain length distribution only defined for subcritical process (R < 1)"))

    p = k / (k + R)
    G(s) = (p / (one(p) - (one(p) - p) * s))^k
    return _chain_length_ll_pgf(data, G, typeof(p); n_initial)
end

# ── Utilities ────────────────────────────────────────────────────────

"""
    logsumexp(x)

Numerically stable log-sum-exp. Any iterable is accepted (generators
are collected first to allow two-pass computation).
"""
function logsumexp(x)
    v = x isa AbstractVector ? x : collect(Float64, x)
    isempty(v) && return -Inf
    mx = maximum(v)
    isinf(mx) && return mx
    return mx + log(sum(exp(xi - mx) for xi in v))
end
