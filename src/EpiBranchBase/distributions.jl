# ── Trio-dispatch helper ────────────────────────────────────────────
# `_sample_value` is the only distribution-related thing the protocol
# layer needs. Concrete distribution helpers (NegBin, scaled
# distributions, generation-time builders) live in `Engine`.

"""
    _sample_value(x, rng, args...) -> Float64

Resolve a value that can be a `Real`, a `Distribution`, or a callable.
The callable is invoked as `f(rng, args...)` — callers choose the
signature by what they pass after `rng`. The return is always
converted to `Float64`.

Used throughout the package for parameters that accept the same
"scalar | distribution | function" trio: attribute builders,
intervention parameters, and competing-risk fields ([`Risk`](@ref)).
"""
_sample_value(x::Real, rng, args...) = float(x)
_sample_value(d::Distribution, rng, args...) = float(rand(rng, d))
_sample_value(f, rng, args...) = float(f(rng, args...))
