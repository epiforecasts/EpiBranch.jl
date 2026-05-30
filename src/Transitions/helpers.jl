# ── Heterogeneity helpers for clinical transitions ──────────────────
#
# Transition fields (`probability`, `delay`) accept three shapes,
# resolved per individual at `resolve_individual!` time:
#
# - a `Real` / `Distribution`: constant across the population.
# - a `Function (rng, ind) -> value`: arbitrary per-individual rule.
#   Use this for age-dependent CFRs, vulnerability-conditioned delays,
#   risk-group-specific reporting, etc. The function is called with the
#   simulation RNG and the individual; return the probability (as a
#   `Real`) or the delay (as a `Real` time, typically days).
#
# Distribution-valued delays sample from the distribution. Function
# delays return a sample directly.
_resolve_probability(p::Real, rng, ind) = float(p)
_resolve_probability(f, rng, ind) = float(f(rng, ind))

_resolve_delay(d::Distribution, rng, ind) = float(rand(rng, d))
_resolve_delay(f, rng, ind) = float(f(rng, ind))

# Anchor for a transition's `delay`. A `Symbol` is looked up in
# `ind.state` (e.g. `:onset_time`, `:test_time`, `:admission_time`); a
# `Function (ind) -> Real` is called directly (use this for fields on
# `Individual` itself, e.g. `ind.infection_time`, or for composite
# anchors). Returning `NaN` signals "no anchor" and the transition is
# skipped.
_resolve_anchor(s::Symbol, ind) = get(ind.state, s, NaN)::Float64
_resolve_anchor(f, ind) = float(f(ind))

# By default each transition's `required_fields` returns `[:onset_time]`
# so the simulation start-up validator catches missing
# `clinical_presentation` setup. When the user overrides `from` to
# anchor on something other than onset, the required field is the
# user's responsibility — typically a downstream transition's output
# key (e.g. `:test_time`) that isn't set by attributes anyway.
_from_required(s::Symbol) = s === :onset_time ? [:onset_time] : Symbol[]
_from_required(_) = Symbol[]

"""
    _finalise_terminal!(individual, transitions)

After all `resolve_individual!`s have run, collect terminal candidates
across all terminal transitions and set `:outcome` and `:outcome_time`
to the earliest. If no terminal transition fires, neither key is set.
"""
function _finalise_terminal!(individual, transitions)
    best_time = Inf
    best_label = :none
    has_any = false
    for t in transitions
        is_terminal(t) || continue
        ev = terminal_event(t, individual)
        ev === nothing && continue
        time, label = ev
        if time < best_time
            best_time = time
            best_label = label
            has_any = true
        end
    end
    if has_any
        individual.state[:outcome_time] = best_time
        individual.state[:outcome] = best_label
    end
    return nothing
end
