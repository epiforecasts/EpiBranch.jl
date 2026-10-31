# ── Serial interval ──────────────────────────────────────────────────
#
# A serial interval is the sum of three delays: the generation time from
# infector to infectee, plus the infectee's incubation period, minus the
# infector's. Building it needs `convolved` and `difference` from
# ConvolvedDistributions.jl, so the method is defined once that package (and
# CensoredDistributions.jl, which depends on it and supplies
# `double_interval_censored` for the censoring step around it) is loaded; see
# `ext/EpiBranchCensoredDistributionsExt.jl`. The stub and its docstring stay
# here so `serial_interval` is exported and documented either way.

"""
    serial_interval(model, infector, infectee)

The distribution of the serial interval (symptom onset in `infector` to onset
in `infectee`) implied by `model`'s generation time and incubation period:
the generation time from `infector` to `infectee`, plus `infectee`'s
incubation period, minus `infector`'s. Pass it to CensoredDistributions.jl's
`double_interval_censored` for the likelihood of an observed gap between two
linked cases' onsets, under daily reporting and the right truncation of an
ongoing outbreak.

`model` is a [`ModelSpec`](@ref) built with a single infectiousness window, a
fixed generation-time distribution (not a per-case function; there is no
general way to resolve a convolution for every possible case), and
`attributes = `[`clinical_presentation`](@ref)`(...)`. `infector` and
`infectee` are [`Individual`](@ref)s, read the same way for both:
`clinical_presentation`'s incubation period does not yet vary between cases,
so today they only name which pair the serial interval is for.

Available once CensoredDistributions.jl is loaded (which pulls in
ConvolvedDistributions.jl).

!!! note "Closed forms rarely survive the composition"
    ConvolvedDistributions.jl has a closed form for the sum or the
    difference of two gamma delays with the same rate, or two normal ones;
    every other pair, including two lognormal or two Weibull delays, falls
    back to its numerical quadrature. The serial interval composes *three*
    delays this way — a convolution of the generation time with the
    *difference* of the two incubation periods — and that closed form, where
    it applies at all, only ever covers the inner difference: the outer sum
    sees an already-built `Difference`, not one of the literal families it
    recognises, so it always falls back to quadrature too. Separately,
    CensoredDistributions.jl's own closed forms for `double_interval_censored`
    cover a gamma, lognormal or Weibull delay directly, which the distribution
    `serial_interval` returns never is (it is always a sum of at least two
    components), so that closed form never applies to it either. In practice
    `serial_interval` is evaluated by nested numerical quadrature however the
    components are built.

# Examples

```julia
using EpiBranch, CensoredDistributions, Distributions, StableRNGs

model = ModelSpec(
    BranchingProcess(NegBin(2.5, 0.16), Gamma(5.95, 1 / 0.77));
    attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5))
)
state = simulate(model; n_initial = 5, rng = StableRNG(1))
infector, infectee = state.individuals[1], state.individuals[2]

si = serial_interval(model, infector, infectee)
obs = double_interval_censored(si; interval = 1, upper = 21.0)
logpdf(obs, 6)
```
"""
function serial_interval end
