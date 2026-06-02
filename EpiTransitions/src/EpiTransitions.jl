"""
EpiTransitions — clinical-state transitions for `EpiBranch`.

Owns the built-in [`Reporting`], [`Hospitalisation`], [`Death`], and
[`Recovery`] transitions. Concrete transition types here extend the
protocol generics declared in `EpiBranchCore`
(`initialise_individual!`, `resolve_individual!`, `is_terminal`,
`terminal_event`, `required_fields`).
"""
module EpiTransitions

using Distributions
using DocStringExtensions
using Random
using EpiBranchCore
using EpiBranchCore: _resolve_probability, _resolve_delay, _resolve_anchor,
                     _from_required

include("reporting.jl")
include("hospitalisation.jl")
include("outcome.jl")

# ── Exports ─────────────────────────────────────────────────────────

export Reporting, Hospitalisation, Death, Recovery

end # module
