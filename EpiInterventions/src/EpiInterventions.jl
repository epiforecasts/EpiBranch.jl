"""
EpiInterventions — the standard library of non-pharmaceutical and
pharmaceutical interventions for `EpiBranch`.

Owns `Isolation`, `ContactTracing`, the vaccination types
(`RingVaccination`, `MassVaccination`), `Scheduled` for time-based
gating, plus their seam traits (`IsolationEligibility`,
`TraceEligibility`, `TraceRate`, `TraceDelay`, `TraceAction`,
`AbstractEffectMode`, `AbstractVaccination`, etc.).

Concrete intervention subtypes here extend the protocol generics
declared in `EpiBranchCore` (`initialise_individual!`,
`resolve_individual!`, `apply_post_transmission!`, `competing_risk`,
`is_active`, `intervention_time`, `reset!`, `required_fields`).
"""
module EpiInterventions

using Distributions
using DocStringExtensions
using Random
using EpiBranchCore
using EpiBranchCore: _sample_value

include("isolation.jl")
include("contact_tracing.jl")
include("vaccination.jl")
include("scheduled.jl")

# ── Exports ─────────────────────────────────────────────────────────

# Isolation
export Isolation
export IsolationEligibility, SymptomaticOnly, AllCases
export is_eligible_for_isolation

# Contact tracing
export ContactTracing
export TraceEligibility, AlwaysEligible, SymptomaticParent
export TraceRate, ConstantRate
export TraceDelay, ConstantDelay
export TraceAction, Quarantine, FlagOnly

# Vaccination
export AbstractVaccination, RingVaccination, MassVaccination
export AbstractEffectMode, LeakyMode, AllOrNothingMode

# Time-based scheduling
export Scheduled

end # module
