"""
Concrete intervention implementations: `Isolation`, `ContactTracing`,
`RingVaccination`, `MassVaccination`, and the `Scheduled` wrapper.
Each plugs into the engine via the four-hook protocol declared in
`EpiBranchBase`.
"""
module EpiBranchInterventions

using Distributions
using Random
using ..EpiBranchBase

# Internal helpers from Base
using ..EpiBranchBase: _sample_value

# Generics from Base that we add methods to.
import ..EpiBranchBase: initialise_individual!, resolve_individual!,
                        apply_post_transmission!, competing_risk, is_active,
                        intervention_time, reset!, required_fields,
                        is_eligible, is_eligible_for_isolation,
                        traces, draw_trace_delay, apply_trace!,
                        required_for_eligibility, required_for_ct_eligibility

# Intervention types
export Isolation, ContactTracing
export IsolationEligibility, SymptomaticOnly, AllCases
export TraceEligibility, AlwaysEligible, SymptomaticParent
export TraceRate, ConstantRate
export TraceDelay, ConstantDelay
export TraceAction, Quarantine, FlagOnly
export AbstractVaccination, RingVaccination, MassVaccination
export AbstractEffectMode, LeakyMode, AllOrNothingMode
export Scheduled

include("isolation.jl")
include("contact_tracing.jl")
include("vaccination.jl")
include("scheduled.jl")

end
