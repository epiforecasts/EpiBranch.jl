"""
Concrete intervention implementations: `Isolation`, `ContactTracing`,
`RingVaccination`, `MassVaccination`, and the `Scheduled` wrapper.
Each plugs into the engine via the four-hook protocol declared in
`EpiBranchBase`.

This submodule also owns the per-intervention state accessors
(`is_isolated`, `is_traced`, `is_vaccinated`, …) and the seam-trait
extension points (`is_eligible`, `is_eligible_for_isolation`,
`traces`, …). Reach for them either through `EpiBranch.is_isolated`
(re-exported at the top level) or via the qualified
`EpiBranch.Interventions.X` path.
"""
module Interventions

using Distributions
using Random
using ..EpiBranchBase

# Internal helpers from Base
using ..EpiBranchBase: _sample_value

# Generics from Base that we add methods to.
import ..EpiBranchBase: initialise_individual!, resolve_individual!,
                        apply_post_transmission!, competing_risk, is_active,
                        intervention_time, reset!, required_fields

# Intervention seam-trait generics — declared here because they are
# concrete-intervention extension points, not part of the universal
# protocol. Other code adds methods to these for their own trait types.
function is_eligible end
function is_eligible_for_isolation end
function traces end
function draw_trace_delay end
function apply_trace! end
function required_for_eligibility end
function required_for_ct_eligibility end

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

# Accessors for intervention-owned state keys
export is_isolated, isolation_time, set_isolated!, is_test_positive
export is_traced, is_quarantined
export is_vaccinated

# Seam-trait extension points
export is_eligible, is_eligible_for_isolation
export traces, draw_trace_delay, apply_trace!
export required_for_eligibility, required_for_ct_eligibility

include("isolation.jl")
include("contact_tracing.jl")
include("vaccination.jl")
include("scheduled.jl")

end
