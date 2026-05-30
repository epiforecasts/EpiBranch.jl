"""
Concrete clinical-state transitions: `Reporting`, `Hospitalisation`,
`Death`, `Recovery`. Same hook shape as interventions; the engine
applies them after attributes and interventions have run.
"""
module Transitions

using Distributions
using Random
using ..EpiBranchBase

# Generics from Base that we add methods to.
import ..EpiBranchBase: initialise_individual!, resolve_individual!,
                        is_terminal, terminal_event, required_fields

export Reporting, Hospitalisation, Death, Recovery

# Transition-specific helpers (probability/delay/anchor resolution,
# terminal arbitration). These live with the transitions because they
# describe the transition resolution semantics.
include("helpers.jl")
include("reporting.jl")
include("hospitalisation.jl")
include("outcome.jl")

end
