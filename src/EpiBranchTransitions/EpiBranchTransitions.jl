"""
Concrete clinical-state transitions: `Reporting`, `Hospitalisation`,
`Death`, `Recovery`. Same hook shape as interventions; the engine
applies them after attributes and interventions have run.
"""
module EpiBranchTransitions

using Distributions
using Random
using ..EpiBranchBase

# Internal helpers from Base
using ..EpiBranchBase: _resolve_probability, _resolve_delay, _resolve_anchor,
                       _from_required

# Generics from Base that we add methods to.
import ..EpiBranchBase: initialise_individual!, resolve_individual!,
                        is_terminal, terminal_event, required_fields

export Reporting, Hospitalisation, Death, Recovery

include("reporting.jl")
include("hospitalisation.jl")
include("outcome.jl")

end
