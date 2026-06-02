"""
EpiBranchProcess — the canonical branching-process simulator.

Owns the engine (`simulate`, `simulate_batch`, `step!`, `make_contact!`,
`draw_offspring`), the `BranchingProcess` transmission model, stopping
rules, and the attribute builders (`clinical_presentation`,
`demographics`, `transmission_traits`, `compose`).

Extends the generics declared in `EpiBranchCore`. Slot-in packages
that need to run simulations depend on this package; packages that
only need the protocol layer (e.g. for new intervention or
observation types) depend only on `EpiBranchCore`.
"""
module EpiBranchProcess

using Distributions
using DocStringExtensions
using Random
using EpiBranchCore
using EpiBranchCore: _sample_value, _finalise_terminal!, _resolve_probability,
                     _resolve_delay, _resolve_anchor, _from_required

# ── Includes ────────────────────────────────────────────────────────
include("options.jl")
include("branching_process.jl")
include("simulation.jl")

# ── Exports ─────────────────────────────────────────────────────────

# The canonical model
export BranchingProcess

# Stopping rules and options
export SimOpts, AbstractStoppingRule
export Extinction, MaxCases, MaxGenerations, MaxTime, should_stop

# Attribute builders
export clinical_presentation, demographics, transmission_traits, compose

end # module
