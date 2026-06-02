# ─────────────────────────────────────────────────────────────────────
# Cross-package extension contract.
#
# Empty generic declarations for every function that more than one
# slot-in package extends. Declaring them here lets each slot-in
# depend only on `EpiBranchCore` (rather than on the package that
# happens to define the first method).
#
# Concrete methods live in:
#   - `EpiBranchProcess`: `simulate`, `simulate_batch`, `step!`,
#     `make_contact!`, `draw_offspring` (default and BranchingProcess
#     methods).
#   - `EpiBranchObservation`: `simulate(::Observed{...})`.
#   - `EpiBranchAnalytics`: `chain_size_distribution` for offspring specs.
# ─────────────────────────────────────────────────────────────────────

"""
    simulate(model::TransmissionModel; kwargs...) -> SimulationState

Run a single outbreak simulation under `model`. The default method for
any `TransmissionModel` lives in `EpiBranchProcess`; `Observed{...}`
wrappers extend it in `EpiBranchObservation` to apply an observation
model after the underlying simulation runs.
"""
function simulate end

"""
    simulate_batch(model::TransmissionModel, n::Int; kwargs...) -> Vector{SimulationState}

Run `n` independent outbreak simulations under `model`. Defined in
`EpiBranchProcess`.
"""
function simulate_batch end

"""
    step!(model::TransmissionModel, state::SimulationState) -> Vector{Individual}

One generation of the model: produce the new contacts arising from the
currently active individuals. Custom transmission models implement
this. `BranchingProcess` does so in `EpiBranchProcess`.
"""
function step! end

"""
    make_contact!(new_contacts, state, parent, infection_time; kwargs...) -> Individual

Construct a new contact of `parent` at `infection_time`, register it in
the simulation state, and append it to `new_contacts`. Called from a
model's `step!`. Defined in `EpiBranchProcess`.
"""
function make_contact! end

"""
    draw_offspring(rng, offspring_spec, individual, state) -> Int or Vector{Int}

Sample the number (or per-type vector) of offspring for `individual`
under the given `offspring_spec`. Defined in `EpiBranchProcess` for
distributions and function-form specs; extended in `EpiBranchAnalytics`
for `ClusterMixed`.
"""
function draw_offspring end

"""
    chain_size_distribution(spec) -> DiscreteUnivariateDistribution

The (closed-form) distribution of total cluster sizes for the offspring
specification or transmission model `spec`. Defined in `EpiBranchAnalytics`
for the standard offspring distributions (`Poisson`, `NegativeBinomial`,
`ClusterMixed`) and extended in `EpiBranchObservation` to thin via
`Observed{<:Any, <:PerCaseObservation}`.
"""
function chain_size_distribution end
