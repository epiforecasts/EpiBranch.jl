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
#     `make_contact!`, `draw_offspring`, `susceptible_fraction`
#     (default and BranchingProcess methods).
#   - `EpiBranchObservation`: `simulate(::Observed{...})`.
#   - `EpiBranchAnalytics`: `chain_size_distribution` for offspring
#     specs.
#
# Note: likelihoods and offspring-distribution fitting are NOT
# declared here. EpiBranchAnalytics extends `Distributions.loglikelihood`
# and `Distributions.fit` (originally `StatsAPI.loglikelihood` /
# `StatsAPI.fit`) directly — those are the cross-package seams for
# analytics. A downstream package adding a new data type or model
# adds methods to the Distributions.jl generics, not to anything
# declared in this file.
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

"""
    susceptible_fraction(state::SimulationState, extra_infected::Int = 0) -> Float64

Fraction of the population still susceptible at this point in the
simulation. Dispatched on the `population_size` type carried by
`SimulationState{<:Any, P}`, so downstream packages can introduce
structured populations (households, contact networks, age-stratified
pools) by defining a new population-size type and a method here.

`extra_infected` accounts for contacts already infected within the
current step but not yet registered in `state.cumulative_cases`.

Default methods for `NoPopulation` (infinite, always 1.0) and `Int`
(global pool of that size) live in `EpiBranchProcess`.
"""
function susceptible_fraction end
