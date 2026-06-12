# ── Simulation ↔ analytical consistency checks ──────────────────────
# Any TransmissionModel with both a defined `chain_size_distribution`
# and the simulation protocol below can be cross-checked: the empirical
# chain-size PMF from simulation should match the analytical PMF
# within sampling error at small sizes.
#
# Protocol a new model type must define to participate:
#
#   generative_model(m)  — strip any observation wrappers; returns
#                          a model that `simulate(model, n)` can run.
#   observe_chain_sizes(m, true_sizes, rng) — transform simulated
#                          chain sizes into the observed form the
#                          likelihood expects. Default: identity.
#
# `chain_size_distribution(m)` must return the corresponding
# distribution over observed chain sizes.

# The generative model is what we simulate from. `chain_statistics`
# counts all infected cases (the observation only flags `:reported`), so
# simulating the model directly yields the true chain sizes.
generative_model(m::TransmissionModel) = m

"""
Transform simulated true chain sizes into the observed sizes matching
the analytical distribution, by applying the model's observation. Default
(no observation) is identity.
"""
function observe_chain_sizes(m::TransmissionModel, true_sizes, rng::AbstractRNG)
    _observe_sizes(EpiBranch.observation(m), true_sizes, rng)
end

_observe_sizes(::NoObservation, true_sizes, ::AbstractRNG) = true_sizes

function _observe_sizes(o::PerCaseObservation, true_sizes, rng::AbstractRNG)
    p = EpiBranch.scalar_detection_prob(o)
    # Keep zero-detection outcomes in the output so the empirical marginal
    # matches the unconditional PMF returned by chain_size_distribution.
    return [rand(rng, Binomial(n, p)) for n in true_sizes]
end

"""
    sim_analytical_consistent(model; n_chains=5000, sizes=1:4, rng, max_cases=500)

Simulate from `generative_model(model)`, transform the simulated chain
sizes via `observe_chain_sizes(model, ...)`, and compare the empirical
PMF at `sizes` against `pdf(chain_size_distribution(model), size)`.

Returns `(empirical, analytical)` — vectors indexed by `sizes`. Use
`isapprox` / `@test` against a suitable `atol` (typically ≈0.02 for
5000 chains).
"""
function sim_analytical_consistent(model::TransmissionModel;
        n_chains::Int = 5000,
        sizes = 1:4,
        rng::AbstractRNG = Random.default_rng(),
        max_cases::Int = 500)
    gen = generative_model(model)
    states = simulate(gen, n_chains;
        max_cases = max_cases, rng = rng)
    true_sizes = Int[]
    for s in states
        append!(true_sizes, chain_statistics(s).size)
    end
    obs_sizes = observe_chain_sizes(model, true_sizes, rng)
    isempty(obs_sizes) && return (Float64[0.0 for _ in sizes],
        [pdf(chain_size_distribution(model), n) for n in sizes])

    d = chain_size_distribution(model)
    empirical = [count(==(n), obs_sizes) / length(obs_sizes) for n in sizes]
    analytical = [pdf(d, n) for n in sizes]
    return empirical, analytical
end
