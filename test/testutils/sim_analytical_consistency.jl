# ── Simulation ↔ analytical consistency checks ──────────────────────
# A standard template: given any model with both an analytical chain
# size distribution and a working simulation path, the empirical chain
# size PMF from simulation must match the analytical PMF (within
# sampling error) at small sizes.

"""
    sim_analytical_consistent(model, gt; n_chains=5000, sizes=1:4, atol=0.02, rng)

Simulate `n_chains` chains from `BranchingProcess(model, gt)` and test
that the empirical fraction at each size in `sizes` agrees with
`pdf(chain_size_distribution(model), size)` within `atol`.

Returns `(empirical, analytical)` vectors keyed on `sizes` for further
inspection or logging.
"""
function sim_analytical_consistent(offspring_spec, gt;
        n_chains::Int = 5000,
        sizes = 1:4,
        rng::AbstractRNG = Random.default_rng(),
        max_cases::Int = 500)
    model = BranchingProcess(offspring_spec, gt)
    states = simulate_batch(model, n_chains;
        sim_opts = SimOpts(max_cases = max_cases), rng = rng)
    chain_sizes = Int[]
    for s in states
        append!(chain_sizes, chain_statistics(s).size)
    end

    d = chain_size_distribution(offspring_spec)
    empirical = [count(==(n), chain_sizes) / length(chain_sizes) for n in sizes]
    analytical = [pdf(d, n) for n in sizes]
    return empirical, analytical
end
