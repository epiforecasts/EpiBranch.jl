The simulation-based `loglikelihood(::ChainSizes, model)` — the route taken
when the model has interventions or an offspring law with no closed form —
now honours `data.seeds`: clusters are simulated grouped by seed count, one
batch per distinct value, and each simulation's chains are summed to a
single cluster size, rather than every simulated chain being compared
against the data as if it were its own single-seed cluster. `prob_concluded`
is also now accepted on this route, mixing each cluster's point mass with
the simulated clusters' empirical right tail instead of being refused.
