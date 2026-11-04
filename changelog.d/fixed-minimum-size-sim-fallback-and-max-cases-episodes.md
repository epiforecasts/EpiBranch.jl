The simulation fallback for chain-size likelihoods and for `rand` on a
`chain_size_distribution` applied a `MinimumSize` threshold to each seed
chain on its own, dropping a multi-seed cluster whose seed chains were
individually below `min_size` but summed above it. It now applies the
threshold to the cluster's summed size, as the closed-form
`TruncatedChainSize` already does.

`cumulative_cases`, and so `MaxCases`, counted a settled episode rather than
a person: on a model that reinfects, each reinfection added to the count
again, so `max_cases = n` could stop a run early, before `n` people had
actually been infected. Both now count people, matching what
`cumulative_cases` reports once the run ends.
