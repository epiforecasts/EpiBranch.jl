The simulation fallback for chain-size likelihoods and for `rand` on a
`chain_size_distribution` applied a `MinimumSize` threshold to each seed
chain on its own, dropping a multi-seed cluster with seed chains
individually below `min_size` but summed above it. It now applies the
threshold to the summed cluster size, as the closed-form
`TruncatedChainSize` already does.

`cumulative_cases`, and with it `MaxCases`, counted a settled episode rather
than a person: on a model that reinfects, each reinfection added to the
count again. As a result, `max_cases = n` could stop a run before `n` people
had been infected. Both now count people, matching what `cumulative_cases`
reports once the run ends.
