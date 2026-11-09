`loglikelihood(data, model)` for households takes a `layout` keyword, so a
fitting loop can build the pair layout once with `compile_household_pairs` and
reuse it instead of rebuilding it at every evaluation.
