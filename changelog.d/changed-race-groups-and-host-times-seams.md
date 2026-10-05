`_kernel_projection` and `_live_kernel`, the `PairKernel`-by-type-parameter
smell `watched_records` already replaced in practice, are deleted: nothing
outside `src/pair_kernels.jl` and its own test reached either.

`EpiBranch.race_groups(model, kernel)` is a new seam a household outbreak's
race partition goes through: one race per household, or every household on a
single shared clock, rather than that choice being an inline `Bool` in
`household_simulate.jl`. A kernel type of your own can override the method for
a given model to pick a different partition outright.

`EpiBranch.host_times(data::InfectionLayer)` is now public and documented, the
shape `followup_end` already had: the default reads a `host_times` field and is
empty otherwise, and a subtype holding its per-host times under another name
overrides the method instead of the likelihood losing them in silence.
