`_kernel_projection` and `_live_kernel`, the `PairKernel`-by-type-parameter
smell `watched_records` already replaced in practice, are deleted: nothing
outside `src/pair_kernels.jl` and its own test reached either.

`EpiBranch.host_times(data::InfectionLayer)` is now public and documented, the
shape `followup_end` already had: the default reads a `host_times` field and is
empty otherwise, and a subtype holding its per-host times under another name
overrides the method instead of the likelihood losing them in silence.
