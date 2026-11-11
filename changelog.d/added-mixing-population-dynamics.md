`MixingMeanFieldProcess` and `MixingLinearNoiseProcess` approximate a
`MixingProcess` with several mixing groups, one SIR compartment per group
rather than the single one `MeanFieldProcess`/`LinearNoiseProcess` keep for a
homogeneous population. Both are built from the `ModelSpec` they
approximate: group sizes come from its `attributes`, and the
between-group rate and any calendar scaling come from its `MixingProcess`.
The mixing structure is specified once. `HybridProcess` now
switches from an exact `MixingProcess` run to either of them, passing each
group's compartment counts across the switch, in addition to the
homogeneous pairing it already supported.
