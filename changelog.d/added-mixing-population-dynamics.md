`MixingMeanFieldProcess` and `MixingLinearNoiseProcess` approximate a
`MixingProcess` with several mixing groups, one SIR compartment per group
rather than the single one `MeanFieldProcess`/`LinearNoiseProcess` keep for a
homogeneous population. Built straight from the `ModelSpec` they
approximate, they read the group sizes off its `attributes`, and the
between-group rate and any calendar scaling it off its `MixingProcess`,
so the mixing structure is never written down twice. `HybridProcess` now
switches from an exact `MixingProcess` run to either of them, passing each
group's compartment counts across the switch, in addition to the
homogeneous pairing it already supported.

Vaccination is not yet read from a model's interventions as a
population-level input to either approximation.
