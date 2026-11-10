`OnRoutes` now reports the wrapped intervention's `removal_block_probability` and
`removal_gap_host_times`, as `Scheduled`, `CapacityConstrained` and `HeldCapacity`
already did. Every `InterventionWrapper` forwards both by default.
