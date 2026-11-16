The `budget_per_period` of `CapacityConstrained` accepts a fixed number or
a calendar schedule (`Steps`, `Seasonal`, or a callable `t -> budget(t)`),
using the same interface as the `calendar` of a `PairKernel`. A campaign
with daily doses known in advance no longer needs one
`Scheduled` window per period. A schedule requires a finite `period`
to determine the period start at which to read the budget.
