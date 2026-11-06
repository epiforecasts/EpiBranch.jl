`CapacityConstrained`'s `budget_per_period` accepts a calendar schedule — a
`Steps`, a `Seasonal`, or a callable `t -> budget(t)` — read through the same
interface a `PairKernel`'s `calendar` is, rather than only a fixed number, so
a campaign whose daily doses are known in advance no longer needs one
`Scheduled` window per period. A finite `period` is required with a schedule,
since there is otherwise no period start to read it at.
