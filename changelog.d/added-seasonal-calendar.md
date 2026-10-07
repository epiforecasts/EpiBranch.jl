`Seasonal(; amplitude, peak_day, period = 365.0)` is a built-in smooth
calendar for a `PairKernel`, rising and falling once a period around
`peak_day`. A plain callable `t -> multiplier` is also accepted as a
calendar, read as a smooth schedule, for forcing of any other shape.
