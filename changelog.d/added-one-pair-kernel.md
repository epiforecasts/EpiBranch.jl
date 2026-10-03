`PairKernel` shares fixed covariates, the infector's infection time, sampled
attributes and dated intervention histories between structured simulation and
likelihoods, and multiplies the contact-interval hazard by a schedule on the
calendar. `Steps` gives a piecewise-constant schedule, and any type with a
`calendar_multiplier` method can be one too, including a smooth schedule such
as a seasonal contact rate. `record_kernel` extracts typed host records for
inference. An infection layer can hold per-host times such as onsets
(`host_times`), which a live `PairKernel` reads in the likelihood, so one
kernel timed from symptom onset serves both simulation and inference.
