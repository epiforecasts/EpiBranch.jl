`loglikelihood(::IncidenceCounts, model)` now reads a model's `Reporting`
step, for a series of reported cases by report date. A report timed from
onset adds the `clinical_presentation` incubation period to the reporting
delay and counts only cases that are not asymptomatic. The step has to be
the model's only one, with a fixed probability and delay.
