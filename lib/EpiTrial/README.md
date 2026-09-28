# EpiTrial

Simulation-based evaluation of vaccine trials run during an outbreak, built on
[EpiBranch.jl](https://github.com/epiforecasts/EpiBranch.jl).

A trial is an EpiBranch model with randomisation composed on as an attribute
(`IndividualRandomisation`) and the trial vaccine as an intervention
(`TrialVaccination`). `Trial` adds the follow-up period and the endpoint;
`simulate` on a `Trial` returns one row per participant. Estimators of vaccine
efficacy (`RiskRatio`, `CoxHazardRatio`) apply to that table, and
`operating_characteristics` summarises replicate trials as power, bias and
coverage. `expected_ve` gives what each estimator converges to under a leaky or
all-or-nothing vaccine, for checking simulations against.

```julia
using EpiBranch, EpiHouseholds, EpiTrial, Distributions

spec = ModelSpec(
    HouseholdProcess(fill(1, 2000), Exponential(1.0); external_hazard = 0.01, obs_end = 100.0);
    progression = [Transition(:recovered; from = :infection, delay = 5.0, terminal = true)],
    attributes = IndividualRandomisation(),
    interventions = [TrialVaccination(efficacy = 0.6)])
trial = Trial(spec; follow_up = 100.0)

operating_characteristics(trial, [RiskRatio(), CoxHazardRatio()]; n_sim = 500)
```

See the "Vaccine trials" tutorial in the EpiBranch documentation.
