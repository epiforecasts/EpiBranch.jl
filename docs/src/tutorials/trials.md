# Vaccine trials

The companion `EpiTrial` package simulates vaccine trials run during an
outbreak and evaluates how their estimates behave: power, bias and coverage
of the confidence interval over many replicate trials. A trial is an
EpiBranch model with two extra layers composed onto it:

- **randomisation**, an attributes element that assigns each individual to the
  `:vaccine` or `:control` arm;
- **the trial vaccine**, a vaccination that doses the vaccine arm at
  enrolment.

A [`Trial`](@ref EpiTrial.Trial) then adds the follow-up period and the
endpoint, and turns each simulated outbreak into one row per participant.

## A cohort under a known force of infection

The simplest trial enrols a cohort who face a force of infection from outside
the trial and do not infect each other. People living alone in a
`HouseholdProcess` with a constant community hazard give exactly this: the
cumulative hazard faced by an unvaccinated participant over follow-up is the
hazard times the follow-up time.

```@example trials
using EpiBranch
using EpiHouseholds
using EpiTrial
using Distributions
using StableRNGs

λ, T = 0.01, 100.0      # community hazard per day, follow-up in days
spec = ModelSpec(
    HouseholdProcess(fill(1, 2000), Exponential(1.0); external_hazard = λ, obs_end = T);
    progression = [Transition(:recovered; from = :infection, delay = 5.0, terminal = true)],
    attributes = IndividualRandomisation(),
    interventions = [TrialVaccination(efficacy = 0.6)])
trial = Trial(spec; follow_up = T)
```

`simulate` on a trial returns the trial data: each participant's arm, the
time they left follow-up, and whether they reached the endpoint (here,
infection).

```@example trials
data = simulate(trial; rng = StableRNG(1))
first(data, 5)
```

## Estimating vaccine efficacy

[`estimate`](@ref EpiTrial.estimate) applies an estimator to the trial data.
[`RiskRatio`](@ref EpiTrial.RiskRatio) compares attack rates over the whole
follow-up; [`CoxHazardRatio`](@ref EpiTrial.CoxHazardRatio) compares hazards.

```@example trials
(risk_ratio = estimate(RiskRatio(), data), cox = estimate(CoxHazardRatio(), data))
```

The two differ, and the difference is predictable. The vaccine here is leaky
(the default mode): it multiplies each vaccinee's hazard by 1 − 0.6. The hazard
ratio is then constant and the Cox estimate targets 0.6, while the ratio of
attack rates drifts towards 1 as the cumulative hazard grows, because the
unvaccinated deplete faster.
[`expected_ve`](@ref EpiTrial.expected_ve) gives what each estimator converges
to in a large trial:

```@example trials
Λ = λ * T
(risk_ratio = expected_ve(RiskRatio(), LeakyMode(), 0.6, Λ),
    cox = expected_ve(CoxHazardRatio(), LeakyMode(), 0.6, Λ))
```

## Operating characteristics

[`operating_characteristics`](@ref EpiTrial.operating_characteristics)
simulates replicate trials, applies each estimator to the same trials, and
summarises them. With `target` given it reports bias and coverage against it;
passing a function lets each estimator be judged against its own expected
value.

```@example trials
operating_characteristics(trial, [RiskRatio(), CoxHazardRatio()];
    n_sim = 200, rng = StableRNG(2),
    target = est -> expected_ve(est, LeakyMode(), 0.6, Λ))
```

`power` is the proportion of trials whose lower confidence bound exceeds
`null_ve` (0 by default). With a vaccine of no effect it is the type I error.

## References

- Smith PG, Rodrigues LC, Fine PEM (1984). Assessment of the protective efficacy of vaccines against common diseases using case-control and cohort studies. *International Journal of Epidemiology* 13(1):87–93. [doi:10.1093/ije/13.1.87](https://doi.org/10.1093/ije/13.1.87)
- Halloran ME, Haber M, Longini IM (1992). Interpretation and estimation of vaccine efficacy under heterogeneity. *American Journal of Epidemiology* 136(3):328–343. [doi:10.1093/oxfordjournals.aje.a116498](https://doi.org/10.1093/oxfordjournals.aje.a116498)
- Halloran ME, Longini IM, Struchiner CJ (2010). *Design and Analysis of Vaccine Studies*. Springer. [doi:10.1007/978-0-387-68636-3](https://doi.org/10.1007/978-0-387-68636-3)
