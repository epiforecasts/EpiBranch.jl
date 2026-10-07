# Glossary

The terms used across the EpiBranch documentation, with the page that uses each
most. Julia syntax is explained in [Julia for R users](julia-for-r-users.md).

## Transmission

**Secondary cases.** The people one case infects. The case that infected them
is their **infector**, and each of them is an **infectee**.

**Potential secondary cases.** The people a case would infect if nothing
intervened: the number drawn from the offspring distribution. Interventions, a
contact's reduced susceptibility or a finite population can prevent some of
these infections. The ones prevented stay in the simulation as contacts who
were not infected. See [Getting started](tutorials/getting-started.md).

**Offspring distribution.** The distribution of the number of potential
secondary cases per case. Without interventions its mean is R.

**R (reproduction number).** The mean number of secondary cases per case. With
no immunity and no control this is R₀. The R in a model is the mean of its
offspring distribution; the number realised under interventions is lower.

**k (dispersion).** How much the number of secondary cases varies between
cases. Smaller k means more superspreading: most cases infect nobody and a few
infect many. A Poisson offspring distribution is the limit as k becomes large.

**`NegBin(R, k)`.** The negative binomial offspring distribution with mean R
and dispersion k (variance R + R²/k). It returns the Distributions package's
`NegativeBinomial`, whose own parameters are different; build it with
`NegBin`. See [Analytical functions](tutorials/analytical.md).

**Multi-type model.** A model with several kinds of host (for example age
groups), each with its own numbers of secondary cases of each kind. See
[Multi-type models](tutorials/multi-type.md).

## Timing

Times are in whatever unit the delay distributions use; the documentation uses
days.

**Generation time.** The time from the infection of an infector to the
infection of an infectee.

**Serial interval.** The time from symptom onset in an infector to onset in an
infectee. EpiBranch models generation times; serial intervals follow from them
and the incubation period.

**Incubation period.** The time from infection to symptom onset, given with
[`clinical_presentation`](@ref). See [Getting started](tutorials/getting-started.md).

**Contact interval.** In network and household models, the time from the
start of an infector's infectious period (by default their infection) to a
contact with a particular person that would infect them if nothing intervened. See
[Covariates and time-varying transmission](tutorials/covariate-transmission.md).

**Index case.** A case an outbreak starts from. `simulate` starts from one
unless you set `n_initial`.

## Control

**Isolation.** Removing a known case from transmission, usually some time after
symptom onset. See [Isolation and contact tracing](tutorials/interventions.md).

**Quarantine.** Removing a traced contact, who is not (yet) a known case, from
transmission from the time they are traced. Quarantine reaches contacts before
they have symptoms, and those who were never infected.

**Isolation and quarantine duration.** How long isolation or quarantine lasts,
set with `duration` on [`Isolation`](@ref) and [`Quarantine`](@ref): a number
of days, a distribution or a function of the individual. When it ends the
person is released, and one who is still infectious can transmit again.
`duration = Inf` means they are never released. When a
duration is finite, the line list records the release date in
`date_isolation_release`.

**Interventions.** The control measures in a model: isolation, contact tracing,
ring and mass vaccination, post-exposure prophylaxis, and others. Each
potential infection goes ahead only if no measure prevents it first; survival
analysis calls this competing risks. See
[Isolation and contact tracing](tutorials/interventions.md) and
[Vaccination](tutorials/vaccination.md).

**Containment probability.** The proportion of simulated outbreaks in which
transmission stopped before reaching the case cap (`max_cases`), the
generation cap or the time cap. [`containment_probability`](@ref) calculates it
from simulations and includes every intervention in the model.
[`probability_contain`](@ref) gives a closed-form version for simpler control.

**Extinction probability.** The probability that transmission from one index
case dies out, calculated exactly from the offspring distribution by
[`extinction_probability`](@ref). It ignores the interventions and population
characteristics in a model and gives the probability without control. Compare
it with the containment probability to see what the interventions add. Its
complement is [`epidemic_probability`](@ref). See
[Analytical functions](tutorials/analytical.md).

## Models and outputs

**Model.** In these pages, the transmission process together with everything
added to it: population characteristics, interventions, clinical progression
and how cases are observed. In code a model is a transmission process (such as
[`BranchingProcess`](@ref)) on its own, or wrapped in a [`ModelSpec`](@ref)
that adds the other parts. See [Getting started](tutorials/getting-started.md).

**Scenario.** One set of parameters and interventions to compare with others,
for example the same pathogen with and without contact tracing.

**Population characteristics (attributes).** Characteristics each person is
given when they enter the simulation, such as age, incubation period or whether
they will develop symptoms. They are passed to a model as `attributes`, built
with functions such as [`clinical_presentation`](@ref) and
[`demographics`](@ref). See [Line lists and contacts](tutorials/linelist.md).

**Clinical progression.** The sequence of clinical events a case goes through,
such as onset, reporting, admission and recovery or death. See
[Clinical transitions](tutorials/transitions.md).

**Line list.** A table with one row per case and columns such as infection
date, onset date, isolation and tracing status, as from outbreak surveillance.
[`linelist`](@ref) builds one from a simulation; [`contacts`](@ref) builds the
matching table with one row per contact. See
[Line lists and contacts](tutorials/linelist.md).

**Transmission chain.** All the cases descending from one index case.
**Chain size** is the number of cases in it, including the index case;
**chain length** is the number of generations after the index case. See
[Chain statistics](tutorials/chains.md).

**Borel and gamma-Borel distributions.** The distribution of chain sizes when
the offspring distribution is Poisson (Borel), and when R also varies between
chains following a gamma distribution (gamma-Borel). See
[Chain statistics](tutorials/chains.md).

## Inference

**Confidence interval.** An interval from maximum-likelihood estimation, such
as a profile-likelihood interval. See [Inference](tutorials/inference.md).

**Credible interval.** An interval from the quantiles of a Bayesian posterior
distribution. See [Inference](tutorials/inference.md).
