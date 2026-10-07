# EpiBranch.jl

**EpiBranch.jl** is a [Julia](https://julialang.org/) package for asking how an
outbreak grows from a few introduced cases, and what isolation, contact tracing
or vaccination would do to it. It simulates outbreaks as branching processes,
calculates exact results where they exist (extinction probability, the
distribution of chain sizes, the share of transmission caused by the most
infectious cases), and fits these models to outbreak data.

With it you can:

- simulate outbreaks under isolation, contact tracing and quarantine, ring or
  mass vaccination and post-exposure prophylaxis, and estimate how often these
  contain an outbreak;
- give cases an incubation period, an age, or a clinical course (symptom onset,
  reporting, hospital admission, recovery or death);
- produce line lists and contact tables shaped like those from a real outbreak;
- estimate R and the dispersion k from offspring counts, chain sizes or chain
  lengths, by maximum likelihood or in a Bayesian model;
- model transmission between types of host (for example age groups), over
  [contact networks](tutorials/networks.md) or within
  [households](tutorials/households.md), the last two through companion
  packages.

If you are new to Julia, [Julia for R users](julia-for-r-users.md) explains the
syntax used in these pages, and the [Glossary](glossary.md) defines the terms.

## Where it comes from

EpiBranch brings together what five R packages do:

- [ringbp](https://github.com/epiforecasts/ringbp): whether isolation and
  contact tracing contain an outbreak;
- [simulist](https://github.com/epiverse-trace/simulist): simulated line lists
  and contact tracing data;
- [epichains](https://github.com/epiverse-trace/epichains): transmission chain
  sizes and lengths, and their likelihoods;
- [superspreading](https://github.com/epiverse-trace/superspreading): exact
  results for overdispersed transmission;
- [pepbp](https://github.com/sophiemeakin/pepbp): post-exposure prophylaxis.

To these it adds multi-type transmission (for example between age groups, or
between hosts who transmit more and hosts who transmit less), and one set of
functions that works on the same model for simulation, exact calculation and
likelihood.

## How a case transmits

Each case draws a number of potential secondary cases from the offspring
distribution: the people it would infect if nothing intervened. Each potential
infection then gets a time from the generation time distribution. Isolation,
quarantine, vaccination, a contact's reduced susceptibility or a finite
population can each prevent it, and whichever happens first decides whether
that person is infected (survival analysis calls this competing risks). With
none of these in the model, every potential secondary case is infected, so the
mean of the offspring distribution is R.

!!! note "Generation times and the number of secondary cases"
    By default each generation time is drawn independently of how many
    secondary cases the case has. A generation time that depends on the case,
    for example on its own incubation period as in
    [`incubation_linked_generation_time`](@ref), is also possible.

[Design](design.md) describes how the package is organised, and
[Extending EpiBranch](tutorials/extending.md) shows how to add your own
intervention or transmission model.

## Quick start

This estimates how often isolation and contact tracing contain an outbreak of a
pathogen with R = 2.5 and strong superspreading (k = 0.16). Times are in days.

```@example quickstart
using EpiBranch
using Distributions
using StableRNGs

# Symptomatic cases isolate on average 2 days after onset, for 7 days
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), isolation_duration = 7.0)
# Each contact of an isolated case is traced with probability 0.5, on average
# 1.5 days after the case isolates, and quarantined
ct = ContactTracing(probability = 0.5, isolation_to_trace_delay = Exponential(1.5))

# R = 2.5, dispersion k = 0.16. Generation time LogNormal with mean 1.6 and
# sd 0.5 on the log scale (a mean of about 5.6 days); incubation period
# LogNormal(1.5, 0.5) (a mean of about 5.1 days).
model = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5));
    interventions = [iso, ct],
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
)

rng = StableRNG(42)  # a fixed random seed, like set.seed() in R
results = simulate(model, 500; max_cases = 5000, rng = rng)

containment_probability(results)
```

This simulates 500 outbreaks, each starting from one index case and stopped once
it reaches 5000 cases. The containment probability is the proportion in which
transmission stopped before that cap. The
[Getting started](tutorials/getting-started.md) tutorial goes through each step.

## Installation

The [installation guide](installation.md) shows how to install EpiBranch and the
household and network packages.
