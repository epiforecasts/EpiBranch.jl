# Design

This page explains, in epidemiological terms, how EpiBranch represents an
outbreak: how each case's potential infections, their timing and the control
measures that stop them are generated, and why simulating an outbreak and
fitting a model to data use the same model. How to write new pieces in Julia is
in [Extending EpiBranch](@ref); the principles and rules for changing the
package itself are in [Notes for contributors](contributing.md).

## Core idea

A contact causes infection unless something stops it first, and the earliest
blocker wins.

Each case first gets a number of potential secondary cases from the offspring
distribution, with mean R and dispersion k, and with no reference to time or
control measures. Each potential infection is then given a time, from the
generation-time distribution. Finally, anything that could prevent it is
considered: the infector being isolated before that time, the contact having
been vaccinated, the contact being less susceptible or the infector less
infectious. The infection happens only if none of these blocks it.

Because these steps are separate, the offspring distribution can be analysed
on its own, and extinction probabilities and chain-size distributions can be
written down exactly where the distribution allows, with no simulation. Control
measures become removals before a given time, as in survival analysis: treating
the blockers as competing risks on the hazard of transmission is the same
mechanism as Kenah's pairwise survival analysis and dynamic survival analysis.

## Three steps for every potential transmission

Every potential transmission goes through the same three steps. Only the first
depends on the transmission model; the other two work the same way for every
model.

As an illustration, an index case gets three potential secondary cases, at days
3, 6 and 9 after infection. It is isolated on day 5. The first infection goes
ahead, the other two are blocked by isolation, and the case has one secondary
case.

### 1. Who could be infected (depends on the model)

For a branching process this is a draw from the offspring distribution: a
single count, or a count per type for a multi-type model. The mean (R) and
dispersion (k) come from that distribution, as in `NegBin(2.5, 0.16)` for
R = 2.5 and k = 0.16. If a contact matrix is supplied, it sets the mixing
between types.

This is the only step a transmission model defines, and it does so in one of
two ways. A branching process and its variants (**offspring-driven** models)
draw a number of potential secondary cases for each case. Network, household
and metapopulation models (**structure-driven** models) cannot, because a
susceptible can be exposed by several infectious neighbours at once and
infections use up a fixed population. They list instead the people each
infectious person is in contact with, and a person exposed by several cases in
the same generation is considered once, with all of those exposures. The
companion `EpiNetwork.jl` package's network model is an example. Either way the
model only says who is in contact with whom.

### 2. When (the same for every model)

Each potential infection is given a transmission time from the infector's
infectiousness profile, the generation-time distribution. That distribution can
be the same for everyone or built for each case, so the timing can depend on
anything known about the case: its incubation period, or any other value
recorded about it. This is the *potential* time of transmission, from the
hazard h(t) in survival-analysis terms (see [Connection to survival
analysis](@ref)).

### 3. Whether anything stops it (the same for every model)

Each potential infection is decided on its own, infected or not, by checking
everything that could block it. A contact is infected only if nothing blocks
it: the earliest removal before the transmission time wins. Possible blockers
are:

- the infector being less infectious (a value below 1);
- the contact being less susceptible (a value below 1);
- the infector having been isolated, or otherwise removed, before the
  transmission time;
- any intervention, such as vaccination of the contact;
- a probability of transmission that belongs to the pair rather than to either
  person, such as the transmission probability of a network contact.

Susceptibility and infectiousness are checked in the same way as interventions,
with no special treatment, and so is the end of an infectious period.

Contacts who were exposed but not infected are kept in the output, because
contact tracing and vaccination reach them too. This is also how the number of
contacts traced, vaccines given or tests used is counted.

### Why this separation matters

Because the offspring distribution does not depend on time or control
measures, it can be analysed with standard tools: extinction probability from
the dominant eigenvalue, chain-size distributions, closed-form likelihoods.
These can also be used with gradient-based fitting. Interventions act on the
timing and on whether a transmission is blocked, never on the offspring
distribution: isolation removes the later part of the infectious period,
contact tracing moves that removal earlier, and vaccination lowers
susceptibility. The cumulative generation-time distribution evaluated at an
intervention time *is* the survival function. The same quantities therefore appear in
simulation and in Kenah's pairwise likelihood, and the effectiveness of an
intervention can be estimated from observed generation times using the
quantities used to simulate.

## The five parts of a model

A model is the whole description of how the data arise. It has five parts:

- **transmission**: how infection spreads between people, and nothing else;
- **disease**: the natural history within each case, the timed states it moves
  through (latent, infectious, onset, severe, recovered or died), with
  treatment as a step it may pass through;
- **population characteristics**: who the people are (age, susceptibility,
  infectiousness), which can affect transmission, the course of disease, and
  whom interventions reach;
- **interventions**: what is done about it (isolation, tracing, vaccination);
- **observation**: how cases are seen (under-reporting, reporting delays).

These match how epidemiologists describe an outbreak: a pathogen spreads,
infection causes disease, in a population of people, under a response, watched
through surveillance. Keeping the five separate, instead of folding the disease
or the policy into the transmission model, means each can be replaced on its
own, and the same disease, interventions or observation can be used with any
transmission model.

The parts depend on each other. Population characteristics affect
transmission, disease and who interventions reach; the infectious period is
defined by the disease states; and a transmission rate given as a reproduction
number depends on the mean infectious period. These links are worked out when
a model is simulated or evaluated, and each part is specified only once. A network or
household model, for example, takes the start of infectiousness, and its rate
when given as a reproduction number, from the disease part of the model.

### Who each person is

Each simulated person has their place in the transmission tree, their
infection time, their susceptibility and their infectiousness, plus a
free-form set of further values (age, isolation time, vaccination status, and
so on) that population characteristics, interventions and clinical transitions
read and write. The values the package uses are listed in [Individual state and
reserved keys](@ref).

### How interventions act

An intervention acts in one of three ways:

- it lowers a person's **susceptibility**, the probability of infection given
  exposure (vaccination, prior immunity);
- it lowers a case's **infectiousness**, its onward transmission (treatment);
- it **removes** a case from transmission from a given time (isolation,
  quarantine), cutting short the infectious period.

A contact is infected only if it passes all three: the infector's
infectiousness, its own susceptibility, and the timing against any removal.
This is step 3 above.

Interventions are applied in the order they are listed. A policy that starts on
a given date, or stops after a number of cases, is described by wrapping the
intervention in [`Scheduled`](@ref) instead of giving every intervention its
own start date. The start date applies to when the intervention would act on a
person, not to when they were infected: with isolation starting on day 14, a
case infected on day 10 whose isolation would fall on day 16 is still isolated.
How to write an intervention is in [Writing an
intervention](tutorials/writing-interventions.md).

### Multi-type branching processes

Several types (age groups, risk groups, spatial patches) are supported in the
draw of secondary cases, as in a stratified model. For an infector of type `j`,
the numbers of secondary cases of each type are drawn together, with the mixing
between types from a contact matrix and the form of the distribution from the
offspring distribution. Each contact is given a type. Interventions and output
are unchanged, because they act on individual people, not on types.

## Host timeline and transmission-route windows

Some diseases spread by several routes, each open over a different part of a
case's illness. Ebola spreads in the community while a case is ill, in
hospital between admission and discharge, and at funerals between death and
burial. Isolation cuts the community route short but not the others.

The simplest model has one offspring distribution and one generation-time
distribution. The general version treats a case's natural history as a
**host timeline**, a sequence of timed states from infection (infectious,
onset, severe, died or recovered, buried), and transmission as a set of
**route windows** on that timeline. A route window has its own offspring
distribution, a state at which infectiousness begins, the states that end it,
and a contact-interval distribution for the timing within it. Because the
timeline is the disease part of the model, the state at which a window opens is
taken from the model's disease part, not fixed in the transmission model.

This brings several mechanisms under one. A funeral route runs between death
and burial, a hospital route between admission and discharge, and isolation
lowering R is a route cut short by removal. The community route is the
simplest window (from infection, never cut short), which gives the plain
branching process. The latent period becomes the transition from infection to
infectiousness, the generation time is the latent period plus the
contact-interval draw, and isolation is one removal state among death, recovery
and burial.

All of this happens in steps 2 and 3; step 1 stays the same, and can still be
analysed on its own. R remains the reproduction number a case would have if
never removed, and the realised R follows from the removals: shortening the
infectious period blocks more contacts and lowers it. k remains the dispersion
of the number of secondary cases, deliberately not tied to the length of the
infectious period. The number of secondary cases is never drawn from a
duration, and step 1 stays independent of timing.

A window contributes contacts only once its opening state has happened, so a
survivor never has funeral contacts and no contact is created only to be
removed. The distribution of secondary cases that drives outbreak size is then
a mixture: community contacts for everyone, plus funeral contacts for the
proportion who die. It has a closed form when each part is a negative binomial
and falls back to simulation otherwise. Exact results and simulation still
agree.

The same quantities let a household or metapopulation model reuse this at a
smaller scale: a household is one route window limited to its members, with
the contact-interval distribution as the window's timing and the infectious
period as its end.

!!! note "Planned"
    The continuous-time half of this is built: `RouteWindow` records a route's
    opening state, the states that end it, its contact-interval distribution
    and whom it reaches, and the continuous-time simulation handles several
    routes per case, each ended separately, for any model that supplies them.
    The branching-process half, with the mixture distribution and its closed
    forms, is designed but not yet built. Exact results are therefore not yet
    available for models with several routes.

## One model for simulating and fitting

`simulate` (generate outbreaks) and `loglikelihood` (score data against a
model) read the same model, with all five parts. Simulating forward and
fitting to data therefore always use the same assumptions. If isolation reduced
transmission in the outbreak that produced the data, the likelihood of the
observed chain sizes is only correct because the same model applies isolation
too. To compare scenarios, build one model per scenario, each the same
transmission model under a different response; a counterfactual is a new model
with the part you want changed.

### Exact results and simulation

Where a closed form exists, EpiBranch uses it; simulation covers the rest, and
the two give the same answers.

New simulation behaviour comes from new interventions and transmission models.
On the closed-form side, two kinds of addition fit in:

- **Offspring specifications** replace what a branching process draws for each
  case, for example letting the offspring parameters vary from chain to chain.
  They are used in simulation for the draw of secondary cases and in closed-form
  results through their chain-size distribution.
- **Observation models** describe how the true outbreak becomes data. An
  observation model turns the distribution of true chain sizes into a
  distribution of observed ones, so it uses the same likelihood as the true
  sizes and needs no likelihood of its own. In simulation it marks the observed
  cases on a finished run.

Some combinations have closed forms. Poisson offspring with a
gamma-distributed rate gives the `gborel` chain-size distribution from
epichains, and the closed form is chosen automatically for that combination.
Otherwise the chain-size distribution is computed numerically.

### Which sampler to use

Closed-form likelihoods are deterministic functions of the parameters. They
work with automatic differentiation, which computes their gradients, and therefore
with gradient-based samplers such as the No-U-Turn Sampler (NUTS), a form of
Hamiltonian Monte Carlo. Simulation-based likelihoods use random numbers, and
each evaluation is a noisy estimate, and the gradient of a single simulation is
not a useful estimate of the gradient of the expected likelihood.
Gradient-based samplers should therefore not be used with them; use
gradient-free samplers (Metropolis–Hastings, particle methods) instead, as the
[Inference](tutorials/inference.md) tutorial shows.

## Connection to survival analysis

The generation-time distribution g(t) = h(t)/R is the normalised
infectiousness profile, and its cumulative distribution G(t) is the cumulative
hazard divided by R. When isolation happens at time t_iso, the probability that
a given transmission falls before isolation is G(t_iso) and after it is
1 − G(t_iso): transmission is right-censored at isolation. The generation-time
distribution is linked to the epidemic growth rate r by the Euler–Lotka
equation R = 1/M_g(−r), where M_g is its moment generating function. The same
two quantities, the generation-time distribution and the censoring time, appear
both in simulation and in Kenah's pairwise likelihood. Inference within this
framework therefore fits the same quantities it simulates from.

## References

- Kenah E, Lipsitch M, Robins JM (2008). Generation interval contraction and epidemic data analysis. *Mathematical Biosciences* 213(1):71–79. [doi:10.1016/j.mbs.2008.02.007](https://doi.org/10.1016/j.mbs.2008.02.007)
- Kenah E (2011). Contact intervals, survival analysis of epidemic data, and estimation of R0. *Biostatistics* 12(3):548–566. [doi:10.1093/biostatistics/kxq068](https://doi.org/10.1093/biostatistics/kxq068)
- KhudaBukhsh WR, Choi B, Kenah E, Rempala GA (2020). Survival dynamical systems: individual-level survival analysis from population-level epidemic models. *Interface Focus* 10(1):20190048. [doi:10.1098/rsfs.2019.0048](https://doi.org/10.1098/rsfs.2019.0048)
- Wallinga J, Lipsitch M (2007). How generation intervals shape the relationship between growth rates and reproductive numbers. *Proceedings of the Royal Society B* 274(1609):599–604. [doi:10.1098/rspb.2006.3754](https://doi.org/10.1098/rspb.2006.3754)
