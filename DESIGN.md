# EpiBranch.jl — Design

## Core idea

The offspring draw is completely decoupled from timing and interventions. The competing risks framework on the transmission hazard is connected naturally to survival analysis (Kenah's pairwise survival analysis, dynamic survival analysis).

## Architecture

### Three separated stages

#### 1. Offspring draw

Pure branching process, no time, no interventions.

For a parent of type `j`, contacts are drawn from an offspring distribution. This is the classical branching process: a count from a distribution (single-type) or a vector of counts per type (multi-type). The mean (R) and overdispersion (k) come from the offspring distribution. If a contact matrix is provided, the mixing pattern across types is determined by it.

#### 2. Timing

Generation times are assigned independently to each contact.

Each contact is given a generation time sampled from a distribution (either fixed or derived from the parent's incubation period, as in the ringbp-style model). This generation time is the *potential* time of transmission — the infectiousness profile, or h(t) in survival analysis terms.

#### 3. Competing risks

Which contacts are actually infected?

Each contact is resolved independently:
- Was the parent isolated before this contact's generation time? (hazard truncation)
- Is the contact susceptible? (vaccination, prior immunity, population depletion)
- Is the parent still infectious? (infectiousness modifier)
- For leaky isolation: does residual transmission succeed?

Contacts that fail any of these checks are not infected but are still stored. They represent contacts that were made but did not result in transmission. Intervention effort (contacts traced, vaccines administered, tests used) is tracked this way.

### Why this separation matters

Because the offspring distribution is a pure probabilistic object, it can be analysed with standard tools: extinction probability from the dominant eigenvalue, chain size distributions, analytical likelihoods. None of this depends on timing or interventions.

At the intervention stage, the timing layer is modified, not the offspring layer. The hazard is truncated by isolation. The truncation point is shifted earlier by contact tracing. Susceptibility is modified by vaccination. All of these are competing risks on the same set of potential contacts.

The generation time CDF evaluated at the intervention time *is* the survival function. The same mathematical objects appear in simulation and in Kenah's pairwise likelihood framework. Intervention effectiveness can be estimated from observed generation times using the same quantities used in simulation.

All contacts are stored, not just successful infections. This gives the **simulist**-style contacts table (with `was_case` flag) and intervention effort tracking, without any additional bookkeeping.

## Individual state

```
Individual (struct)
├── Core (used by the engine)
│   ├── id, parent_id, generation, chain_id    # transmission tree
│   ├── infection_time                          # timing
│   ├── susceptibility, infectiousness          # universal modifiers
│   └── secondary_case_ids                      # filled during simulation
│
└── state::Dict{Symbol, Any}                    # everything else
    ├── :onset_time, :asymptomatic, :test_positive   # clinical (set by init)
    ├── :isolated, :isolation_time                     # set by Isolation
    ├── :traced, :quarantined                          # set by ContactTracing
    ├── :infected                                      # set by competing risk resolution
    └── :age, :sex, :risk_group, ...                   # user-defined
```

Only `susceptibility`, `infectiousness`, and `infection_time` are read by the engine. Everything else is owned by interventions, attributes functions, or the user. Fields are initialised via `initialise_individual!` and accessed through accessor functions with safe defaults.

## Intervention interface

Three hooks, all optional:

- `initialise_individual!(intervention, individual, state)` — set up fields on a new contact
- `resolve_individual!(intervention, individual, state)` — determine intervention state before transmission (e.g. compute isolation time from onset + delay)
- `apply_post_transmission!(intervention, state, new_contacts)` — act on contacts after creation (e.g. contact tracing, ring vaccination). All contacts, infected and non-infected, are passed.

Interventions are composable. They are stacked in a vector and applied in order. Each intervention has its own fields on the individual and declares what fields it requires.

## Transmission modifiers

All interventions ultimately map onto two numbers:

- `susceptibility` (0--1): probability of being infected given exposure. Reduced by vaccination, prior immunity, or population depletion.
- `infectiousness` (0--1): modifier on onward transmission. Reduced by isolation (via hazard truncation), treatment, or asymptomatic status.

For a contact to be infected, it must survive the parent's infectiousness check AND the contact's susceptibility check AND the timing check (generation time vs isolation time).

## Multi-type branching processes

Multiple types (age groups, risk groups, spatial patches) are supported in the offspring draw. For a parent of type `j`, offspring counts per type are drawn from a joint distribution. The mixing pattern comes from a contact matrix; the count distribution comes from the offspring distribution family.

Each contact is allocated to a type, stored as `:type` in its state dict. Interventions and output are unchanged — they operate on individual-level state, not types.

## Connection to survival analysis

The generation time distribution g(t) = h(t)/R is the normalised infectiousness profile. Its CDF, G(t), is the cumulative hazard. When isolation occurs at time t_iso, the probabilities are:

- P(transmission before isolation) = G(t_iso)
- P(transmission after isolation) = 1 - G(t_iso)

The result is right-censoring of the transmission process. The generation time distribution (derived from the hazard) is connected to the population growth rate through the Euler-Lotka equation R = 1/M_g(-r).

The same objects -- the generation time distribution and the censoring time -- appear in both simulation and Kenah's pairwise likelihood. Inference built on top of this framework would fit the same quantities we simulate from.

## References

- Kenah E, Lipsitch M, Robins JM (2008). Generation interval contraction and epidemic data analysis. *Mathematical Biosciences* 213(1):71–79. [doi:10.1016/j.mbs.2008.02.007](https://doi.org/10.1016/j.mbs.2008.02.007)
- Kenah E (2011). Contact intervals, survival analysis of epidemic data, and estimation of R0. *Biostatistics* 12(3):548–566. [doi:10.1093/biostatistics/kxq068](https://doi.org/10.1093/biostatistics/kxq068)
- KhudaBukhsh WR, Choi B, Kenah E, Rempala GA (2020). Survival dynamical systems: individual-level survival analysis from population-level epidemic models. *Interface Focus* 10(1):20190048. [doi:10.1098/rsfs.2019.0048](https://doi.org/10.1098/rsfs.2019.0048)
- Wallinga J, Lipsitch M (2007). How generation intervals shape the relationship between growth rates and reproductive numbers. *Proceedings of the Royal Society B* 274(1609):599–604. [doi:10.1098/rspb.2006.3754](https://doi.org/10.1098/rspb.2006.3754)
