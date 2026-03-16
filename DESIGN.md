# EpiBranch.jl — Design

## Core idea

A branching process simulation where the offspring draw is completely decoupled from timing and interventions. Interventions operate through a competing risks framework on the transmission hazard, connecting the simulation engine to survival analysis (Kenah's pairwise survival analysis, dynamic survival analysis).

## Architecture

### Three separated stages

**1. Offspring draw** — pure branching process, no time, no interventions.

A parent of type `j` generates contacts according to an offspring distribution. This is the classical branching process: draw a count from a distribution (single-type) or a vector of counts per type (multi-type). The offspring distribution controls the mean (R) and overdispersion (k). The contact matrix, if present, determines the mixing pattern across types.

**2. Timing** — generation times assigned independently to each contact.

Each contact receives a generation time sampled from a distribution (fixed, or a function of the parent's incubation period for the ringbp-style model). The generation time defines the *potential* time of transmission. This is the infectiousness profile — the hazard function h(t) in survival analysis terms.

**3. Competing risks** — which contacts are actually infected?

Each contact is resolved independently:
- Was the parent isolated before this contact's generation time? (hazard truncation)
- Is the contact susceptible? (vaccination, prior immunity, population depletion)
- Is the parent still infectious? (infectiousness modifier)
- For leaky isolation: does residual transmission succeed?

Contacts that fail any of these checks are **not infected** but are still stored — they represent contacts that were made but did not result in transmission. This is essential for tracking intervention effort (contacts traced, vaccines administered, tests used).

### Why this separation matters

**For the branching process**: the offspring distribution is a pure probabilistic object. It can be analysed with standard tools — extinction probability from the dominant eigenvalue, chain size distributions, analytical likelihoods. None of this depends on timing or interventions.

**For interventions**: they operate on the timing layer, not the offspring layer. Isolation truncates the hazard. Contact tracing shifts the truncation point earlier. Vaccination modifies susceptibility. All of these are competing risks on the same set of potential contacts.

**For inference**: the generation time CDF evaluated at the intervention time *is* the survival function. The simulation engine and the inference framework (pairwise likelihood à la Kenah) share the same mathematical objects. You can estimate intervention effectiveness from observed generation times using the same quantities the simulation uses.

**For the contacts table**: all contacts are stored, not just successful infections. This gives the simulist-style contacts table (with `was_case` flag) and enables intervention effort tracking — without any additional bookkeeping.

## Individual state

```
Individual (struct)
├── Core (engine needs these)
│   ├── id, parent_id, generation, chain_id    # transmission tree
│   ├── infection_time                          # timing
│   ├── susceptibility, infectiousness          # universal modifiers
│   └── secondary_case_ids                      # filled during simulation
│
└── state::Dict{Symbol, Any}                    # everything else
    ├── :onset_time, :asymptomatic, :test_positive   # clinical (set by engine)
    ├── :isolated, :isolation_time                     # set by Isolation intervention
    ├── :traced, :quarantined                          # set by ContactTracing
    ├── :infected                                      # set by competing risk resolution
    └── :age, :sex, :risk_group, ...                   # user-defined
```

The engine only reads `susceptibility`, `infectiousness`, and `infection_time`. Everything else is owned by interventions or the user. Interventions initialise their own fields via `initialise_individual!` and read/write them through accessor functions with safe defaults.

## Intervention interface

Three hooks, all optional:

- `initialise_individual!(intervention, individual, state)` — set up fields on a new contact
- `resolve_individual!(intervention, individual, state)` — determine intervention state before transmission (e.g. compute isolation time from onset + delay)
- `apply_post_transmission!(intervention, state, new_contacts)` — act on contacts after creation (e.g. contact tracing, ring vaccination). Receives ALL contacts, infected and non-infected.

Interventions are composable: stack them in a vector, the engine applies them in order. Each intervention owns its own state on the individual and documents what fields it requires.

## Transmission modifiers

All interventions ultimately map onto two numbers:

- **`susceptibility`** (0–1): probability of being infected given exposure. Modified by vaccination, prior immunity, population depletion.
- **`infectiousness`** (0–1): modifier on onward transmission. Modified by isolation (via hazard truncation), treatment, asymptomatic status.

The competing risk check uses both: a contact must survive the parent's infectiousness check AND the contact's susceptibility check AND the timing check (generation time vs isolation time) to be infected.

## Multi-type branching processes

The offspring draw supports multiple types (age groups, risk groups, spatial patches). A parent of type `j` draws offspring counts per type from a joint distribution. The contact matrix provides the mixing pattern; the offspring distribution family provides the count distribution.

The step function allocates each contact to a type and sets `:type` in their state dict. The rest of the pipeline (interventions, output) works unchanged — it operates on individual-level state, not types.

## Connection to survival analysis

The generation time distribution g(t) = h(t)/R is the normalised infectiousness profile. The CDF G(t) is the cumulative hazard. When isolation occurs at time t_iso:

- P(transmission before isolation) = G(t_iso)
- P(transmission after isolation) = 1 - G(t_iso)

This is right-censoring of the transmission process. The Euler-Lotka equation R = 1/M_g(-r) connects the generation time distribution (derived from the hazard) to the population growth rate.

The simulation engine and Kenah's pairwise likelihood share the same objects: the generation time distribution and the censoring time. Building inference on top of this framework means fitting the same quantities we simulate from.

## References

- Kenah & Robins (2007). Generation interval contraction and epidemic data analysis.
- Kenah (2011). Contact intervals, survival analysis of epidemic data, and estimation of R0.
- KhudaBukhsh et al. (2020). Survival dynamical systems.
- Hart et al. (2022). Generation time and effective reproduction number under preventive measures.
- Wallinga & Lipsitch (2007). How generation intervals shape the relationship between growth rates and reproductive numbers.
