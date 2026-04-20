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

Interventions are stacked in a vector and applied in order. Each intervention has its own fields on the individual and declares what fields it requires.

### Time-based scheduling

Interventions with a `start_time` field represent policies that are not available from the start. The framework enforces `start_time` generically: after each hook call, it checks whether the intervention's effect on an individual falls before the policy start and undoes it if so.

Each intervention declares:

- `intervention_time(intervention, individual)` — the time at which the effect occurs (e.g. isolation time, trace time). Default: `-Inf` (always applies).
- `reset!(intervention, individual)` — reverts all state changes made by the intervention. Default: no-op.
- `start_time(intervention)` — the policy start time. Default: `0.0`.

The check is on *action time*, not infection time. An individual infected before the policy start can still be affected if their computed action time (e.g. `onset + delay`) falls after `start_time`. This is the correct competing-risk interpretation: the testing infrastructure must be available at the time the individual would be tested.

The `Scheduled` wrapper adds population-level conditions (case-count triggers, time windows, custom predicates) on top of this mechanism. It forwards `start_time` to the inner intervention when provided.

## Transmission modifiers

All interventions ultimately map onto two numbers:

- `susceptibility` (0--1): probability of being infected given exposure. Reduced by vaccination, prior immunity, or population depletion.
- `infectiousness` (0--1): modifier on onward transmission. Reduced by isolation (via hazard truncation), treatment, or asymptomatic status.

For a contact to be infected, it must survive the parent's infectiousness check AND the contact's susceptibility check AND the timing check (generation time vs isolation time).

## Multi-type branching processes

Multiple types (age groups, risk groups, spatial patches) are supported in the offspring draw. For a parent of type `j`, offspring counts per type are drawn from a joint distribution. The mixing pattern comes from a contact matrix; the count distribution comes from the offspring distribution family.

Each contact is allocated to a type, stored as `:type` in its state dict. Interventions and output are unchanged — they operate on individual-level state, not types.

## Composing analytical methods

The simulation side is extended by implementing the intervention protocol. The analytical side uses a different pattern. Each extension is a Julia type with a method on `chain_size_distribution` or `loglikelihood`, or both. There is no new abstract type beyond the existing `TransmissionModel`; multiple dispatch handles the rest.

### Two kinds of extension

Offspring specifications replace what a branching process draws per individual. An example is `ClusterMixed(build, mixing)`, which lets the offspring distribution's parameters vary from chain to chain. These are stored in `BranchingProcess.offspring` and participate via:

- `_draw_offspring(rng, offspring_spec, individual, state)` for simulation
- `chain_size_distribution(offspring_spec)` for analytics (returns a distribution)

Transmission-model wrappers modify how a model is observed. They subtype `TransmissionModel` and hold a wrapped model. `PartiallyObserved(model, detection_prob)` is the current example. They participate via:

- `chain_size_distribution(wrapper)` returning a distribution that transforms the wrapped model's chain size distribution (e.g. `ThinnedChainSize(chain_size_distribution(m.model), p)`)
- `loglikelihood(data, wrapper)`, which usually just routes through `chain_size_distribution`

### Why wrappers return distributions

`chain_size_distribution(m::PartiallyObserved)` returns `ThinnedChainSize(base, p)`. The result is itself a `DiscreteUnivariateDistribution` with its own `logpdf`, so a future wrapper such as `TimeCensored` can wrap it: `CensoredChainSize(ThinnedChainSize(base, p), p_c)`. Composition is then just nesting the distributions.

If instead the thinning were computed inside `loglikelihood(::PartiallyObserved)`, adding a second wrapper would require working out its joint likelihood with `PartiallyObserved` by hand, and every new pair would need the same treatment. Returning a distribution means each wrapper can be written once.

### Closed forms as dispatch optimisations

Some combinations of offspring and mixing have closed forms. Poisson offspring with a Gamma-distributed rate gives `PoissonGammaChainSize`, which is the `gborel` likelihood from `epichains`. These plug in by specialising `chain_size_distribution`:

```julia
chain_size_distribution(o::ClusterMixed) = ChainSizeMixture(o.build, o.mixing)
chain_size_distribution(o::ClusterMixed{PoissonFamily, <:Gamma}) = PoissonGammaChainSize(...)
```

The generic case uses adaptive Gauss-Kronrod quadrature via `ChainSizeMixture`. When a closed form applies, dispatch uses it without the user having to ask.

### Pipe composition

Wrappers have curried constructors so pipes work:

```julia
model |> PartiallyObserved(0.7)
model |> PartiallyObserved(0.5) |> PartiallyObserved(0.5)   # compounds to 0.25
```

When the wrappers don't commute, order matters and the pipe reads left to right.

## Simulation, mutation, and automatic differentiation

The simulation engine uses in-place mutation: `step!` appends individuals, updates case counts, and modifies individual state via interventions. This is deliberate — branching process simulations grow an unbounded tree, and copying the full state at every generation would be prohibitively expensive.

### Analytical likelihoods

`loglikelihood(ChainSizes(data), Poisson(R))` and similar analytical methods are deterministic scalar functions of the parameters. They work with any AD backend (ForwardDiff, Enzyme, Mooncake, ReverseDiff) and with gradient-based samplers like NUTS.

### Simulation-based likelihoods

`loglikelihood(ChainSizes(data), model; interventions=...)` runs stochastic simulations internally. Because the simulation draws random numbers, the output is a noisy estimate of the true likelihood, not a smooth function of the parameters. Gradients of a single stochastic realisation are not useful estimates of the gradient of the expected likelihood. Gradient-based samplers (NUTS, HMC) should not be used here.

Use gradient-free samplers instead: Metropolis-Hastings (`MH()`), particle methods, or similar. The inference tutorial demonstrates this with `MH()`.

### Sim ↔ analytical consistency

An extension with both an analytical chain size distribution and a simulation path should have a regression test confirming they agree. The test suite has a helper at `test/testutils/sim_analytical_consistency.jl`. A new type plugs in by defining two methods:

- `generative_model(m)` strips observation wrappers so `simulate_batch` can run
- `observe_chain_sizes(m, true_sizes, rng)` transforms simulated true chain sizes into observed ones (defaults to the identity; observation wrappers override it)

`sim_analytical_consistent(model; n_chains, sizes, rng)` then simulates, applies `observe_chain_sizes`, and compares the empirical PMF against `chain_size_distribution(model)`. The same helper already covers bare offspring, `ClusterMixed`, and `PartiallyObserved`.

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
