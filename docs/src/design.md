# EpiBranch.jl — Design

This page is the high-level design: the ideas the package is built on and
the reasons for them. It deliberately avoids type signatures, field
layouts, and API names, which live with the code and in the
[Extending guide](@ref "Extending EpiBranch") and [API reference](api.md).
The design principles below are the basis on which the shape here is
judged.

## Design principles

These are the principles EpiBranch is meant to satisfy. They should
be revisited when adding anything substantial, and the architecture
should be reviewed against them periodically.

### 1. Simple but rigorous

Express only what we need. A new mechanism earns its place only when an
analysis we actually do can't be done without it. Prefer closed-form
likelihoods over simulation when both are available and equivalent. One
verb (`loglikelihood`, `simulate`) does the dispatch, without
specialised wrapper functions per data type or model variant.

### 2. Self-explanatory

Type names, function names, and signatures should match the intuition
the epidemiology gives for what they do. If a user has to read
source to understand what a public name means, the name is wrong. The
mathematical names (`Borel`, `GammaBorel`) are fair when they match
the literature the user comes from; the operational names should
match how the epidemiology describes what's happening.

### 3. Cleanly separable concerns

Process model, observation model, data, inference, simulation, and
output each own one thing. Their interfaces are explicit. A new
alternative (a network model, multi-stream observation, time-varying
reporting, aggregated counts) slots in by implementing the relevant
interface rather than by editing core code. Each concern is replaceable
independently.

### 4. Extensible from outside

A user can add their own transmission model, observation model,
intervention, or data type as a separate package or script, reusing
all framework infrastructure. The contracts they have to satisfy are
documented and small. Adding a custom piece does not require editing
EpiBranch.

### 5. Documented with examples

Principle 4 is empty without 5. Every public extension point has a
worked example. Tutorials are checked at build time so the prose
stays consistent with the code.

## Core idea

The offspring draw is completely decoupled from timing and from
interventions. Who infects whom is a pure probabilistic object; when
transmission happens, and whether a control measure stops it, are separate
layers laid on top. This separation is what connects the package to
survival analysis: the competing-risks framework on the transmission
hazard is the same mechanism as Kenah's pairwise survival analysis and
dynamic survival analysis.

## The model is a composition of layers

A *model* is the whole generative specification of how data arises, built by
composing a transmission process with the modelling layers laid on top of it.
The transmission process is the between-host mechanism alone — how
infection spreads — and it carries nothing else. Onto it compose four layers:

- **disease** — the within-host natural history: the timed states a case moves
  through (latent, infectious, onset, severe, recovered or died), with
  treatment expressible as a step it may pass through;
- **attributes** — who the people are (age, susceptibility, infectiousness),
  which can bear on transmission, on the disease course, and on how
  interventions find their targets;
- **interventions** — what is done about it (isolation, tracing, vaccination);
- **observation** — how cases are seen (under-reporting, reporting delays).

These match how the epidemiology decomposes an outbreak: a pathogen spreads,
infection causes disease, in a population of people, under a response, watched
through surveillance. Keeping the five distinct — rather than folding the
disease or the policy into the process — is what lets each be replaced on its
own and lets the same layer sit on any process.

The *composed* model, not the bare process, is what both entry points read, so
"what could this produce?" (`simulate`) and "how likely was this data?"
(`loglikelihood`) can never disagree: a simulation-based likelihood reproduces
the same generative specification that produced the data. If isolation
suppressed transmission in the observed outbreak, the likelihood of those chain
sizes is only correct because the same composition applies isolation. A
scenario sweep is a map over compositions, each scenario the same process under
a different response; a counterfactual is a fresh composition with the layer
you want. There is no in-place "swap one input" helper, so a model's layers are
always explicit.

The layers are interlinked by nature — attributes bear on transmission,
disease, and targeting at once; the infectious window is defined by disease
states; a transmission rate expressed as a reproduction number depends on the
mean infectious period. The composition is what gives each layer access to the
others it needs, resolving those couplings where both sides are in hand rather
than by fusing the tiers. A structure-driven process, for one, derives its
infectious window — and, where transmission is given as a reproduction number,
its rate — from the composed disease when the model is simulated or scored, so
the process stays purely the transmission and the disease stays a single,
separately specified layer.

## Three separated stages

The engine resolves every transmission in three stages. Only the first is
model-specific; the other two run the same way for every model.

### 1. Offspring draw (model-specific)

Pure branching process: no time, no interventions. For a parent, contacts
are drawn from an offspring distribution: a single count, or a count per
type for a multi-type model. The mean (R) and overdispersion (k) come from
that distribution; if a contact matrix is supplied, it sets the mixing
pattern across types.

This is the only stage a transmission model defines, and it does so in one
of two ways. An **offspring-driven** model (the branching process and its
variants) can produce its candidates one parent at a time, returning a
count. A **structure-driven** model (a network, household, or
metapopulation process) cannot, because a susceptible may be reachable by
several infectious neighbours at once and infections deplete a fixed pool;
it instead names the candidate contacts each infectious node reaches, and
the engine resolves a node reached several times in one generation once.
The companion `EpiNetwork.jl` package's network process is the worked
example. Either way the model only says who contacts whom.

### 2. Timing (shared)

Each candidate is given a transmission time from the parent's
infectiousness profile (the generation-time distribution). That
distribution can be fixed or built per individual, so the timing can read
any per-individual quantity: the parent's incubation period, or anything
an attributes function has stored. This is the *potential* time of
transmission: h(t) in survival-analysis terms.

### 3. Competing risks (shared)

Each candidate is resolved independently to infected or not, by a single
per-pair decision composed of a list of risk sources. Parent
infectiousness and contact susceptibility are not privileged engine checks:
they are default risk sources on the same surface an intervention uses, and
isolation truncation or any risk an intervention contributes joins the same
list. A model can contribute its own sources here too, for a transmission
term that belongs to the *edge* rather than a node (a network's per-edge
probability): the per-node susceptibility and infectiousness terms cannot
carry a per-pair quantity without forcing one shared value across a node's
edges, so it goes here instead. A contact is infected only if no risk blocks
it, so the earliest removal before the contact's time wins, with no special
min-logic.

Contacts that fail a check are still stored. They are contacts that were
made but did not transmit, and they carry the contact-tracing table and
intervention-effort tracking (contacts traced, vaccines administered, tests
used) with no extra bookkeeping.

### Why this separation matters

Because the offspring distribution is a pure probabilistic object, it can
be analysed with standard tools: extinction probability from the dominant
eigenvalue, chain-size distributions, analytical likelihoods. None of these
depends on timing or interventions, and the draw stays differentiable
where the distribution permits. Interventions act on the timing and
competing-risks layers, not the offspring layer: isolation truncates the
hazard, contact tracing shifts the truncation earlier, vaccination lowers
susceptibility. The generation-time CDF evaluated at an intervention time
*is* the survival function, so the same objects appear in simulation and in
Kenah's pairwise likelihood, and intervention effectiveness can be
estimated from observed generation times using the quantities used to
simulate.

## Extension by dispatch

New behaviour is added by defining a new type and a method, not by growing
options on an existing struct. A user wanting a variant should be able to
write a small struct plus one or two methods, with no edits to the package
source and no copy-pasting of existing function bodies. A `Union` field
whose members trigger different branches, a `Symbol` that switches
behaviour inside a function, or a `Bool` that selects a policy are all
signals that a seam is in the wrong place and should become a dispatched-on
type.

This holds on every axis:

- **Transmission models** — spatial, network, and immunity dynamics enter
  through new model subtypes or reusable wrapper types, not flags on the
  branching process.
- **Interventions** are orchestrators of smaller dispatched pieces. An
  intervention struct is a thin shell wiring together independently
  dispatched components (eligibility, fire-rate, delay, effect), each a
  type with a method. The intervention body holds no hardcoded policy
  branching. Composition then works at two levels: between interventions
  (the stack the model carries) and within each intervention (its pieces).
- **Output, observation, and outcome rules** follow the same shape:
  mortality, hospitalisation, reporting, stopping conditions, and line-list
  columns are typed objects with methods, not closed sets of fields.

The test of correctness for any component: can a plausible new variant be
added without editing the component's source? If not, the component is
doing too much, and the varying part should be lifted into a dispatched-on
trait. The concrete contracts (which methods each axis requires, with
worked examples) are in the [Extending guide](@ref "Extending EpiBranch").

## Individual state

Each individual carries a small typed core that the engine reads, plus an
open dictionary for everything else. The core holds its place in the
transmission tree, its infection time, and the two universal modifiers
(susceptibility and infectiousness). The dictionary is the deliberate
extension hatch: interventions, attributes builders, clinical transitions,
and observation models each own a small set of keys, and the engine never
inspects them.

A typed core would couple the struct to every intervention's state shape
and would break the dose-label namespacing that lets multi-dose
vaccination schedules coexist. The dictionary keeps the individual
independent of which pieces a user composes. The keys the package itself
reserves, and the convention for naming keys added from other packages, are
listed in the [Extending guide](@ref "Extending EpiBranch").

## Interventions

All interventions map onto two numbers: **susceptibility**
(probability of infection given exposure, reduced by vaccination, prior
immunity, or population depletion) and **infectiousness** (a modifier on
onward transmission, reduced by isolation, treatment, or asymptomatic
status). A contact is infected only if it survives the parent's
infectiousness check, its own susceptibility check, and the timing check
(generation time versus isolation time), the competing-risks resolution of
stage 3.

Interventions are stacked and applied in order, each owning its own state
and declaring the fields it requires. Time-dependent policies (start on day
14, stop after N cases) are expressed by wrapping an intervention in a
schedule rather than by giving every intervention its own start-time field;
the gate is on the *action* time, so an individual infected before a policy
starts can still be affected if the time they would be tested falls after
it. The hook contract and a worked example are in the
[Extending guide](@ref "Extending EpiBranch").

## Multi-type branching processes

Multiple types (age groups, risk groups, spatial patches) are supported
in the offspring draw: for a parent of type `j`, offspring counts per type
are drawn from a joint distribution, with the mixing pattern from a contact
matrix and the count family from the offspring distribution. Each contact
is allocated to a type, and interventions and output are unchanged because
they operate on individual-level state, not on types.

## Analytical and simulation duality

Where a closed form exists, EpiBranch uses it; simulation covers the rest,
and the two are kept consistent.

The simulation side is extended through the intervention and model
protocols above. The analytical side uses dispatch on the existing model
types, with no new abstract type. Two kinds of extension plug in:

- **Offspring specifications** replace what a branching process draws per
  individual, for example letting the offspring parameters vary from chain
  to chain. They participate in simulation through the offspring draw and in
  analytics by returning a chain-size distribution.
- **Observation models** capture how the latent process generates data. An
  observation returns a *transformed distribution* of the latent
  chain-size law, so it goes through the same likelihood path as the latent
  law and needs no bespoke likelihood method; each observation is written
  once, as a transform. On the simulation side it marks observed cases on a
  finished run.

Some combinations have closed forms. Poisson offspring with a
Gamma-distributed rate gives the `gborel` law from epichains, and these
plug in by specialising the chain-size distribution on the relevant type
combination, so dispatch picks the closed form without the user asking. The
generic case falls back to quadrature.

Any extension with both an analytical chain-size distribution and a
simulation path should have a regression test confirming they agree; the
test suite provides a helper for this.

### Which sampler to use

Analytical likelihoods are deterministic scalar functions of the
parameters: they work with any AD backend and with gradient-based samplers
like NUTS. Simulation-based likelihoods draw random numbers, so the output
is a noisy estimate and the gradient of a single realisation is not a
useful estimate of the gradient of the expected likelihood, so
gradient-based samplers should not be used. Use gradient-free samplers
(Metropolis–Hastings, particle methods) instead, as the inference tutorial
shows.

### Why mutate in place

The engine works in place, one generation at a time. Copying the whole
state every generation would be far too expensive for an unbounded tree, so
the engine mutates on purpose.

## Connection to survival analysis

The generation-time distribution g(t) = h(t)/R is the normalised
infectiousness profile, and its CDF G(t) is the cumulative hazard. When
isolation occurs at time t_iso, P(transmission before isolation) = G(t_iso)
and P(transmission after) = 1 − G(t_iso): the transmission process is
right-censored. The generation-time distribution connects to the population
growth rate through the Euler–Lotka equation R = 1/M_g(−r). The same two
objects, the generation-time distribution and the censoring time, appear
in both simulation and Kenah's pairwise likelihood, so inference built on
this framework fits the same quantities it simulates from.

## Host timeline and transmission-route windows

The three stages above describe the simplest model: one offspring law, one
generation-time distribution. The branching process generalises this against a
**host timeline** (the disease layer: the case's natural history as a sequence
of timed states from infection — infectious, onset, severe, died or recovered,
buried) composed onto it, treating transmission as a set of **route windows**
keyed off that timeline. A route window is an offspring law, a `from` state
where infectiousness begins, the states that end it, and a survival kernel for
the timing within it. Because the timeline is a separate layer, the window's
`from` state is resolved against the composed disease rather than fixed on the
process.

This is what lets several mechanisms become one. A funeral route runs
between death and burial; a nosocomial route between admission and
discharge; isolation lowering R is a route cut short by a removal state.
The community route is the simplest window (from infection, no censoring),
which recovers today's model unchanged. The latent period becomes the
infection→infectious transition, the generation interval is derived from
the latent period plus the kernel draw, and isolation truncation becomes
one removal state among death, recovery, and burial.

The generalisation lands entirely in stages 2 and 3; stage 1 stays the
pure, analysable layer. R remains the intrinsic reproduction number a case
would make if never removed, and the realised R falls out of the censoring:
shortening the infectious window blocks more contacts and lowers it. k
remains the intrinsic offspring dispersion and is deliberately not the
shape of the infectious period, so the count is never drawn from a
duration, which would couple stage 1 to timing and break the
branch-first decoupling the engine rests on.

A window contributes contacts only once its `from` state has occurred, so a
survivor never materialises funeral contacts and nothing is created only to
be censored. The offspring law driving outbreak size is therefore a
fate-mixture (community offspring for everyone, plus funeral offspring for
the fraction who die), which stays closed-form when each branch is a
negative binomial and falls back to simulation otherwise, so the
analytical/simulation duality carries over.

This is also the seam through which a household- or
metapopulation-structured model in the wider ecosystem reuses the same
survival objects at a smaller scale: a household is one route window scoped
to a clique, with the contact-interval kernel as the window kernel and the
infectious period as its censoring.

## References

- Kenah E, Lipsitch M, Robins JM (2008). Generation interval contraction and epidemic data analysis. *Mathematical Biosciences* 213(1):71–79. [doi:10.1016/j.mbs.2008.02.007](https://doi.org/10.1016/j.mbs.2008.02.007)
- Kenah E (2011). Contact intervals, survival analysis of epidemic data, and estimation of R0. *Biostatistics* 12(3):548–566. [doi:10.1093/biostatistics/kxq068](https://doi.org/10.1093/biostatistics/kxq068)
- KhudaBukhsh WR, Choi B, Kenah E, Rempala GA (2020). Survival dynamical systems: individual-level survival analysis from population-level epidemic models. *Interface Focus* 10(1):20190048. [doi:10.1098/rsfs.2019.0048](https://doi.org/10.1098/rsfs.2019.0048)
- Wallinga J, Lipsitch M (2007). How generation intervals shape the relationship between growth rates and reproductive numbers. *Proceedings of the Royal Society B* 274(1609):599–604. [doi:10.1098/rspb.2006.3754](https://doi.org/10.1098/rspb.2006.3754)
