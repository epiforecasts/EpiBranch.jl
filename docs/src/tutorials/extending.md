# Extending EpiBranch

If you only want to *use* the interventions, attributes and models that ship
with EpiBranch, you don't need this page — start with
[Interventions](interventions.md) and the other tutorials. This page is for
writing new pieces in Julia.

## Extension points

There are a handful of places to extend EpiBranch, in increasing order of how
much you write:

- **Configure a built-in intervention** — add an existing control measure
  (`Isolation`, `ContactTracing`, `RingVaccination`, `MassVaccination`) to a
  model by keyword. This is applied, end-user work and lives in
  [Interventions](interventions.md), not here.
- **Write a custom intervention** — subtype `AbstractIntervention` and implement
  its hooks to add a risk the built-ins don't cover. A new *behaviour* on an
  existing process. Covered below.
- **Add a transmission model** — subtype `TransmissionModel` to add a whole new
  transmission *process* (network-, household- or metapopulation-structured, a
  continuous-time alternative). The deepest surface. Covered below.
- **Add an observation or data type** — subtype `ObservationModel`, or define a
  `loglikelihood` method for a new data type. Covered below.

The two surfaces most people reach for are a **custom intervention** (a new risk
on an existing model) and a **custom transmission model** (a new process); both
are developer work in Julia. This guide also covers custom attributes and
offspring along the way.

## Individual state and reserved keys

Each individual carries a small typed core read by the engine plus an open
`state` dictionary that everything else writes into (see
[Individual state](@ref) in the design notes for why). Interventions,
attributes functions, clinical transitions, and observation models each own
a few keys in that dictionary.

Read a key through a one-line accessor that pins the result type and
supplies a safe default: `onset_time(ind) = get(ind.state, :onset_time, NaN)::Float64`.
New code should add an accessor in `src/state_accessors.jl` rather than
calling `get(ind.state, …)` directly.

### Reserved keys

The keys below are reserved by the package. Custom interventions and
downstream packages should pick names that do not collide.

| Key | Type | Default | Owner | When set |
|---|---|---|---|---|
| `:infected` | `Bool` | `true` | Engine | Competing-risks resolution |
| `:type` | `Int` | `1` | Engine (multi-type) | Contact creation |
| `:onset_time` | `Float64` | `NaN` | `clinical_presentation` | Init |
| `:asymptomatic` | `Bool` | `false` | `clinical_presentation` | Init |
| `:age` | `Real` | — | `demographics` | Init |
| `:sex` | `Symbol` | — | `demographics` | Init |
| `:risk_group` | `Symbol` | — | `demographics` | Init |
| `:isolated` | `Bool` | `false` | `Isolation` | `resolve_individual!` |
| `:isolation_time` | `Float64` | `Inf` | `Isolation` | `resolve_individual!` |
| `:test_positive` | `Bool` | `false` | `Isolation` | `resolve_individual!` |
| `:traced` | `Bool` | `false` | `ContactTracing` | `apply_post_transmission!` |
| `:quarantined` | `Bool` | `false` | `ContactTracing` | `apply_post_transmission!` |
| `:traced_isolation_time` | `Float64` | `Inf` | `ContactTracing` → `Isolation` | Internal handoff |
| `:trace_time` | `Float64` | — | `ContactTracing` (`depth > 1`) | `apply_post_transmission!` |
| `:ring_remaining` | `Int` | `0` | `ContactTracing` (`depth > 1`) | `apply_post_transmission!` |
| `:vaccinated[_<label>]` | `Bool` | `false` | `AbstractVaccination` | Init / `apply_post_transmission!` |
| `:vaccination_time[_<label>]` | `Float64` | `Inf` | `AbstractVaccination` | `apply_post_transmission!` |
| `:vaccine_efficacy[_<label>]` | `Float64` | — | `AbstractVaccination` | `apply_post_transmission!` |
| `:reporting_time` | `Float64` | `Inf` | `Reporting` transition | `resolve_individual!` |
| `:admitted` | `Bool` | `false` | `Hospitalisation` transition | `resolve_individual!` |
| `:admission_time` | `Float64` | `Inf` | `Hospitalisation` transition | `resolve_individual!` |
| `:death_candidate_time` | `Float64` | `Inf` | `Outcome` transition | `resolve_individual!` |
| `:recovery_candidate_time` | `Float64` | `Inf` | `Outcome` transition | `resolve_individual!` |
| `:outcome` | `Symbol` | — | `Outcome` transition | `resolve_individual!` (terminal) |
| `:outcome_time` | `Float64` | — | `Outcome` transition | `resolve_individual!` (terminal) |
| `:reported` | `Bool` | `false` | `PerCaseObservation` *or* `Reporting` transition | Post-simulation projection / `resolve_individual!` |
| `:report_time` | `Float64` | — | `PerCaseObservation` | Post-simulation projection |
| `:cluster_theta` | `Float64` | — | `ClusterMixed` analytics | First simulation read |

The vaccination keys are namespaced by `dose_label`: the default label
writes to plain `:vaccinated` / `:vaccination_time` / `:vaccine_efficacy`,
and any other label suffixes the key (so `dose_label = :boost` writes
`:vaccinated_boost`, etc.). This lets multi-dose schedules compose without
colliding.

`:reported` is shared between the `Reporting` clinical transition (which
sets it from a probability gate) and `PerCaseObservation` (which sets it
post-simulation from a detection-probability draw). Composing both in the
same simulation is not supported, because they will overwrite each other.

Built-in keys use short bare names like `:isolated`, `:traced`, `:age`, and
those names are reserved. If you add keys from another package, prefix them
with a short tag for your package so they do not collide with built-ins or
with keys other packages might add.

## Custom interventions

Every intervention is a struct that subtypes `AbstractIntervention`. The
engine calls these hooks on each intervention; you implement only the
ones your intervention needs (all default to no-ops).

### Hook contract

| Hook | Called | Receives | Must return |
|---|---|---|---|
| `initialise_individual!(iv, individual, state)` | Once when each individual is created | An `Individual` whose typed fields are set but whose `state` dict is empty | `nothing` (mutate `individual.state` in place) |
| `resolve_individual!(iv, individual, state)` | Once per active individual at the start of each generation, before offspring are drawn | The parent for the upcoming step | `nothing` (mutate `individual.state` in place) |
| `apply_post_transmission!(iv, state, new_contacts)` | Once per generation after all contacts for that generation have been created (across every active parent) | A `Vector{Individual}` of the new contacts | `nothing` (mutate any of the contacts' `state` in place) |
| `competing_risk(iv, parent, contact, state)` | Per `(parent, contact)` pair during infection resolution, after `apply_post_transmission!` has run | The parent and a single new contact | `nothing`, a single [`Risk`](@ref), or an `NTuple{N, Risk}` for interventions that gate transmission via more than one mechanism |
| `keep_active(iv, state, targets, is_new)` | Once per generation after infection is resolved, while the engine builds the next active set | This generation's `targets` and an `is_new` flag per target | An iterable of contact ids to keep generating contacts into the next generation (default: none) |

Ordering guarantees:

- `resolve_individual!` runs strictly before any `competing_risk` call for that generation, so a competing risk can read whatever `resolve_individual!` wrote on the parent.
- `apply_post_transmission!` runs strictly before any `competing_risk` call, so a competing risk can read whatever post-transmission hook wrote on the contact (e.g. `:vaccination_time`).
- `keep_active` runs after infection is resolved, so it can read each target's `:infected` and anything `apply_post_transmission!` wrote on it this generation.
- Interventions are applied in the order they appear in `interventions = [...]`. For `apply_post_transmission!` and `competing_risk`, every intervention sees the state written by earlier interventions in the same generation.

A `Risk` applies to a contact when `event_time <= contact.infection_time`; in that case transmission is blocked with probability `block_probability`. Returning multiple risks (as a tuple) lets one intervention gate transmission through several mechanisms — `RingVaccination` returns both a susceptibility risk on the contact and an onward-infectiousness risk on the parent.

Tree-shaping changes — capping offspring per parent, gathering-size limits, anything that's really "this parent produces fewer contacts than its natural offspring distribution would say" — belong in the offspring distribution itself, not in the intervention protocol. See [Tree-shaping via the offspring distribution](#tree-shaping-via-the-offspring-distribution) below.

### What each hook looks like in practice

Short snippets from the built-in interventions, one per hook, to make the contract above concrete. The full source lives in `src/interventions/`.

**`initialise_individual!`** — `ContactTracing` initialises the two flags it owns on every new individual so accessors elsewhere get a defined value:

```julia
function initialise_individual!(::ContactTracing, individual, state)
    individual.state[:traced] = false
    individual.state[:quarantined] = false
    return nothing
end
```

**`resolve_individual!`** — `Isolation` computes the isolation time for the upcoming generation's parent from the individual's onset time plus a sampled delay, and folds in any earlier trace-driven isolation time that `ContactTracing` may have written on a previous generation:

```julia
function resolve_individual!(iso::Isolation, individual, state)
    is_isolated(individual) && return nothing
    is_test_positive(individual) || return nothing

    iso_delay = rand(state.rng, iso.onset_to_isolation_delay)
    iso_time = onset_time(individual) + iso_delay

    traced_time = get(individual.state, :traced_isolation_time, Inf)
    set_isolated!(individual, min(iso_time, traced_time))
    return nothing
end
```

**`apply_post_transmission!`** — `ContactTracing` walks the new contacts, looks up each contact's parent, and applies the configured trace action (`Quarantine` or `FlagOnly`) when the eligibility and rate traits both pass:

```julia
function apply_post_transmission!(ct::ContactTracing, state, new_contacts)
    rng = state.rng
    for ind in new_contacts
        ind.parent_id == 0 && continue
        parent = state.individuals[ind.parent_id]
        is_eligible(ct.eligibility, parent, ind, state) || continue
        traces(ct.trace_rate, parent, ind, state, rng) || continue
        trace_delay = draw_trace_delay(ct.isolation_to_trace_delay, parent, ind, state, rng)
        trace_time = isolation_time(parent) + trace_delay
        apply_trace!(ct.action, ind, state, trace_time, rng)
    end
    return nothing
end
```

**`competing_risk`** — see the [`BorderClosure` minimal example](#minimal-example-a-custom-competing-risk) below for a complete worked custom intervention.

### Verifying your intervention

The engine never errors when a hook is missing — every hook has a no-op default. That is convenient for partial implementations but means that *forgotten* hooks fail silently. Quick checks:

- Run a tiny simulation (`max_cases = 50`) with and without your intervention in the stack. If the outcome looks the same in both, your `competing_risk` or `apply_post_transmission!` is probably not being called for the cases you think.
- Override `required_fields` (see below) so the engine fails at simulation start when an upstream attributes function hasn't set a field your intervention needs.
- Inspect `state.individuals[1].state` after a small run to confirm your hook actually wrote the keys downstream code reads.

### Minimal example: a custom competing risk

A "border closure" intervention that blocks transmission between
contacts in different regions after a given date. Each individual
carries `:region` as a custom attribute; the intervention's
`competing_risk` reads both parent and contact regions and contributes
a blocking risk when they differ.

```@example extending
using EpiBranch
using Distributions
using StableRNGs

struct BorderClosure <: AbstractIntervention
    start_time::Float64
    leakage::Float64   # residual cross-border transmission probability
end

function EpiBranch.competing_risk(bc::BorderClosure, parent, contact, state)
    parent.state[:region] == contact.state[:region] && return nothing
    return Risk(event_time = bc.start_time,
        block_probability = 1.0 - bc.leakage)
end
```

`event_time = bc.start_time` means the risk only applies to contacts
whose transmission time is on or after the closure date; cross-border
transmissions before the closure are unaffected.

### Built-in transmission terms are risk sources too

The host's susceptibility and the infector's infectiousness are not
special engine rules. They are default risk sources on the same
`competing_risk` surface your `BorderClosure` plugs into. The engine
evaluates `[built-ins; your interventions]` through one shared path and
privileges neither, so `competing_risk` is the whole vocabulary for
gating transmission: a vaccine, a border closure, and the host's own
susceptibility all speak it.

Three defaults ship, each contributing a block probability:

- [`EpiBranch.HostSusceptibility`](@ref) — `1 - susceptibility` on the contact.
- [`EpiBranch.InfectorInfectiousness`](@ref) — `1 - infectiousness` on the parent.
- [`EpiBranch.InfectiousSource`](@ref) — a full block when the source is
  not infected, so an uninfected node can stay active (see below) and
  generate contacts without infecting them. A no-op in the usual case
  where every active node is infected.

A trait of `1.0` contributes no risk, so the defaults are silent unless
an attributes function sets a susceptibility or infectiousness below one.
You can replace or extend them by adding your own `competing_risk` the
same way.

### Growing the contact graph with `keep_active`

By default the only nodes that carry into the next generation are the
cases infected this generation: they stay active and generate their own
contacts, and an uninfected contact is a dead end. `keep_active` lets an
intervention keep other nodes active. Return the ids of this generation's
targets that should keep generating contacts, and the engine unions them
into the next active set.

The case that needs it is contact tracing to a depth beyond direct
contacts. To reach contacts-of-contacts, the engine has to grow the
contacts of an infected case's contacts even when those in-between nodes
were never infected. Keep them active here, and pair that with the
`InfectiousSource` default so they grow their contacts without becoming a
second wave of infections:

```julia
struct KeepUninfectedActive <: AbstractIntervention end

function EpiBranch.keep_active(::KeepUninfectedActive, state, targets, is_new)
    [t.id for t in targets if !is_infected(t)]
end
```

[`ContactTracing`](@ref) with `depth > 1` is the built-in user of this
hook: it keeps the uninfected ring members active for as many hops as the
ring radius, so a level-2 ring reaches the contacts-of-contacts a ring
vaccination then targets.

### Making the intervention schedulable

Time-based scheduling is provided uniformly by [`Scheduled`](@ref):
wrap any intervention with `Scheduled(iv; start_time = ...)` to delay
its activation. Individual interventions do not carry a `start_time`
field of their own.

For `Scheduled` to perform per-individual reset (the case where the
population gate has opened but a specific individual's sampled action
time would fall pre-policy), the intervention declares two methods:

- **`EpiBranch.intervention_time(intervention, individual)`** — the time
  at which this intervention's effect occurs for the individual (e.g.
  isolation time).
- **`EpiBranch.reset!(intervention, individual)`** — undo the
  intervention's effect on the individual.

Here is how `Isolation` implements these:

```julia
EpiBranch.intervention_time(::Isolation, ind::Individual) = isolation_time(ind)

function EpiBranch.reset!(::Isolation, ind::Individual)
    ind.state[:isolated] = false
    ind.state[:isolation_time] = Inf
    return nothing
end
```

Then a user schedules the intervention like:

```julia
# Activate border closure on day 10
Scheduled(BorderClosure(0.0, 0.05); start_time = 10.0)
```

### Requiring fields on individuals

If your intervention depends on fields set by an attributes function (e.g.
`:onset_time`), override `EpiBranch.required_fields` to get a clear error
at simulation start:

```julia
EpiBranch.required_fields(::MyIntervention) = [:onset_time, :asymptomatic]
```

### Composing with built-in interventions

Custom interventions compose naturally with the built-in ones. The
engine applies all interventions in order each generation. Building on
the `BorderClosure` above (interpreted here as everyone being in one
region so closure does nothing — illustrative only):

```@example extending
clinical_with_region = compose(
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    (rng, ind) -> (ind.state[:region] = :only),
)
iso = Isolation(onset_to_isolation_delay = Exponential(2.0))
bc = BorderClosure(10.0, 0.05)
model = BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0);
    interventions = [iso, bc], attributes = clinical_with_region)

rng = StableRNG(42)
results = simulate(model, 200; max_cases = 500, rng = rng)
println("Isolation + border closure: $(round(containment_probability(results), digits=3))")
```

## Tree-shaping via the offspring distribution

Some interventions don't filter individual transmissions — they change
*how many* contacts a parent makes. Gathering limits, event-size caps,
and superspreading-event surveillance all fit this pattern. They are
genuinely modifications to the offspring distribution, not per-contact
risks, and EpiBranch handles them by accepting a function-form
offspring distribution to [`BranchingProcess`](@ref).

A hard cap on offspring per parent:

```@example extending
capped_offspring(rng, ind) = min(rand(rng, NegBin(2.5, 0.16)), 5)
model_capped = BranchingProcess(capped_offspring, Exponential(5.0))
```

A state-aware cap that takes effect once the outbreak crosses 20
cases (mirroring what `Scheduled` does for risk-based interventions,
but for a tree-shape change):

```@example extending
function policy_offspring(rng, ind, state)
    n = rand(rng, NegBin(2.5, 0.16))
    return state.cumulative_cases >= 20 ? min(n, 5) : n
end
model_policy = BranchingProcess(policy_offspring, Exponential(5.0))
```

The function form supports either two or three arguments — `(rng, ind)`
when the offspring rule only needs the individual, `(rng, ind, state)`
when it also reads simulation state.

Time-varying R falls out of the same mechanism. If `R(t)` is a
function of (say) the parent's infection time, pass an offspring
distribution that reads `ind.infection_time`:

```@example extending
r_at_time(t) = max(1.0, 3.0 - 2.0 * t / 50.0)
time_varying = (rng, ind) -> rand(rng, Poisson(r_at_time(ind.infection_time)))
model_rt = BranchingProcess(time_varying, Exponential(5.0))
```

Use `ind.infection_time` when R varies with each parent's own
infection timing, or `state.max_infection_time` (via the
three-argument form) when R varies with the population-level outbreak
clock.

## Custom attributes functions

The `attributes` argument to `simulate` is a function `(rng, individual) -> nothing`
that sets fields on each individual when they are created (before any
intervention hooks run). The built-in constructors `clinical_presentation`,
`demographics`, and `transmission_traits` return such functions.

### Writing your own

For fields without a dedicated builder — anything in `ind.state` — write a
plain closure. Below, `:risk_group` is a custom state field, so it needs
the closure form; `susceptibility` is derived from it via
`transmission_traits`, which accepts a function:

```@example extending
risk_group = (rng, ind) -> (ind.state[:risk_group] = rand(rng) < 0.2 ? :high : :low)

attrs = compose(
    risk_group,
    transmission_traits(
        susceptibility = (rng, ind) -> ind.state[:risk_group] == :high ? 0.8 : 0.3,
    ),
)

model = BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0); attributes = attrs)
rng = StableRNG(42)
state = simulate(model; max_cases = 100, rng = rng)
n_high = count(ind -> get(ind.state, :risk_group, :low) == :high, state.individuals)
println("High-risk individuals: $n_high / $(length(state.individuals))")
```

### Composing attributes functions

`compose` calls its arguments in order, so later builders or closures can
read fields set by earlier ones:

```@example extending
combined = compose(
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Normal(40, 15)),
    risk_group,
    transmission_traits(
        susceptibility = (rng, ind) -> ind.state[:risk_group] == :high ? 0.8 : 0.3,
    ),
)

model_combined = BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0);
    attributes = combined)

rng = StableRNG(42)
state = simulate(model_combined; max_cases = 100, rng = rng)
ind = state.individuals[1]
println("Individual 1: age=$(ind.state[:age]), sex=$(ind.state[:sex]), risk=$(ind.state[:risk_group])")
```

## Generation time as a function of the individual

`generation_time` can be a `Distribution` shared by everyone, or a
function. When it is a function, the engine calls it with each infected
individual and uses the `Distribution` it returns, so the generation
time can read anything the individual carries in `individual.state`.

The common case is linking it to the individual's own incubation
period, read with [`incubation_period`](@ref):

```@example extending
gt = ind -> Gamma(2.0, incubation_period(ind) / 2)
linked = BranchingProcess(NegBin(2.5, 0.16), gt;
    attributes = clinical_presentation(incubation_period = LogNormal(1.5, 0.5)))

rng = StableRNG(42)
state = simulate(linked; max_cases = 500, rng = rng)
println("Cases: $(state.cumulative_cases)")
```

Because the incubation period is drawn once per individual and the
generation time is built from it, the two are correlated: a later-onset
case also tends to transmit later. [`incubation_linked_generation_time`](@ref)
is a ready-made version (the skew-normal model from Hellewell et al.
2020).

The function sees the whole individual, so the generation time can
depend on any quantity an attributes function has stored, not only the
incubation period. Store a per-individual value and read it back:

```@example extending
# Attributes function draws a per-individual infectiousness scale and
# sets the onset time from the same draw.
host = function (rng, ind)
    scale = 3.0 + rand(rng)
    ind.state[:gt_scale] = scale
    ind.state[:onset_time] = ind.infection_time + scale
end

scaled = BranchingProcess(Poisson(2.0), ind -> Exponential(ind.state[:gt_scale]); attributes = host)

rng = StableRNG(42)
state = simulate(scaled; max_cases = 500, rng = rng)
println("Cases: $(state.cumulative_cases)")
```

This is the seam to use whenever generation time and onset should come
from one per-individual draw instead of two independent ones.

## Custom offspring distributions

`BranchingProcess` takes anything its sampling path can use. Two paths
are supported, with different trade-offs:

- A **`Distribution` subtype** (a struct `<: Distribution` from
  Distributions.jl). Full Distributions.jl interop — `rand`, `logpdf`,
  fitting, mixtures all work — and the analytical helpers
  (`extinction_probability`, `chain_size_distribution`, etc.) compose
  cleanly via [`single_type_offspring`](@ref).
- A **function** `(rng, individual)` or `(rng, individual, state)`
  returning the offspring count. Escape hatch for state-dependent
  rules that don't fit a single fixed Distribution (time-varying R,
  policy-dependent caps, etc.).

### `Distribution` subtype

For an offspring rule that's a proper probability distribution,
subtype `Distribution` and implement `Distributions.rand`. The
branching process picks it up via the standard constructor:

```julia
using Distributions
using Random: AbstractRNG

struct MyOffspring <: Distribution{Univariate, Discrete}
    # ... your parameters
end

function Distributions.rand(rng::AbstractRNG, d::MyOffspring)
    # ... return an integer offspring count
end

model = BranchingProcess(MyOffspring(...), Exponential(5.0))
```

The simulation loop calls `rand(rng, offspring)`, so any Distribution
subtype with a `rand` method works.

To enable the analytical helpers (`extinction_probability`,
`chain_size_distribution`, `proportion_transmission`), also specialise
`chain_size_distribution` for your type:

```julia
function EpiBranch.chain_size_distribution(d::MyOffspring)
    # return a Distribution over chain sizes — e.g. via numerical
    # iteration of the offspring PGF.
end
```

Simulation works without this specialisation; only the closed-form
analytics require it.

### Function-based offspring

For state-dependent rules that don't fit a single fixed Distribution,
pass a function. It receives `(rng, individual)` or `(rng, individual,
state)` and returns the number of offspring (an `Int` for single-type
models, or a `Vector{Int}` for multi-type).

### Example: context-dependent transmission

Here the reproduction number depends on a custom attribute:

```@example extending
function risk_offspring(rng, individual)
    base_R = individual.state[:risk_group] == :high ? 4.0 : 1.5
    return rand(rng, Poisson(base_R))
end

model_risk = BranchingProcess(risk_offspring, Exponential(5.0); n_types = 1, attributes = risk_group)

rng = StableRNG(42)
results = simulate(model_risk, 200; max_cases = 500, rng = rng)
println("Risk-stratified model: $(round(containment_probability(results), digits=3))")
```

### Example: generation-dependent R

The offspring function can also read the individual's generation to model
waning transmission over the course of an outbreak:

```@example extending
function waning_offspring(rng, individual)
    R = 3.0 * exp(-0.1 * individual.generation)
    return rand(rng, Poisson(R))
end

model_waning = BranchingProcess(waning_offspring, Exponential(5.0); n_types = 1)

rng = StableRNG(42)
results = simulate(model_waning, 200;
    max_cases = 500,
    rng = rng,
)
println("Waning-R model: $(round(containment_probability(results), digits=3))")
```

## Adding a transmission model

Most use cases stay inside `BranchingProcess` and customise via the
offspring distribution (function-based, `ClusterMixed`, multi-type).
But if you need a fundamentally different transmission process (a
density-dependent model, a network-structured one, a continuous-time
SEIR-like alternative), you can subtype `TransmissionModel` directly
and reuse the rest of the framework.

The contract is small. You implement what your model needs and reuse
defaults for the rest.

### What the framework expects

For **simulation** there are two paths, depending on whether your model
can produce its candidates one parent at a time.

An **offspring-driven** model (a branching process and its variants)
defines one method:

- [`generate_offspring`](@ref)`(model, parent, state)` — return how many
  contacts `parent` makes this generation: a single count, or a count
  per type for a multi-type model. The default
  `simulate(::TransmissionModel)` loop calls it once per active parent,
  creates that many candidate contacts, gives each an infection time
  from your model's `generation_time`, and resolves competing risks.

`generate_offspring` returns a count and nothing else: it assigns no
timing, builds no `Individual`s, and takes no `interventions` argument.
Return the number of *potential* contacts, and don't pre-filter by
parent intervention state (`:isolated`, `:vaccinated`, …). The engine's
competing-risks resolution decides afterwards which contacts are
infected, and that is the only place intervention effects on
transmission apply. The engine also runs `resolve_individual!` on each
parent first, then `initialise_individual!` and
`apply_post_transmission!` on the new contacts, the clinical
transitions, and the bookkeeping fields (`cumulative_cases`,
`current_generation`, `active_ids`, `extinct`, `max_infection_time`).

A **structure-driven** model produces candidates a count can't name: a
contact network, or a household/metapopulation process where a
susceptible can be reached by several infectious sources at once and
infections deplete a fixed pool. It defines two methods instead:

- [`contacts_of`](@ref)`(model, node, state)` — the contacts an
  infectious `node` reaches this generation, as `(contact, infection_time)`
  pairs. Return existing nodes (a network), or mint fresh ones with
  [`make_contact!`](@ref). Do not set `:infected` yourself.
- override [`collect_exposures`](@ref) with [`gather_by_target`](@ref),
  so a node reached by several infectious neighbours in one generation
  collects all its incoming edges and is resolved once.

`contacts_of` has no `interventions` argument either, and the same rule
applies: produce every *potential* contact and let the engine's
competing-risks resolution decide infection. If the model's own
transmission probability belongs to the *edge* (a network's per-edge
probability, a metapopulation coupling), don't filter on it in
`contacts_of` — return the contact and let the probability decide infection
by overriding
[`transmission_risks`](@ref EpiBranch.transmission_risks)`(model)` to return
a risk source with a `competing_risk` method. The contact is then still
produced and seen by `apply_post_transmission!` (so contact tracing and ring
vaccination work), and the probability is weighed against susceptibility,
infectiousness and interventions together. Everything else — gathering
the exposures, `initialise_individual!` and `apply_post_transmission!` on
new contacts, competing risks, clinical transitions, and bookkeeping — is
the shared engine. A structure-driven model also defines
[`initialise_state`](@ref EpiBranch.initialise_state) to set up its fixed
population, building it with the public helpers
[`new_state`](@ref EpiBranch.new_state),
[`add_individuals!`](@ref EpiBranch.add_individuals!) and
[`seed!`](@ref EpiBranch.seed!). The companion EpiNetwork.jl package's
`NetworkProcess` is the worked example.

Models whose contacts can be *shared* across parents within a generation
(networks, households, clustering) also override
[`collect_exposures`](@ref) with [`gather_by_target`](@ref), which
deduplicates shared targets so a node reached several times resolves once.

For **analytical inference helpers** that route through the offspring
specification (`extinction_probability`, `epidemic_probability`,
`probability_contain`, `proportion_transmission`,
`chain_size_distribution`), define one method:

- [`single_type_offspring`](@ref)`(model)` returning the offspring
  distribution (or any object for which `chain_size_distribution` is
  defined). Specialise this and you get the analytical helpers for
  free.

For **likelihoods** on data types that don't go through the offspring
spec, define methods on `loglikelihood` directly.

For optional **state accessors**, override `population_size` and
`n_types` if your model has values for them. The defaults
(`NoPopulation()`, `1`) are fine if not.

If your model carries its own interventions, attributes or observation in
fields, define the matching accessors — `EpiBranch.interventions(m)`,
`attributes(m)`, `observation(m)` — so the engine and the likelihood can read
them. `BranchingProcess` defines these over its fields; a new model opts in the
same way (the defaults are no interventions, no attributes, no observation). And
if your generation-time distribution is not stored in a field literally named
`generation_time`, override [`model_generation_time`](@ref EpiBranch.model_generation_time)`(m)` to point at it —
that accessor is what the engine calls.

If your model carries a clinical natural history (incubation, onset,
recovery), expose it as the model's progression: store the transitions in
a field and define `EpiBranch._progression(m::MyModel) = m.progression`.
The engine reads progression off the model and applies the transitions to
each new contact. `BranchingProcess` does this for the `progression`
keyword; a custom model opts in the same way.

### Minimal sketch

A skeleton for a custom transmission model:

```julia
struct MyModel{O, G} <: TransmissionModel
    offspring::O
    generation_time::G
    # ... your model parameters
end

# Required for simulation: how many contacts this parent makes. The
# engine creates them, assigns each a generation time, and handles
# `:infected`, post-transmission hooks, transitions, and bookkeeping.
EpiBranch.generate_offspring(model::MyModel, parent, state) =
    rand(state.rng, model.offspring)

# Required for analytical helpers (optional but recommended).
EpiBranch.single_type_offspring(m::MyModel) = m.offspring

# Optional accessors, with defaults if unset.
EpiBranch.population_size(m::MyModel) = NoPopulation()
EpiBranch.n_types(m::MyModel) = 1
```

If `single_type_offspring(m)` returns a NegBin or a `ClusterMixed` or
anything else with a `chain_size_distribution` method, the analytical
chain-size likelihood works automatically:

```julia
loglikelihood(ChainSizes(data), MyModel(NegBin(0.8, 0.5), ...))
```

### Composing with the observation side

Your model carries an observation like any other process: store it in a field,
define `EpiBranch.observation(m) = m.observation` (see the accessors above), and
the likelihood reads it automatically:

```julia
MyModel(...; observation = PerCaseObservation(detection_prob = 0.7))
```

works the same way as it does for `BranchingProcess`, which defines that
accessor for you.

## Adding an observation model

Observation models attach to the process the same way interventions do.
They subtype `ObservationModel` and join in through
two methods dispatched on the observation type, with no model type
parameter:

1. A struct holding the observation parameters, subtyping `ObservationModel`.
2. [`observe`](@ref)`(base_distribution, ::YourObservation)` — the analytical side: return a `Distribution` transforming the latent chain-size distribution. Often a small new `DiscreteUnivariateDistribution`.
3. `apply_observation!(::YourObservation, state, rng)` — the simulation side: mark observed cases on a finished `SimulationState` (only needed for the simulation-based likelihood).

### Minimal sketch

```julia
# 1. Observation model
struct CensoredAtSize <: ObservationModel
    cap::Int
end

# 2. Transformed chain size distribution
struct TruncatedChainSize{D} <: DiscreteUnivariateDistribution
    base::D
    cap::Int
end
Distributions.minimum(::TruncatedChainSize) = 1
Distributions.maximum(d::TruncatedChainSize) = d.cap
Distributions.insupport(d::TruncatedChainSize, n::Integer) = 1 <= n <= d.cap

function Distributions.logpdf(d::TruncatedChainSize, n::Integer)
    1 <= n <= d.cap || return -Inf
    Z = sum(pdf(d.base, m) for m in 1:d.cap)
    return logpdf(d.base, n) - log(Z)
end

# 3. The analytical side of the protocol: one method, dispatched on the
#    observation. loglikelihood(data, model) routes through it.
EpiBranch.observe(base, o::CensoredAtSize) = TruncatedChainSize(base, o.cap)
```

Usage: `BranchingProcess(...; observation = CensoredAtSize(10))`. No
per-observation `loglikelihood` method is needed — returning a distribution
from `observe` means the shared machinery evaluates `logpdf` on it.

### Sim ↔ analytical consistency test

The helper in `test/testutils/sim_analytical_consistency.jl` cross-checks
simulation against your new distribution. It reads the model's observation
and thins the simulated true sizes accordingly; add a method for your
observation type to its `_observe_sizes` dispatch:

```julia
# Transform simulated true sizes into observed ones
_observe_sizes(o::CensoredAtSize, true_sizes, ::AbstractRNG) =
    filter(n -> n <= o.cap, true_sizes)
```

With that in place,
`sim_analytical_consistent(model; n_chains=5000, rng=StableRNG(1))`
returns empirical and analytical PMFs that should agree within
sampling error.

## Adding an offspring specification

Offspring specifications replace what `BranchingProcess` draws per
individual. `ClusterMixed(build, mixing)` (per-chain parameter
variation) is the reference. A new offspring type needs:

1. Simulation dispatch: `draw_offspring(rng, offspring, individual, state)` returning the number of offspring.
2. Analytical dispatch (optional but recommended): `chain_size_distribution(offspring)` returning the analytical PMF. Without it, the likelihood falls back to simulation.
3. A `BranchingProcess` constructor so the type can be stored in the `offspring` field.

See `src/analytical/cluster_mixed.jl` for the full pattern, including how `ClusterMixed` caches per-chain state on the index case and has descendants inherit it through `parent_id`.

## Adding per-observation metadata

[`ChainSizes`](@ref) already supports two per-observation fields: `seeds`
(multi-seed clusters) and `concluded` (right-censored ongoing clusters).
Both are decisions made by the analyst, not properties the framework
derives — and they show the pattern for any new field.

If your analysis needs different or richer per-cluster information, you
have two options.

### Stay in `ChainSizes` and pre-compute

If the new information resolves to a flag or a count that the existing
likelihood already handles, derive it upstream and pass it in. The Endo
7-day time-censoring rule is an example: it looks like time censoring
but is just a way to compute `concluded`.

```julia
using Dates
is_ongoing(latest_case, cutoff; window_days = 7) =
    cutoff - latest_case < Day(window_days)

data = ChainSizes(sizes;
    seeds = imports_per_cluster,
    concluded = .!is_ongoing.(last_case_dates, cutoff_date))
```

No new types or methods needed — the decision rule lives wherever it
belongs in the analysis.

### Define a new data type when the likelihood needs new information

If the likelihood itself needs to use new per-observation data (not just
collapse it into an existing flag), define a new struct and a
`loglikelihood` method.

```julia
struct MultiTypeChainSizes
    data::Vector{Int}
    type::Vector{Int}   # which strain/patch/group
end

# Different offspring distribution per type; pick by observation.
function Distributions.loglikelihood(data::MultiTypeChainSizes,
        offsprings::Vector{<:Distribution})
    total = 0.0
    for i in eachindex(data.data)
        d = chain_size_distribution(offsprings[data.type[i]])
        total += logpdf(d, data.data[i])
    end
    return total
end
```

The internal `EpiBranch._chain_size_logpdf(d, x, s)` is the reusable
piece — call it from your method if you need multi-seed support, and
your new data type inherits the same closed forms for `Borel`,
`GammaBorel`, `PoissonGammaChainSize` as the built-in `ChainSizes` uses.

## Summary of extension points

| Extension point | Mechanism | When called |
|---|---|---|
| Custom intervention | Struct `<: AbstractIntervention` + hook methods | Each generation |
| Time-dependent intervention | `Scheduled(iv; start_time = ...)` + `intervention_time`, `reset!` on `iv` | After each hook |
| Custom attributes | Function `(rng, ind) -> nothing` | Individual creation |
| Composed attributes | `compose(f1, f2, ...)` | Individual creation |
| Custom offspring (function) | Function `(rng, ind) -> Int` | Offspring draw |
| Multi-type offspring | Function `(rng, ind) -> Vector{Int}` | Offspring draw |
| Custom offspring (type) | Struct + `draw_offspring`, `chain_size_distribution` | Offspring draw + analytics |
| Custom transmission model | Struct `<: TransmissionModel` + `generate_offspring` (offspring-driven) or `initialise_state` + `contacts_of` + `gather_by_target` (structure-driven); optional `single_type_offspring`, accessors | Simulation + analytics |
| Custom observation model | Struct `<: ObservationModel` + `observe(base, ::YourObs)` (analytics) and/or `apply_observation!(::YourObs, state, rng)` (simulation) | Analytics / inference |
| Per-observation metadata | Either pre-compute into existing `ChainSizes` fields, or define a new data type with a `loglikelihood` method that calls `_chain_size_logpdf` | Likelihood evaluation |
| Sim ↔ analytical test | `generative_model`, `observe_chain_sizes` | Regression test |
