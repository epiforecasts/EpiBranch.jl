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
- **Add a transmission route** — give a process a `RouteWindow` so a case
  transmits over several routes at once, each opening and closing on different
  states of its natural history. Covered below.
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

Read a key through a one-line accessor that supplies a safe default. For a
real-valued timing key, keep the accessor element-type generic so a gradient
can flow through it under automatic differentiation:
`onset_time(ind::Individual{T}) where {T} = convert(T, get(ind.state, :onset_time, T(NaN)))`.
A Boolean, integer, or symbol key can pin a concrete type instead
(`is_isolated(ind) = get(ind.state, :isolated, false)::Bool`). New code should
add an accessor in `src/state_accessors.jl` rather than calling
`get(ind.state, …)` directly.

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
| `:group` | `Int` | — | `groups` | Init |
| `:isolated` | `Bool` | `false` | `Isolation` | `resolve_individual!` |
| `:isolation_time` | `Float64` | `Inf` | `Isolation` | `resolve_individual!` |
| `:isolated_by_isolation` | `Bool` | `false` | `Isolation` | `resolve_individual!` |
| `:test_positive` | `Bool` | `false` | `Isolation` | `resolve_individual!` |
| `:traced` | `Bool` | `false` | `ContactTracing` | `apply_post_transmission!` / `trace_contacts!` |
| `:quarantined` | `Bool` | `false` | `ContactTracing` | `apply_post_transmission!` / `trace_contacts!` |
| `:traced_isolation_time` | `Float64` | `Inf` | `ContactTracing` → `Isolation` | Internal handoff; may precede onset, so hold it back to onset |
| `:trace_time` | `Float64` | — | `ContactTracing` | `apply_post_transmission!` / `trace_contacts!` |
| `:ring_remaining` | `Int` | `0` | `ContactTracing` (`depth > 1`) | `apply_post_transmission!` / `trace_contacts!` |
| `:traced_by` | `Int` | — | `ContactTracing` | `apply_post_transmission!` / `trace_contacts!` |
| `:trace_level` | `Int` | — | `compute_trace_level!` | Post-simulation |
| `:vaccinated[_<label>]` | `Bool` | `false` | `AbstractVaccination` | Init / `apply_post_transmission!` |
| `:vaccination_time[_<label>]` | `Float64` | `Inf` | `AbstractVaccination` | `apply_post_transmission!` |
| `:vaccine_efficacy[_<label>]` | `Float64` | — | `AbstractVaccination` | `apply_post_transmission!` |
| `:post_exposure_efficacy[_<label>]` | `Float64` | — | `RingVaccination` (varying `post_exposure_efficacy`) | `apply_post_transmission!` |
| `:onward_efficacy[_<label>]` | `Float64` | — | `RingVaccination` (varying `onward_efficacy`) | `apply_post_transmission!` |
| `:immunity_time[_<label>]` | `Float64` | — | `AbstractVaccination` | `apply_post_transmission!` |
| `:severity_efficacy[_<label>]` | `Float64` | — | `AbstractVaccination` | `apply_post_transmission!` |
| `:coverage_declined[_<label>]` | `Bool` | `false` | `GroupVaccination` | `apply_post_transmission!` |
| `:infection_aborted_time` | `Float64` | — | `RingVaccination` (`post_exposure_efficacy`) | `apply_post_transmission!` |
| `:capacity_admission_time_<capacity_key>` | `Float64` | — | `CapacityConstrained` | `apply_post_transmission!` |
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
| `:vaccine_acceptance` | `Float64` | — | `vaccine_acceptance` | Init (default key; customisable) |
| `:infectious_time` | `Float64` | `Inf` | `Transition(:infectious, …)` | `resolve_individual!` |
| `:recovered_time` | `Float64` | `Inf` | `Transition(:recovered, …)` | `resolve_individual!` |

The vaccination keys are namespaced by `dose_label`: the default label
writes to plain `:vaccinated` / `:vaccination_time` / `:vaccine_efficacy` /
`:immunity_time` (and, on `RingVaccination`, `:post_exposure_efficacy` /
`:onward_efficacy`), and any other label suffixes the key (so
`dose_label = :boost` writes `:vaccinated_boost`, etc.). This lets multi-dose
schedules compose without colliding. `:immunity_time` (the vaccination time
plus a draw from `delay_to_immunity`), `:post_exposure_efficacy`, and
`:onward_efficacy` each hold one draw taken at vaccination time from a field
that may be a `Real`, a `Distribution`, or a function, so every exposure of an
individual is judged against the same value. `:post_exposure_efficacy` and
`:onward_efficacy` are written only when the field is a distribution or a
function; a scalar is the same for everyone and is read straight off the
intervention.

`:immunity_time` (`:vaccination_time` plus the dose's `delay_to_immunity`)
and `:severity_efficacy` let a clinical transition read a vaccine's effect
on disease severity — mortality, or any other outcome a `progression`
transition decides — without gating transmission. Neither participates in
`competing_risk`; a transition's `probability` reads them through the
[`immunity_time`](@ref) and [`severity_efficacy`](@ref) accessors, gating
on the former so a dose whose immunity has not yet developed by the
outcome it would affect confers no protection.

`:infection_aborted_time` marks an infection that a post-exposure dose ended
before symptom onset. The individual is still infected but transmits nothing
from that time, and the engine applies this block for as long as the key is
present. It has no onset: `:onset_time` is `NaN` while `:asymptomatic` stays
`false`, so isolation, tracing and clinical transitions triggered by onset never
happen.

Its clinical course ends at the abort time. When transitions are resolved, any
transition that would take effect at or after that time, whatever its `from`,
is undone and the keys it wrote are restored, so no hospitalisation, death or
`:outcome` follows. Transitions that take effect earlier stand. The check reads
the `_time` keys a transition writes, so it covers a custom transition that
records when it happens under a `_time` key, as the built-ins do.

The key is drawn against a particular exposure, before infection is resolved:
when the dose is given, and again each time a contact that already has the dose
is exposed, using its recorded vaccination time. The engine removes the key and
restores the onset when resolution does not confirm that exposure, which happens
on a contact the exposure did not infect and on one infected through a later
exposure at or after the abort time. The key is therefore only present on an
infected individual whose infection it ended, and a pre-created node that
escapes one exposure gets a fresh draw against the exposure that later infects
it.

`:reported` is shared between the `Reporting` clinical transition (which
sets it from a probability gate) and `PerCaseObservation` (which sets it
post-simulation from a detection-probability draw). Composing both in the
same simulation is not supported, because they will overwrite each other.

Isolation is recorded under `:isolation_time`. A window that isolation should
end lists [`EpiBranch.INTERVENTION_REMOVAL`](@ref) in its `until` (see
[Transmission routes](#Transmission-routes)), which respects leaky isolation.
`:isolated` in an `until` refers to a `Transition(:isolated, …)` in the natural
history. Set and undo isolation with `set_isolated!` and `clear_isolated!`.

The tracing keys name two hooks because the two engines reach them
differently: `apply_post_transmission!` on the generation-based engine, and
`trace_contacts!` on the continuous-time models. Both funnel through the same
per-pair policy, so the keys and their meanings are identical either way; see
[Which hooks fire on which engine](#which-hooks-fire-on-which-engine).

`:traced_by` is the source a node was traced from — the *first*,
earliest-exposure tracer, since the engine makes one trace attempt per node.
`compute_trace_level!` walks it back to the index case post-simulation to set
`:trace_level` (distance from the index, anchor `0`). On a tree the level is
exact; on a cyclic `NetworkProcess` it is the depth along the first-traced
path, **not** a guaranteed shortest distance to the nearest index — do not
read it as one.

Built-in keys use short bare names like `:isolated`, `:traced`, `:age`, and
those names are reserved. If you add keys from another package, prefix them
with a short tag for your package so they do not collide with built-ins or
with keys other packages might add.

State times follow a convention. A generic `Transition(:state; …)` writes the
flag `:state` and the time `:state_time` (that is, `Symbol(state, :_time)`).
Infectiousness windows read the same convention: `from = :infectious` reads
`:infectious_time`, and `until = (:recovered,)` reads `:recovered_time`. So the
`_time` keys your transitions produce are the names your windows refer to, and
they need to match. `:infectious_time` and `:recovered_time` are the common
natural-history pair; any other state you transition into produces its own
`<state>_time` the same way.

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
| `competing_risk(iv, parent, contact, state)` | Per `(parent, contact)` pair: on the generation engine during infection resolution, after `apply_post_transmission!` has run; on the continuous-time models as each infection is proposed | The parent and a single contact | `nothing`, a single [`Risk`](@ref), or an `NTuple{N, Risk}` for interventions that gate transmission via more than one mechanism |
| `keep_active(iv, state, targets, is_new)` | Once per generation after infection is resolved, while the engine builds the next active set | This generation's `targets` and an `is_new` flag per target | An iterable of contact ids to keep generating contacts into the next generation (default: none) |
| `trace_contacts!(iv, state, infector, contacts[, not_before])` | Continuous-time models only: once per case, when the race settles it | The case, the contacts it reached that are not yet settled, and, from a model whose contacts can come about after the case's infection, when each became a contact (the four-argument method is called when the model gives no times, and by default for interventions that ignore them) | `nothing` (mutate the contacts' `state` in place) |
| `traces_contacts(iv)` | Whenever a continuous-time model decides whether to gather contacts at all | Nothing | `true` if this intervention implements `trace_contacts!` (default `false`) |
| `infectious_removal_time(iv, individual)` | Continuous-time models only: when a case's infectious window is closed | An individual | The time this intervention takes it out of onward transmission (default `Inf`) |
| `risk_applies(iv, route)` | Continuous-time models selecting risks for a route (`nothing` for an external introduction) | Nothing | `Bool`; defaults to `true` |

### Which hooks fire on which engine

The hooks above are not all available everywhere, because the engines are
built differently. The generation-based engine creates a fresh `Individual`
for every contact, infected or not, so it can hand you a batch of contact
objects. The continuous-time (Sellke) models have no such objects: every node
exists from the start and the simulation only settles *when* each is infected,
by a race between contact-interval draws. What they do have is the potential
infection itself — a drawn time for a named pair — so a `Risk` has somewhere to
hang after all, alongside the infectious window.

| Hook | Generation engine | Network / household (Sellke race) | Homogeneous pool |
|---|---|---|---|
| `initialise_individual!` | yes | yes | yes |
| `resolve_individual!` | yes | yes | yes |
| `competing_risk` | yes | yes | yes |
| `infectious_removal_time` | not read | yes | yes |
| `trace_contacts!` | not called | yes | no contact set |
| `apply_post_transmission!` | yes | not called | not called |
| `keep_active` | yes | not called | not called |

What this means in practice:

- An intervention whose effect is a **removal** — isolation, quarantine on
  being traced, hospitalisation — works everywhere. It shortens the
  infectious window, which every engine has.
- An intervention whose effect is a **per-contact competing risk** against
  the infection event — leaky vaccination, a partial-efficacy prophylaxis —
  works everywhere too, and so do per-individual susceptibility and
  infectiousness, which ride the same surface. The continuous-time models put
  each potential infection to the composed risks at the moment they propose it,
  against the time they propose it for. A blocked contact does not transmit and
  the contact process carries on: on a graph the pair's next contact is drawn
  from its own hazard conditioned on falling later, and in the mass-action pool
  the susceptible waits for the next contact with a fresh resistance. Blocking
  each contact with probability `p` therefore thins the force of infection to
  `(1 - p)` of it on both, so the two agree — a two-person clique with a
  one-day mean contact interval and a two-day infectious period is the same
  process as a pool of two at `β = 2`, and at efficacy 0.5 both infect
  `1 - exp(-1) = 0.63` of the time.
- Per-individual susceptibility and infectiousness reach the same thinning by a
  shorter route. They are constants of the two people rather than something that
  arrives at a time, so the models fold them into the draw: a multiplier `m`
  turns a pair's contact-interval survival `S(t)` into `S(t)^m`, the pool scales
  each susceptible's threshold and weights each infective's share of the force,
  and a community introduction's hazard is scaled the same way. A multiplier of
  0 never transmits and draws nothing. Static proportional effects can use these
  traits or an effective kernel directly, avoiding repeated rejected contacts.
- Repeated-contact sampling after a blocked proposal requires finite remaining
  integrated hazard. A race rejects a continuation if the kernel's survival is
  zero at the end of its window, including an unbounded Exponential window or a
  continuous bounded kernel whose support ends inside the window. A pool
  requires finite removal times for all active sources when a contact is
  blocked. These cases raise `ArgumentError`, even for a risk that might later
  permit infection: an opaque callback cannot establish eventual termination.
  Supply a finite infectious/introduction window with nonzero kernel survival
  at its end, or represent static protection through host traits or the kernel.
  A `Dirac` kernel has no contact after its atom and needs no continuation.
  Numerical accuracy still depends on the kernel's survival implementation:
  one computed as `1 - cdf` loses tail precision when the CDF rounds to one.
- That is the per-exposure reading of a leaky vaccine, and it is **not** what
  the same `Risk`
  does on the generation engine. There a parent's contacts are a fixed set of
  draws, so a blocked one is a transmission lost with nothing to follow it, and
  an efficacy of 0.5 halves that pair's transmissions. The same efficacy bites
  less per pair on a continuous-time model, because the pair goes on meeting
  (0.63 above, against 0.43 for a halved probability).
- Thinning the hazard leaves the pair's contact process in the family the
  pairwise likelihood works with, its hazard scaled: a constant susceptibility
  is still representable wherever that family is closed under proportional
  hazards, as an exponential contact interval is. A risk that arrives partway
  through the window — an isolation, or a dose a trace gives — is not, so
  simulating with those and scoring the result with `loglikelihood` will
  disagree.
- A community introduction, on a model with an `external_hazard`, is put to the
  risks that act on the person being introduced: their susceptibility, a
  vaccine's protection, a risk of your own. Isolation and quarantine do not
  remove an external source. An introduction has no infector record, so the
  person stands in for one; return `nothing` from your risk when
  `parent === contact` if it reads properties a community source cannot have.
- [`EpiBranch.risk_applies`](@ref) selects which routes an intervention's risk
  acts on. It receives the existing route window, or `nothing` for a community
  introduction. Its default `true` keeps vaccination protection on every route.
  Isolation and contact tracing test whether the route lists
  `EpiBranch.INTERVENTION_REMOVAL` in `until`. Wrappers forward the predicate.
  The generation engine applies every risk to every contact; model-provided
  risks and host multipliers also apply on every route.
- An external intervention can choose any subset of routes without adding a
  scope type. For example, a removal effect can follow the window's censoring:

  ```julia
  EpiBranch.risk_applies(::MyLeakyQuarantine, route) =
      route !== nothing && EpiBranch.INTERVENTION_REMOVAL in route.until
  ```
- An intervention that reaches its targets through `apply_post_transmission!`
  or `keep_active` — `MassVaccination`'s rollout doses each new contact as the
  engine creates it — has nothing to act on when no contacts are created. You
  need not declare this: when your type has a method of its own for either hook,
  the continuous-time models name it in their warning. The exception is an
  intervention that also traces contacts (`traces_contacts` returns `true`),
  whose `trace_contacts!` is taken as the continuous-time counterpart of those
  hooks; it is honoured on a model that can name a case's contacts and reported
  on one that cannot, such as the mass-action pool.
- **Contact tracing** spans the two. Its action is a removal, so it applies
  on both, but it needs to know who a case's contacts were. The generation
  engine reads that off each contact's `parent_id`; the continuous-time
  models get it from the process, which must report
  `EpiBranch.supplies_contacts(model) = true` and pass a `contacts` closure.
  The closure yields contact ids, or `(id, time)` pairs when some contacts come
  about only after the case's infection, such as at a funeral
  (`contacts_from = :died` on a `RoutedNetwork` route). `time` is when each
  person became a contact, and `ContactTracing` runs that contact's trace delay
  from no earlier than it.
  A graph names a node's neighbours and a household its members; the
  homogeneous pool is mass-action and has no pairwise contact structure, so
  tracing stays unhonoured there.

Two ordering differences follow from this, and they matter when you write an
intervention that has to work on both:

- On the generation engine, a contact resolves its own state
  (`resolve_individual!`) **before** tracing runs. On the continuous-time
  models the order inverts: a contact is traced when its *infector* settles,
  which is before the contact settles anything of its own. An intervention
  that writes onto a contact must therefore not assume the contact is
  unwritten, and one that reads a contact's own state must not assume it is
  already set. This is why `Isolation` treats a standing quarantine as a
  competing pathway and keeps the earliest time, instead of returning early.
- Tracing on the continuous-time path reaches only contacts that have not
  themselves settled. Settling a case fixes its window and its onward
  proposals, so a trace arriving afterwards has nothing left to shorten.
  Tracing *backwards*, to the already-settled neighbour a case was infected
  by, is not supported on either engine.

Ordering guarantees:

- `resolve_individual!` runs strictly before any `competing_risk` call for that generation, so a competing risk can read whatever `resolve_individual!` wrote on the parent.
- `apply_post_transmission!` runs strictly before any `competing_risk` call, so a competing risk can read whatever post-transmission hook wrote on the contact (e.g. `:vaccination_time`).
- `keep_active` runs after infection is resolved, so it can read each target's `:infected` and anything `apply_post_transmission!` wrote on it this generation.
- Interventions are applied in the order they appear in `interventions = [...]`. For `apply_post_transmission!` and `competing_risk`, every intervention sees the state written by earlier interventions in the same generation.
- On the continuous-time models the counterpart holds through tracing: a case is traced when it settles, before it proposes any infection of its own, so a risk can read what `trace_contacts!` wrote on a contact. Built-in vaccination delivery still uses the generation engine’s post-transmission hook and is reported as unsupported on the continuous-time path.

A `Risk` applies to a contact when `event_time <= contact.infection_time`; in that case transmission is blocked with probability `block_probability`. On the continuous-time models the transmission time it is compared against is the candidate infection time the race has just drawn for that pair. Returning multiple risks (as a tuple) lets one intervention gate transmission through several mechanisms: `RingVaccination` returns a susceptibility risk on the contact alongside a risk on the parent for reduced onward infectiousness.

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

    # A contact traced before its onset was known has only the bare trace
    # time, so hold it back to the onset.
    traced_time = max(get(individual.state, :traced_isolation_time, Inf), onset_time(individual))
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

Four defaults ship, each contributing a block probability:

- [`EpiBranch.HostSusceptibility`](@ref) — `1 - susceptibility` on the contact.
- [`EpiBranch.InfectorInfectiousness`](@ref) — `1 - infectiousness` on the parent.
- [`EpiBranch.InfectiousSource`](@ref) — a full block when the source is
  not infected, so an uninfected node can stay active (see below) and
  generate contacts without infecting them. A no-op in the usual case
  where every active node is infected.
- [`EpiBranch.AbortedInfection`](@ref) blocks every transmission an
  infector makes from its `:infection_aborted_time`, so an infection
  aborted by a post-exposure dose stays ended whether or not the
  intervention that aborted it is still active.

A trait of `1.0` contributes no risk, so the defaults are silent unless
an attributes function sets a susceptibility or infectiousness below one.
You can replace or extend them by adding your own `competing_risk` the
same way.

`HostSusceptibility` and `InfectorInfectiousness` are the generation engine's
sources for the two traits. The continuous-time models carry the same two as
multipliers on the transmission hazard instead (see above), so they do not
resolve them contact by contact; everything else on this surface, yours
included, is resolved there as it is here.

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

### Making the intervention capacity-constrained

[`CapacityConstrained`](@ref) rations `apply_post_transmission!` — the one
hook the engine calls with a whole generation's contacts at once, so it is
the only point where several individuals compete for a shared, finite
resource in the same call. To let a custom intervention be wrapped this
way, define:

- **`EpiBranch.capacity_key(intervention)`** — the `Individual.state` flag
  that records the resource having been used (a dose flag, a "traced"
  flag, …).
- **`EpiBranch.capacity_time_key(intervention)`** — the key recording *when*
  it was used, needed only if the intervention is ever wrapped with
  `carry_over = false`. There it places usage `CapacityConstrained` did not
  itself admit, such as a dose from another intervention writing the same
  `capacity_key`, in a period; usage the wrapper admitted is placed by the
  time of the call that admitted it.

`RingVaccination` and `MassVaccination` implement these with their
dose-recording keys:

```julia
EpiBranch.capacity_key(v::RingVaccination) = _vaccinated_key(dose_label(v))
EpiBranch.capacity_time_key(v::RingVaccination) = _vaccination_time_key(dose_label(v))
```

This only rations an intervention whose effect is actually recorded inside
`apply_post_transmission!` on the contacts it is handed. `GroupVaccination`
is the counter-example: it reaches a triggered group by scanning the whole
population, not the batch this hook receives, so limiting that batch would
not limit the doses given, and it does not define `capacity_key`.

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
clinical_with_region = [
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    (rng, ind) -> (ind.state[:region] = :only),
]
iso = Isolation(onset_to_isolation_delay = Exponential(2.0))
bc = BorderClosure(10.0, 0.05)
model = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0));
    interventions = [iso, bc], attributes = clinical_with_region)

rng = StableRNG(42)
results = simulate(model, 200; max_cases = 500, rng = rng)
println("Isolation + border closure: $(round(containment_probability(results), digits=3))")
```

### A custom vaccination

A new vaccination differs from the built-in ones in who it reaches and when.
The parameters describing what a dose does once given (`efficacy`,
`severity_efficacy`, `delay_to_immunity`, `mode` and `dose_label`) live in a
[`VaccineEffect`](@ref), which every [`AbstractVaccination`](@ref) holds. A
subtype stores one and returns it from `EpiBranch.vaccine_effect`; the rest of
the vaccination machinery reads these parameters only through that method. The
subtype then inherits:

- `initialise_individual!`, which sets `:vaccinated` and `:vaccination_time`
  (namespaced by `dose_label`) on every individual;
- `competing_risk`, the susceptibility-side block described in
  [`AbstractVaccination`](@ref);
- the dose-schedule checks made when a `ModelSpec` is built, so it can give the
  dose a later [`RingVaccination`](@ref) names in `requires_dose`.

It adds an `apply_post_transmission!` method choosing whom to vaccinate and
when. That method records each dose with `EpiBranch._record_vaccination!(v, ind,
vaccination_time, rng)`, which writes the per-dose keys listed under
[Reserved keys](#Reserved-keys) and draws `efficacy`, `severity_efficacy` and
`delay_to_immunity` for that individual, whichever of the `Real`,
`Distribution` and function forms they were given in. Here, a campaign on day
10 reaches everyone aged 60 or over:

```@example extending
struct OlderAdultVaccination{V <: VaccineEffect, B} <: AbstractVaccination
    effect::V
    min_age::Int
    campaign_time::Float64
    booster_uptake::B
end

function OlderAdultVaccination(; min_age, campaign_time, booster_uptake = 0.0,
        kwargs...)
    OlderAdultVaccination(VaccineEffect(; kwargs...), min_age, campaign_time,
        booster_uptake)
end

EpiBranch.vaccine_effect(v::OlderAdultVaccination) = v.effect
EpiBranch.required_fields(::OlderAdultVaccination) = [:age]

function EpiBranch.apply_post_transmission!(v::OlderAdultVaccination, state, new_contacts)
    for ind in new_contacts
        ind.state[:age] >= v.min_age || continue
        EpiBranch._record_vaccination!(v, ind, v.campaign_time, state.rng)
    end
    return nothing
end

older = OlderAdultVaccination(min_age = 60, campaign_time = 10.0,
    efficacy = 0.8, delay_to_immunity = 14.0)
older_model = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0));
    interventions = [older], attributes = demographics())
older_results = simulate(older_model, 50; max_cases = 200, rng = StableRNG(1))
n_cases = sum(s -> length(s.individuals), older_results)
n_vaccinated = sum(s -> count(is_vaccinated, s.individuals), older_results)
println("Vaccinated: $n_vaccinated of $n_cases cases")
```

Forwarding keywords to `VaccineEffect` lets the constructor take the same effect
keywords as the built-in vaccinations. A parameter describing what a dose does
belongs in `VaccineEffect`, where every vaccination gains it at once; a
parameter describing whom a dose reaches belongs on the subtype.

An effect only your vaccination has is a field on it, `booster_uptake` above,
and its per-dose draw goes through the `_record_effect_draws!` hook, which
`_record_vaccination!` calls for every vaccination. `RingVaccination` records
`post_exposure_efficacy` and `onward_efficacy` that way:

```julia
_booster_uptake_key(label) = Symbol("booster_uptake_", label)

function EpiBranch._record_effect_draws!(v::OlderAdultVaccination, contact, label, rng)
    EpiBranch._store_draw!(v.booster_uptake, _booster_uptake_key, label, contact, rng)
    return nothing
end
```

For a scalar, `_store_draw!` stores nothing and `EpiBranch._dose_value` reads
the value straight off the vaccination; for a distribution or a function it
stores the draw.

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

Observation parameters, attribute-builder parameters and intervention predicates
accept callable objects as well as functions. Their argument signatures stay the
same. For example, a reporting rule can hold its threshold in a struct:

```@example extending
struct AgeDetection
    minimum_age::Float64
end
(rule::AgeDetection)(rng, ind) = ind.state[:age] >= rule.minimum_age ? 1.0 : 0.0
age_observation = PerCaseObservation(detection_prob = AgeDetection(50.0))
```

A callable observation anchor takes only `ind`, and a callable `Scheduled`
predicate takes the simulation state. Scalar and distribution inputs retain
their usual meanings wherever those forms are supported.

### Writing your own

For fields without a dedicated builder — anything in `ind.state` — write a
plain closure. Below, `:risk_group` is a custom state field, so it needs
the closure form; `susceptibility` is derived from it via
`transmission_traits`, which accepts a function:

```@example extending
risk_group = (rng, ind) -> (ind.state[:risk_group] = rand(rng) < 0.2 ? :high : :low)

attrs = [
    risk_group,
    transmission_traits(
        susceptibility = (rng, ind) -> ind.state[:risk_group] == :high ? 0.8 : 0.3,
    ),
]

model = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0)); attributes = attrs)
rng = StableRNG(42)
state = simulate(model; max_cases = 100, rng = rng)
n_high = count(ind -> get(ind.state, :risk_group, :low) == :high, state.individuals)
println("High-risk individuals: $n_high / $(length(state.individuals))")
```

### Sharing attributes within groups

Use `group_attribute` for a numeric value shared by a household, village or
other group. It samples once for the first member of each group and keeps that
value for the run. Here reporting probabilities vary between households:

```@example extending
reporting_attributes = [groups(50; key = :household),
    group_attribute(:reporting_probability; value = Beta(6, 4),
        group_key = :household)]
reporting_model = ModelSpec(BranchingProcess(Poisson(0.5), Exponential(5.0));
    attributes = reporting_attributes,
    observation = PerCaseObservation(
        detection_prob = (rng, ind) -> ind.state[:reporting_probability],
        from = :infection_time))
reporting_state = simulate(reporting_model; n_initial = 10, rng = StableRNG(42))
```

The group label must be set before the shared attribute. Reusing these builders
in further simulations draws fresh values, including when running in parallel.

### Composing attributes functions

The attributes list is applied in order, so later builders or closures can
read fields set by earlier ones:

```@example extending
combined = [
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Normal(40, 15)),
    risk_group,
    transmission_traits(
        susceptibility = (rng, ind) -> ind.state[:risk_group] == :high ? 0.8 : 0.3,
    ),
]

model_combined = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0));
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
linked = ModelSpec(BranchingProcess(NegBin(2.5, 0.16), gt);
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

scaled = ModelSpec(BranchingProcess(Poisson(2.0), ind -> Exponential(ind.state[:gt_scale])); attributes = host)

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

model_risk = ModelSpec(BranchingProcess(risk_offspring, Exponential(5.0); n_types = 1); attributes = risk_group)

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

## Infectiousness windows

A `BranchingProcess` is built from one or more `Infectiousness` windows. A
window is a source of secondary contacts attached to a case's timeline:

```julia
Infectiousness(offspring; from = :infection, until = (), kernel = NoGenerationTime())
```

- `offspring`: how many contacts this source makes (a `Distribution`, or a
  function for multi-type).
- `from`: the state at which the window opens. `:infection` (the default)
  resolves to the individual's infection time; any other symbol `s` resolves
  to `Symbol(s, :_time)` in `ind.state`, so `from = :infectious` reads
  `:infectious_time`. The window contributes nothing until that time is finite.
- `kernel`: the contact interval, measured from the `from` time. Each contact
  lands at the `from` time plus a draw from `kernel`. `NoGenerationTime()`
  places contacts at the `from` time itself.
- `until`: a tuple of state names. Each resolves to `Symbol(s, :_time)` on the
  infector, and the earliest of them ends the window. A contact whose time
  falls at or after it does not transmit, because the infector was removed first.

`from` and `until` are how transmission keys off natural history. You supply
the state times with transitions: a generic `Transition(:state; from, delay)`
writes `:state` and `:state_time`. A latent period, an infectious period, and a
window combine like this:

```julia
progression = [
    Transition(:infectious, from = :infection,  delay = latent_period),
    Transition(:recovered,  from = :infectious, delay = infectious_period),
]
window = Infectiousness(NegBin(R, k);
    from   = :infectious,
    until  = (:recovered,),
    kernel = contact_interval)
process = ModelSpec(BranchingProcess(window); progression = progression)
```

The window opens when the case becomes infectious and closes at recovery. The
`until` censor is a default competing risk on the same surface as
interventions, so isolation, tracing and vaccination compose with it.
Isolation comes from the `Isolation` intervention, not from a state in `until`.

The default single window with `from = :infection`, `until = ()` and a
generation-time `kernel` is the plain branching process.

Whether `until` is empty changes what the kernel means. With `until = ()` there
is no removal race, so the kernel is the realised generation interval. With a
non-empty `until` the kernel is the contact interval, and the generation
interval emerges from the race against removal. Giving a generation interval
and a competing recovery state for the same window counts the infectious period
twice; use one or the other.

A case can carry several windows with different timing and censoring, for
example community spread and a separate funeral source:

```julia
process = ModelSpec(BranchingProcess(
        (Infectiousness(NegBin(R, k); from = :infectious, until = (:recovered, :died), kernel = gt),
            Infectiousness(Poisson(λ); from = :died, until = (:buried,), kernel = funeral_kernel)));
    progression = progression)
```

Each window draws its own offspring, times them from its own `from` state, and
is censored by its own `until` states.

Two constraints:

- The analytical helpers (`single_type_offspring`, the chain-size laws) need a
  single window. Offspring across several windows is a fate-mixture with no
  closed form, so multi-window models are simulation only.
- If a window's `from` is a state that nothing writes, the window never opens.
  The constructor warns when it can detect this.

## Transmission routes

A case need not have one infectiousness profile and one set of people it can
reach. Real transmission is often several routes at once, each open over a
different stretch of the case's natural history and each ended by different
things. A [`RouteWindow`](@ref) is the unit that makes those one mechanism:

```julia
RouteWindow(name; from = nothing, until, kernel, reach = name,
            contacts_from = :infection, traceable = 1.0)
```

- `from` is the state at which this route's infectiousness begins. `:infection`
  opens it at the infection time; any other state opens it at that state's
  `<state>_time`. The default, `nothing`, takes the start the model derives from
  its progression, as the continuous-time processes do for their own `from`.
  A route whose `from` state is never reached contributes
  nothing, so a case that recovers never opens a funeral route and nothing is
  created only to be censored.
- `until` names the states that end the route, and the window closes at the
  earliest of their times. **A state listed by one window and not another
  censors only the first.** That is the whole point: it is how a control measure
  cuts one route and leaves another.
- `kernel` is the route's contact-interval distribution, measured from the
  window opening. A model reads it when it resolves `reach` into the route's
  contacts.
- `reach` tags who the route reaches, for the model to resolve — only the model
  knows its own structure.
- `contacts_from` is the state from which the people the route reaches count as
  the case's contacts for tracing. The default, `:infection`, suits standing
  relationships such as a household. A funeral route sets `contacts_from =
  :died`, so its contacts are traced only if the funeral happened before the
  route was cut, and not before it. This is separate from `from`: a route whose
  infectiousness starts at onset still reaches the same household from infection.
- `traceable` is the probability that a case can name a contact made on the
  route. People can name the people they live with and cannot name the strangers
  they stood next to, so a household route keeps the default `1.0` and an
  anonymous community route might take `0.0`. `true` and `false` also work, as
  `1.0` and `0.0`.

### Traceability and contact tracing

Naming and tracing are two steps. A route's `traceable` is the chance that the
case can identify a contact at all. The tracing intervention's own probability
(its `TraceRate`) is the chance that the programme then reaches a contact it has
been told about. A contact is traced only if both succeed, so the probabilities
multiply. With a community route at `traceable = 0.5` and
`ContactTracing(probability = 0.8)`, 40% of the contacts a case meets only in
the community are traced. Set each probability for what it describes: a limit
on naming belongs in `traceable` alone, and counting it again in the tracing
probability would reduce tracing twice.

A model applies `traceable` when it assembles the contacts it hands to
[`trace_contacts!`](@ref EpiBranch.trace_contacts!), so tracing interventions
never see the routes. Each model sets its own rule for a contact reachable on
several routes. `RoutedNetwork` makes one draw per pair of case and contact,
names the contact with the highest of its routes' probabilities, and traces it
no earlier than the routes it was named on allow. Draws use the simulation's
random number generator, so runs are reproducible. A route at exactly `0.0` or
`1.0` needs no draw, so a model that leaves every route at the default uses no
random numbers for naming.

### Being cut by an intervention

Route censoring is otherwise written in states the natural history produces, but
an intervention removal cannot be read off a state key alone: perfect isolation
takes a case out of transmission, whereas leaky isolation only reduces it and a
window cannot express that. `infectious_removal_time` resolves the difference,
and a window opts into it by listing the reserved
[`EpiBranch.INTERVENTION_REMOVAL`](@ref) in its `until`.

So self-isolation is two routes differing in one tuple:

```julia
community = RouteWindow(:community;
    until = (:recovered, EpiBranch.INTERVENTION_REMOVAL),
    kernel = Exponential(12.0), reach = community_adjacency)
household = RouteWindow(:household; until = (:recovered,),
    kernel = Weibull(1.5, 3.0), reach = household_adjacency)
```

A case that isolates stops transmitting in the community and goes on infecting
the people it lives with, to the end of its infectious period.

### What stays fixed

`R` remains the intrinsic reproduction number a case would achieve if never
removed, and the realised figure falls out of which routes were cut and when.
The dispersion `k` remains the intrinsic offspring dispersion and is
deliberately kept separate from the shape of the infectious period, so a count
is never drawn from a duration — that would couple the offspring draw to timing
and break the decoupling the engine rests on.

### Reading them in a process

A process that carries routes passes them to the continuous-time race as
`(window, targets)` pairs instead of a single `from`/`until`/`targets`, and
resolves each window's `reach` into its own targets closure, yielding
`(target_id, kernel)` pairs. Passing both `routes` and the shorthand is an
error, because the routes would silently drop the shorthand's censoring. A
model that passes no routes gets a single window that is cut by intervention
removal.

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
transmission probability belongs to the *edge* (an edge-owned transmission
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
[`seed!`](@ref EpiBranch.seed!).

Models whose contacts can be *shared* across parents within a generation
(networks, households, clustering) also override
[`collect_exposures`](@ref) with [`gather_by_target`](@ref), which
deduplicates shared targets so a node reached several times resolves once.

A model that drives its **own simulation loop** — rather than the
generation-based engine — resolves each case's natural history itself by
calling [`resolve_transitions!`](@ref EpiBranch.resolve_transitions!)`(state, individual)`
once per case, after its attributes and intervention state are set. This runs
the model's clinical transitions (placed on the state by
[`new_state`](@ref EpiBranch.new_state)) and stamps the timeline keys
(`:onset_time`, `:outcome_time`, …) the line list and any likelihood read. The
continuous-time household and network processes, which step cases in
infection-time order instead of by generation, are the worked examples.

For **analytical inference helpers** that route through the offspring
specification (`reproduction_number`, `extinction_probability`,
`epidemic_probability`, `probability_contain`, `proportion_transmission`,
`chain_size_distribution`), define one method:

- [`single_type_offspring`](@ref)`(model)` returning the offspring
  distribution (or any object for which `chain_size_distribution` is
  defined). Specialise this and you get the analytical helpers for
  free.

For **likelihoods** on data types that don't go through the offspring
spec, define methods on `loglikelihood` directly.

A structure-driven model simulated by the continuous-time race can reuse the
pairwise survival likelihood, whose generative model is that race. Beyond the
infection times, the density needs to know who could have infected whom. Define
an infection-layer type that subtypes [`InfectionLayer`](@ref) and give it a
[`contact_structure`](@ref EpiBranch.contact_structure) method that returns a
membership vector for groups whose members all mix, or an adjacency list for
anything else. [`compile_contact_pairs`](@ref) and [`pairwise_surv_loglik`](@ref)
then work on it with no further methods, including the per-edge, covariate and
community-hazard terms, and `loglikelihood` needs one method that forwards to
them:

```julia
struct MyInfections{T <: Real} <: InfectionLayer
    contacts::Vector{Vector{Int}}    # contacts[i]: who host i can infect
    infection_time::Vector{T}        # NaN if never infected
    infectious_time::Vector{T}       # the infectious window opens
    removal_time::Vector{T}          # and closes (Inf if still open)
    is_index::Vector{Bool}           # introduced from outside
    obs_end::T                       # community introductions stop
    followup_end::T                  # observation ends (optional; Inf if absent)
end
EpiBranch.contact_structure(d::MyInfections) = d.contacts

Distributions.loglikelihood(d::MyInfections, m::MyModel) =
    pairwise_surv_loglik(m.kernel, d; external_hazard = m.external_hazard)
```

`HouseholdInfections` in `EpiHouseholds` and `NetworkInfections` in `EpiNetwork`
are the worked examples.

For optional **state accessors**, override `population_size` and
`n_types` if your model has values for them. The defaults
(`NoPopulation()`, `1`) are fine if not.

Your process describes the transmission alone — it does **not** carry the modelling
layers (a clinical `progression`, `interventions`, `attributes`, or an
`observation` model). Those are composed onto it by the user with a
[`ModelSpec`](@ref) and threaded into the run by the engine, so an
offspring-driven model gets them for nothing: `simulate(ModelSpec(MyModel(...);
progression = [...], interventions = [...], attributes = attr))` applies each
layer without your model storing a field or defining an accessor for it. The
engine reads the composed progression and applies its transitions to each new
contact, the same as for `BranchingProcess`.

If your generation-time distribution is not stored in a field literally named
`generation_time`, override [`model_generation_time`](@ref EpiBranch.model_generation_time)`(m)` to point at it —
that accessor is what the engine calls.

A **structure-driven** model that runs its own simulation loop (in
infection-time order, rather than the generation-based engine) receives the
composed layers as arguments instead: define
`EpiBranch._simulate(m::MyModel, sim_opts; interventions, attributes,
progression, observation, rng, condition, max_attempts)`, read the layers off
the arguments, and derive any window state you need (the infectious-window
`from`, say) from the `progression` there. The continuous-time household and
network processes are the worked examples; `ModelSpec` routes `simulate` and
`loglikelihood` to that method for you.

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

An observation model is composed onto your process with a `ModelSpec`, like any
other layer — your model stores nothing and defines nothing:

```julia
simulate(ModelSpec(MyModel(...); observation = PerCaseObservation(detection_prob = 0.7)))
```

The engine applies the observation after the run, and `loglikelihood(data,
spec)` reads it off the spec — the same path `BranchingProcess` takes.

## A fixed-size population on the Sellke pool

The built-in [Homogeneous models](homogeneous.md) tutorial covers
`HomogeneousProcess`, a closed population where everyone mixes with everyone else
at the same rate. Mixing is often uneven, though: age bands, sex, income strata
or spatial patches contact each other at different rates, so susceptibles in
different groups feel a different force of infection. You can build a model like
that on the same pool without touching the simulation itself. Only one thing
changes from the homogeneous case: how the force of infection depends on which
group a susceptible belongs to. You supply it as two pieces:

1. **Which attributes define the mixing groups** — `mixing_by`, a tuple of
   attribute keys each individual already carries (`:age_band`, `:ses`, `:patch`;
   real attributes, not a synthetic group index). A susceptible's group is the
   tuple of those attribute values. With `mixing_by = ()` everyone lands in one
   group, which recovers the homogeneous case.
2. **The force of infection** `force(group, counts)` — the hazard on a
   susceptible in a given group. `counts` is a `Dict` mapping each mixing group to
   the infectiousness-weighted number currently infectious in it, each case
   contributing its own `infectiousness` (1 by default). Homogeneous mixing is
   `β/N` times the total of those counts; structured mixing applies a contact
   matrix to the per-group prevalence.

A mixing group is always the *tuple* of `mixing_by` values, so it stays a tuple
even when there is a single attribute. Under `mixing_by = (:age_band,)` a
susceptible in band `b` has group `(b,)`, not bare `b`. Inside `force` you
therefore read the band out with `group[1]` and key `counts` by `(h,)`. That one
wrinkle is what usually trips people up on a first read.

### A worked age-structured example

Below is a two-age-band population with an asymmetric contact matrix, where the
younger, more socially active band mixes more than the older band. It drives the
pool through `EpiBranch._sellke_pool!`. That function is internal for now: the
underscore means it is not part of the public API and may be renamed or given a
public wrapper in a later release. The `mixing_by`/`force` contract shown here is
the stable part and will carry over; only the call site would change. If you
build on it, pin your package version.

```julia
using EpiBranch, Distributions, Random

# A closed population of N split into two age bands of equal size. Band 1 is the
# more socially active one; `band_of` reads an individual's band off its id.
N = 2000
n = [N ÷ 2, N ÷ 2]                     # band sizes
band_of = i -> (i <= n[1] ? 1 : 2)

# A 2×2 contact matrix: M[b, h] is the mean rate at which one infectious
# individual in band h contacts a susceptible in band b. Band 1 mixes far more.
M = [3.0 0.5;
     0.5 0.5]

# Force of infection on a susceptible in mixing group `group`, given `counts`.
# A group is the tuple of :age_band values, so band b is `(b,)`: read the band
# with group[1] and index counts by (h,). This sums, over bands h, the contact
# rate M[b, h] times band h's prevalence counts[(h,)] / n[h].
force = (group, counts) -> begin
    b = group[1]
    sum(M[b, h] * get(counts, (h,), 0) / n[h] for h in 1:2)
end

# A HomogeneousProcess supplies only the fixed pool and its removal states; the
# force above replaces its transmission rate, so any placeholder value does. The
# natural history (progression) and the empty forcing layers are handed to the
# pool directly. Tag each individual's :age_band as it is created; any attribute
# works, including the built-in demographics (:age, :sex, :risk_group).
carrier = HomogeneousProcess(; transmission_rate = 1.0, population_size = N)
progression = [Transition(:recovered; from = :infection,
    delay = Exponential(1.0), terminal = true)]
rng = MersenneTwister(1)
state = EpiBranch.new_state(carrier, progression, NoAttributes(), rng)
EpiBranch.add_individuals!(state, N, AbstractIntervention[];
    setup = (ind, i) -> (ind.state[:age_band] = band_of(i)))

# Run the Sellke pool. `mixing_by = (:age_band,)` names the attribute that groups
# individuals; each group is read from it, and the model supplies only the force.
EpiBranch._sellke_pool!(state, collect(1:N), rng; mixing_by = (:age_band,),
    force = force, n_initial = 5,
    from = EpiBranch._resolve_infectious_from(carrier.from, progression),
    until = carrier.until)

linelist(state)
```

Band-1 susceptibles feel a higher force and reach a higher attack rate. To check
the wiring, set `M` uniform: the two bands should collapse back to a single
homogeneous pool with the SIR final size. The same pattern extends to further
strata. Give individuals a `:ses` attribute and pass
`mixing_by = (:age_band, :ses)`, and `force` now receives a `(band, ses)` tuple
as its group and a `counts` Dict keyed by `(band, ses)` pairs. From there you can
write whatever contact structure you want: a full matrix over every
`(band, ses)` combination, or a factorised one where band and SES contacts
multiply independently.

Competing risks carry over with one restriction. The pool attributes each
contact to an infector drawn in proportion to infectiousness, which is a uniform
draw while every infective is at the default, because `force` does not say how
much each infective contributes to it. With more than one mixing group that
attribution is not weighted by the contact matrix, so a risk that depends on who
the infector is would be applied against the wrong infectors. The pool therefore
refuses, with an error, a leaky `Isolation` and any intervention with its own
`competing_risk` other than the vaccinations' protection of the contact.
Per-individual infectiousness is not refused: it reaches the force through the
weighted counts, so it needs no attribution to be exact. Risks on the contact
alone, such as a per-individual susceptibility, apply exactly. Differences in infectiousness
between groups belong in `force`.

The natural history, isolation and line-list output are all unchanged from
`HomogeneousProcess`. Internally these map to the engine's build, time, intervene
and resolve phases, described in the [design overview](../design.md), but you do
not need any of that to write a structured model; the two pieces above are the
whole interface.

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

Usage: `ModelSpec(BranchingProcess(...); observation = CensoredAtSize(10))`. No
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
3. Threshold and extinction dispatch (optional): [`reproduction_number`](@ref)`(offspring)` and [`extinction_probability`](@ref)`(offspring)`, so the model-level helpers answer for models built from the type. `src/analytical/cluster_mixed.jl` and `src/analytical/multi_type.jl` are the examples.
4. A `BranchingProcess` constructor so the type can be stored in the `offspring` field.

See `src/analytical/cluster_mixed.jl` for the full pattern, including how `ClusterMixed` caches per-chain state on the index case and has descendants inherit it through `parent_id`.

## Adding per-observation metadata

[`ChainSizes`](@ref) carries one per-observation field, `seeds` (the number
of index cases in each multi-seed cluster). A cluster's real-time "is it
finished?" weight is a second analyst decision, but it is supplied at
likelihood time through the `prob_concluded` keyword of `loglikelihood` rather
than stored on the data — the mixture it drives is only defined against the
analytical chain-size law. These two show the pattern for any per-cluster
information.

If your analysis needs different or richer per-cluster information, you
have two options.

### Stay in `ChainSizes` and pre-compute

If the new information resolves to a flag or a count that the existing
likelihood already handles, derive it upstream and pass it in. The Endo
7-day time-censoring rule is an example: it looks like time censoring but is
just a way to compute a per-cluster `prob_concluded` (`1.0` for a finished
cluster, `0.0` for an ongoing one).

```julia
using Dates
is_ongoing(latest_case, cutoff; window_days = 7) =
    cutoff - latest_case < Day(window_days)

prob_concluded = Float64.(.!is_ongoing.(last_case_dates, cutoff_date))
data = ChainSizes(sizes; seeds = imports_per_cluster)
loglikelihood(data, offspring; prob_concluded = prob_concluded)
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
| Custom vaccination | Struct `<: AbstractVaccination` holding a `VaccineEffect` + `vaccine_effect` + `apply_post_transmission!` | Each generation |
| Time-dependent intervention | `Scheduled(iv; start_time = ...)` + `intervention_time`, `reset!` on `iv` | After each hook |
| Capacity-constrained intervention | `CapacityConstrained(iv; budget_per_period = ...)` + `capacity_key`, `capacity_time_key` on `iv` | `apply_post_transmission!` |
| Custom attributes | Function `(rng, ind) -> nothing` | Individual creation |
| Layered attributes | `[f1, f2, ...]` | Individual creation |
| Custom offspring (function) | Function `(rng, ind) -> Int` | Offspring draw |
| Multi-type offspring | Function `(rng, ind) -> Vector{Int}` | Offspring draw |
| Custom offspring (type) | Struct + `draw_offspring`, `chain_size_distribution` | Offspring draw + analytics |
| Custom transmission model | Struct `<: TransmissionModel` + `generate_offspring` (offspring-driven) or `initialise_state` + `contacts_of` + `gather_by_target` (structure-driven); optional `single_type_offspring`, accessors | Simulation + analytics |
| Transmission route | `RouteWindow(name; from, until, kernel, reach)` on a process that reads them | Continuous-time race, per case |
| Structured fixed-size pool | Reuse the Sellke pool: name the mixing attributes with `mixing_by` (a tuple of attribute keys) and supply a `force(group, counts)` | Simulation |
| Pairwise likelihood for a structure | Struct `<: InfectionLayer` + `contact_structure`; `compile_contact_pairs` and `pairwise_surv_loglik` then apply | Likelihood evaluation |
| Custom observation model | Struct `<: ObservationModel` + `observe(base, ::YourObs)` (analytics) and/or `apply_observation!(::YourObs, state, rng)` (simulation) | Analytics / inference |
| Per-observation metadata | Either pre-compute into existing `ChainSizes` fields, or define a new data type with a `loglikelihood` method that calls `_chain_size_logpdf` | Likelihood evaluation |
| Sim ↔ analytical test | `generative_model`, `observe_chain_sizes` | Regression test |

### Callable rules in branching processes

Offspring rules accept `(rng, individual)` or `(rng, individual, state)`.
When both methods exist, simulation uses the method with `state`. A generation-time
rule accepts the individual and returns a distribution. These rules can be
closures or callable objects:

```julia
struct OffspringRate
    mean::Float64
end
(rule::OffspringRate)(rng, ind) = rand(rng, Poisson(rule.mean))

struct ContactInterval
    mean::Float64
end
(rule::ContactInterval)(ind) = Exponential(rule.mean)

process = BranchingProcess(OffspringRate(0.6), ContactInterval(2.0))
simulate(process; rng = Xoshiro(42))
```

The matrix constructor also accepts a callable distribution family as its second
argument. Distributions and custom offspring specifications with a specialised
`draw_offspring` method keep their existing dispatch. Analytical calculations
require an offspring law with the corresponding analytical methods; accepting a
callable for simulation does not provide a closed form for that rule.

### Choosing initial cases in a fixed population

Network and household simulations accept population IDs through `initial_cases`.
Selection criteria belong in the calling code:

```julia
using EpiBranch, EpiNetwork, Distributions, Random

adjacency = [Int[] for _ in 1:5]
process = NetworkProcess(adjacency, Exponential(2.0))
chosen = [2, 4]
state = simulate(ModelSpec(process); initial_cases = chosen, rng = Xoshiro(42))
```

With no edges, only IDs 2 and 4 are infected. The same keyword works with
`RoutedNetwork`, `HouseholdProcess` and repeated or parallel simulation. IDs refer
to the whole population, including across households. An empty vector starts
with no infections. The simulator copies the vector and checks for duplicates and
IDs outside the population.

Omitting `initial_cases` preserves default seeding and its random draws. A chosen
vector replaces that rule: it cannot be combined with `n_initial` or an active
`external_hazard`. Initial cases are infections at time zero; ongoing external
introductions describe a separate process. Select IDs with an explicit RNG in
caller code when selection itself is random.
