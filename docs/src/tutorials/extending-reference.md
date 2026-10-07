# Extension reference

This page collects the details needed when writing interventions, transitions
and models of your own: the values the package keeps on each person, how the
continuous-time models treat risks, which interventions the infection
likelihoods can fit exactly, and the rules for custom clinical transitions and
intervention actions. The guides are [Extending EpiBranch](extending.md),
[Writing an intervention](writing-interventions.md) and [New transmission
structures](new-structures.md).

## Individual state and reserved keys

Each person has a few fixed fields that the simulation reads (their place in
the transmission tree, infection time, susceptibility and infectiousness), and
a dictionary, `ind.state`, holding everything else. Interventions, population
characteristics, clinical transitions and observation models each keep a few
named values (keys) there. The reasons for this split are in [Notes for
contributors](@ref "Why individual state is an open dictionary").

Read a value with a one-line accessor that supplies a default when the value is
missing, such as `is_isolated(ind) = get(ind.state, :isolated, false)::Bool`.
`get(dict, key, default)` returns `default` when `key` is absent, and `::Bool`
asserts the type. For a time, keep the accessor generic in the number type so
that likelihoods can be differentiated through it:
`onset_time(ind::Individual{T}) where {T} = convert(T, get(ind.state, :onset_time, T(NaN)))`.
Code inside the package adds such accessors in `src/state_accessors.jl` rather
than calling `get(ind.state, ...)` directly.

### Reserved keys

The package reserves the keys below. Custom interventions and other packages
should choose names that do not clash with them. Keys starting with an
underscore are internal.

| Key | Type | Default | Set by | When set |
|---|---|---|---|---|
| `:infected` | `Bool` | `true` | Simulation | When infection is decided |
| `:infection_route` | `Symbol` | — | Simulation (models with routes) | When infection is decided |
| `:type` | `Int` | `1` | Simulation (multi-type) | When the contact is created |
| `:onset_time` | `Float64` | `NaN` | `clinical_presentation` | Creation |
| `:asymptomatic` | `Bool` | `false` | `clinical_presentation` | Creation |
| `:age` | `Real` | — | `demographics` | Creation |
| `:sex` | `Symbol` | — | `demographics` | Creation |
| `:risk_group` | `Symbol` | — | `demographics` | Creation |
| `:group` | `Int` | — | `groups` | Creation |
| `:isolated` | `Bool` | `false` | `Isolation` | `resolve_individual!` |
| `:isolation_time` | `Float64` | `Inf` | `Isolation` | `resolve_individual!` |
| `:isolation_release_time` | `Float64` | `Inf` | `Isolation`, `ContactTracing`'s `Quarantine` | `resolve_individual!` / `apply_trace!`; when the block ends |
| `:_removal_stretches` | `Vector{Tuple{Float64,Float64}}` | `[]` | `set_isolated!` | Internal. Every `(start, release)` a removal took the person out for, merged and sorted; a release of `Inf` for one that never ends |
| `:_quarantine_stretches` | `Vector{Tuple{Float64,Float64}}` | `[]` | `ContactTracing`'s `Quarantine` | `apply_trace!`; internal. The quarantine's own stretches, apart from the shared history |
| `:_isolated_by_isolation` | `Bool` | `false` | `Isolation` | `resolve_individual!`; internal |
| `:_isolation_unrecorded` | `Bool` | `false` | `Isolation`, `ContactTracing` | `resolve_individual!` / `apply_trace!`; internal. The isolation removes the case from transmission without counting as a detection |
| `:_isolation_time_before_isolation` | `Float64` | — | `Isolation` | `resolve_individual!`; internal. The time of an isolation already in place before this one |
| `:_isolation_release_time_before_isolation` | `Float64` | — | `Isolation` | `resolve_individual!`; internal. The release time of an isolation already in place before this one |
| `:_isolation_unrecorded_before_isolation` | `Bool` | — | `Isolation` | `resolve_individual!`; internal |
| `:test_positive` | `Bool` | `false` | `Isolation` | `resolve_individual!` |
| `:traced` | `Bool` | `false` | `ContactTracing` | `apply_post_transmission!` / `trace_contacts!` |
| `:quarantined` | `Bool` | `false` | `ContactTracing` | `apply_post_transmission!` / `trace_contacts!` |
| `:_traced_isolation_time` | `Float64` | `Inf` | `ContactTracing` → `Isolation` | Internal hand-over; may precede onset, so `Isolation` holds it back to onset |
| `:trace_time` | `Float64` | — | `ContactTracing` | `apply_post_transmission!` / `trace_contacts!` |
| `:_ring_remaining` | `Int` | `0` | `ContactTracing` (`depth > 1`) | `apply_post_transmission!` / `trace_contacts!`; internal |
| `:_ring_propagated` | `Bool` | `false` | `ContactTracing` (`depth > 1`) | `trace_contacts!`; internal |
| `:traced_by` | `Int` | — | `ContactTracing` | `apply_post_transmission!` / `trace_contacts!` |
| `:trace_level` | `Int` | — | `compute_trace_level!` | After the simulation |
| `:vaccinated[_<label>]` | `Bool` | `false` | `AbstractVaccination` | Creation / `apply_post_transmission!` |
| `:vaccination_time[_<label>]` | `Float64` | `Inf` | `AbstractVaccination` | `apply_post_transmission!` |
| `:vaccine_efficacy[_<label>]` | `Float64` | — | `AbstractVaccination` | Creation / `apply_post_transmission!` |
| `:post_exposure_efficacy[_<label>]` | `Float64` | — | `RingVaccination` (varying `post_exposure_efficacy`) | `apply_post_transmission!` |
| `:onward_efficacy[_<label>]` | `Float64` | — | `RingVaccination` (varying `onward_efficacy`) | `apply_post_transmission!` |
| `:immunity_time[_<label>]` | `Float64` | — | `AbstractVaccination` | `apply_post_transmission!` |
| `:severity_efficacy[_<label>]` | `Float64` | — | `AbstractVaccination` | `apply_post_transmission!` |
| `:coverage_declined[_<label>]` | `Bool` | `false` | `GroupVaccination` | `apply_post_transmission!` |
| `:infection_aborted_time` | `Float64` | — | Simulation, through `abort_infection!` (for example by `RingVaccination`'s `post_exposure_efficacy`) | Any intervention hook |
| `:capacity_admission_time_<capacity_key>` | `Float64` | — | `CapacityConstrained` | `apply_post_transmission!` |
| `:reporting_time` | `Float64` | `Inf` | `Reporting` transition | `resolve_individual!` |
| `:admitted` | `Bool` | `false` | `Hospitalisation` transition | `resolve_individual!` |
| `:admission_time` | `Float64` | `Inf` | `Hospitalisation` transition | `resolve_individual!` |
| `:death_candidate_time` | `Float64` | `Inf` | `Outcome` transition | `resolve_individual!` |
| `:recovery_candidate_time` | `Float64` | `Inf` | `Outcome` transition | `resolve_individual!` |
| `:outcome` | `Symbol` | — | `Outcome` transition | `resolve_individual!` (terminal) |
| `:outcome_time` | `Float64` | — | `Outcome` transition | `resolve_individual!` (terminal) |
| `:reported` | `Bool` | `false` | `PerCaseObservation` *or* `Reporting` transition | After the simulation / `resolve_individual!` |
| `:report_time` | `Float64` | — | `PerCaseObservation` | After the simulation |
| `:cluster_theta` | `Float64` | — | `ClusterMixed` | First read in a simulation |
| `:vaccine_acceptance` | `Float64` | — | `vaccine_acceptance` | Creation (default key; can be changed) |
| `:infectious_time` | `Float64` | `Inf` | `Transition(:infectious, …)` | `resolve_individual!` |
| `:recovered_time` | `Float64` | `Inf` | `Transition(:recovered, …)` | `resolve_individual!` |
| `:susceptible_again_time` | `Float64` | `Inf` | `Transition(:susceptible_again, …)` | `resolve_individual!` |

!!! warning "Reporting transition and per-case observation clash"
    `:reported` is written both by the `Reporting` clinical transition (from
    its probability) and by `PerCaseObservation` (after the simulation, from
    its detection probability). Using both in the same simulation is not
    supported, because each overwrites the other.

### Vaccination keys

The vaccination keys are named after the `dose_label`. The default label writes
plain `:vaccinated`, `:vaccination_time`, `:vaccine_efficacy` and
`:immunity_time` (and, for `RingVaccination`, `:post_exposure_efficacy` and
`:onward_efficacy`); any other label is appended, so `dose_label = :boost`
writes `:vaccinated_boost` and so on. Doses of a multi-dose schedule then do
not overwrite each other.

Under `AllOrNothingMode`, `:vaccine_efficacy` holds whether the person
responded, `1.0` or `0.0`, drawn once from the dose's efficacy. A dose recorded
by a population characteristic before the run has its efficacy turned into
that response at creation. `:immunity_time` (the vaccination time plus a draw
from `delay_to_immunity`), `:post_exposure_efficacy` and `:onward_efficacy`
each hold one draw taken at vaccination, from a parameter that may be a number,
a distribution or a function, so every exposure of a person is judged against
the same value. `:post_exposure_efficacy` and `:onward_efficacy` are written
only when the parameter is a distribution or a function; a number is the same
for everyone and is read from the intervention.

`:immunity_time` and `:severity_efficacy` let a clinical transition read a
vaccine's effect on severity (mortality, or any other outcome a `progression`
transition decides) without affecting transmission. Neither is used by
`competing_risk`. A transition's `probability` reads them through the
[`immunity_time`](@ref) and [`severity_efficacy`](@ref) accessors, and checks
the immunity time so that a dose whose immunity has not developed by the time
of the outcome gives no protection.

### Infections ended early

`:infection_aborted_time` marks an infection that ended before symptom onset,
as a post-exposure dose of `RingVaccination` or an antiviral can end it. Any
intervention records it by calling [`EpiBranch.abort_infection!`](@ref), which
keeps the earliest time, and [`EpiBranch.infection_aborted_time`](@ref) reads
it. The person is still infected but transmits nothing from that time, for as
long as the key is present. There is no onset: `:onset_time` is `NaN` while
`:asymptomatic` stays `false`, so isolation, tracing and clinical transitions
that start from onset never happen.

The clinical course ends at the abort time. When transitions are worked out,
any transition that would take effect at or after that time, whatever its
`from`, is undone and the values it wrote are restored, so no
hospitalisation, death or `:outcome` follows. Transitions that take effect
earlier stand. The check reads the `_time` keys a transition writes, so it
covers a custom transition that records its time under a `_time` key, as the
built-in ones do.

On branching processes `apply_post_transmission!` runs before infection is
decided, so an abort recorded there is set against a contact's provisional
infection time, its earliest exposure. The key is removed and the onset
restored when the contact turns out not to have been infected by an exposure
before the abort: when no exposure infected it, or when it was infected by a
later exposure at or after the abort time. The key is therefore only present on
an infected person whose infection it ended. `RingVaccination` draws again each
time a contact who already has the dose is exposed: a person created in advance
who escapes one exposure gets a new draw for the exposure that later infects
them. On the continuous-time models the infection time is final by the time
`on_infection_settled!` runs, and an abort recorded there needs no such check.

### Isolation keys

Isolation is recorded under `:isolation_time`, with `:isolation_release_time`
for when it ends. `set_isolated!` requires that release as its `release_time`
keyword; a release of `Inf` means the isolation never ends. These two describe the removal in force, which
is what a detection reads. The full history, which a likelihood needs, is the
list of stretches under `:_removal_stretches`, because one pair of times cannot
record that a person was quarantined, released and later isolated again. A
window that isolation should end lists [`EpiBranch.INTERVENTION_REMOVAL`](@ref)
in its `until` (see [Transmission routes](new-structures.md#Transmission-routes)), which respects leaky
isolation. `:isolated` in an `until` refers to a `Transition(:isolated, …)` in
the natural history. Set and undo isolation with `set_isolated!` and
`clear_isolated!`.

`:isolation_time` is when the case leaves transmission and
`:isolation_release_time` when it may resume; competing risks and
`INTERVENTION_REMOVAL` read these. Whether the isolation also counts as a
detection is a separate question, answered by `is_isolated`, which `OnIsolation`
tracing, group vaccination and the line list read. `Isolation` answers no for
an isolation at or after [`outcome_time`](@ref), since a self-report or trace
reaching a case that has already recovered or died describes a detection that
did not happen. It then sets `:_isolation_unrecorded` and keeps the removal.
The eligibility makes that decision through
[`EpiBranch.records_isolation`](@ref): define a method of it for a policy that
does record a late detection, such as a death found at burial.

### Tracing keys

The tracing keys are written by two hooks, because the two kinds of model reach
contacts differently: `apply_post_transmission!` on branching processes and
`trace_contacts!` on the continuous-time models. Both use the same per-pair
policy, so the keys and their meanings are the same either way; see [Which
hooks run on which model](@ref).

`:traced_by` is the person a contact was traced from: the *first*, from the
earliest exposure, since each person is traced at most once.
`compute_trace_level!` follows it back to the index case after the simulation
to set `:trace_level` (the distance from the index case, which has level `0`).
On a tree the level is exact. On a `NetworkProcess` with cycles it is the
depth along the path by which the person was first traced, and **not**
necessarily the shortest distance to the nearest index case; do not read it as
one.

### Naming your own keys

Built-in keys use short names like `:isolated`, `:traced` and `:age`, and these
are reserved. If you add keys from another package, start them with a short
tag for your package so they clash neither with built-in keys nor with those of
other packages.

A key an intervention keeps purely for its own records (where a value came
from, such as `Isolation`'s `:_isolated_by_isolation`, or a stored earlier
value, such as its `:_isolation_time_before_isolation`) starts with an
underscore, as [`EpiBranch._action_cache`](@ref)'s `:_intervention_actions`
does. The underscored names in the table are reserved along with the others, so
a key of your own puts your package's tag after the underscore:
`:_mypkg_budget`, not `:_budget`. [`linelist`](@ref) drops every key starting
with an underscore. Any other key becomes a line-list column once a model part
writes it, whether or not the package expected it.

Values that belong to a whole run, such as an index an intervention builds once
and reuses, go in `state.scratch`, a `Dict` on the
[`SimulationState`](@ref) that the simulation never reads and discards with the
state. Its keys follow the same rule. Built-in interventions use a tuple whose
first element names what the entry holds, as `GroupVaccination` keeps each
group's members under `(:group_members, key)`, and a key added from another
package starts with that package's tag.

### State times

A generic `Transition(:state; …)` writes the flag `:state` and the time
`:state_time` (that is, `Symbol(state, :_time)`). Infectiousness windows read
the same names: `from = :infectious` reads `:infectious_time`, and
`until = (:recovered,)` reads `:recovered_time`. The `_time` keys your
transitions write are therefore the names your windows refer to, and they must
match. `:infectious_time` and `:recovered_time` are the usual natural-history
pair; any other state you add a transition to records its own `<state>_time`
the same way.

Reinfection after waning immunity uses the same convention: a progression
listing `Transition(:susceptible_again, from = :recovered, delay = Exponential(180))`
writes `:susceptible_again_time`, which [`susceptible_again_time`](@ref) reads
and [`EpiBranch.HostImmunity`](@ref) checks. Storage is a separate matter:
[`Individual`](@ref)'s fields describe only the current infection episode, so
a model whose `contacts_of` offers an already-infected person as a contact
again (once `HostImmunity` lets the exposure through) gets the closing episode
archived onto `episodes` instead of overwritten; see
[`InfectionEpisode`](@ref). No built-in model does this yet. A model that allows
reinfection writes `contacts_of` to keep offering people after their first
infection, instead of leaving them out.

## Continuous-time models: further details

These points concern the network, household and homogeneous models, which
simulate a fixed population in continuous time (see [Which hooks run on which
model](@ref)).

- **Susceptibility and infectiousness.** These are fixed properties of the two
  people rather than events at a time, so the models build them into the
  timing of contacts: a multiplier `m` turns a pair's contact-interval survival
  `S(t)` into `S(t)^m`, the homogeneous pool scales each susceptible's
  resistance and weights each infectious person's share of the force of
  infection, and the rate of introductions from outside is scaled the same
  way. A multiplier of 0 never transmits and draws nothing. A fixed
  proportional effect can use these per-person values, or an adjusted
  contact-interval distribution, instead of a risk, which avoids drawing
  contacts only to block them.
- **Blocked contacts need an end to the infectious period.** After a contact is
  blocked, the next contact of the pair is drawn later in the same window,
  which needs the window's remaining cumulative rate to be finite. A model
  stops with an `ArgumentError` when the contact-interval survival is zero at
  the end of the window: an unbounded window with an exponential contact
  interval, or a bounded contact interval that ends inside the window. The
  homogeneous pool needs finite removal times for every infectious person
  when a contact is blocked. This happens even for a risk that might later let
  infection through, since the simulation cannot tell from a risk function
  whether contacts will ever stop. Give the infectious or introduction window a
  finite end with nonzero contact-interval survival at that end, or express a
  fixed protection through the per-person values or the contact interval. A
  `Dirac` contact interval has no contact after its single time and needs
  nothing more. Accuracy still depends on how the contact interval computes its
  survival: one computed as `1 - cdf` loses precision in the tail when the CDF
  rounds to one.
- **Introductions from outside.** On a model with an `external_hazard`, an
  introduction is put to the risks acting on the person introduced: their
  susceptibility, a vaccine's protection, a risk of your own. Isolation and
  quarantine do not remove an outside source. An introduction has no infector,
  so the person introduced stands in for one; return `nothing` from your risk
  when `parent === contact` (the same person) if it reads properties an
  outside source cannot have.
- **Choosing routes.** [`EpiBranch.risk_applies`](@ref) chooses which routes an
  intervention's risk acts on. It receives the route window, or `nothing` for
  an introduction from outside. Its default `true` keeps vaccine protection on
  every route. Isolation and contact tracing test whether the route lists
  `EpiBranch.INTERVENTION_REMOVAL` in `until`, and `Scheduled` and
  `CapacityConstrained` pass the question on to the intervention they hold.
  An intervention of your own can choose any set of routes, for example only
  the routes that intervention removal ends:

  ```julia
  EpiBranch.risk_applies(::MyLeakyQuarantine, route) =
      route !== nothing && EpiBranch.INTERVENTION_REMOVAL in route.until
  ```

  Branching processes apply every risk to every contact. Risks from the model
  itself and per-person multipliers apply on every route.
- **Permanent blocks.** [`EpiBranch.standing_block`](@ref) tells a
  continuous-time model that a certain block of yours, once in force for a
  pair, never ends, so it can stop proposing contacts for that pair instead of
  drawing again towards an answer it already has. The `Risk` you return cannot
  say this, because `competing_risk` reads the current state: a block that is
  certain at one proposal may have ended by the next, and both cases return
  the same numbers. Declare it only when the block is permanent. A window with
  no end needs the declaration for the simulation to finish, and without it a
  certain block raises an `ArgumentError` instead of quietly dropping
  transmission that could still happen.

  ```julia
  EpiBranch.standing_block(::MyClosedWard) = true
  ```

  Dropping the pair leaves the infection outcome unchanged, and tracing and
  ring construction read who is in contact with whom, not these proposals. It
  does lose the later contact *events* on that pair: an output counting them
  (exposures a vaccine averted, say, or effort spent on contacts who were not
  infected) needs the draws that would otherwise be skipped. See [Recording
  every contact event](@ref).

### Recording every contact event

A continuous-time model stops proposing contacts for a pair once a
[`EpiBranch.standing_block`](@ref) has settled it for good. The transmission
outcome is unchanged and the two people stay in each other's contacts for
tracing and ring construction. The later contact *events* between them are
lost, though, and an output built to count them needs those draws back.

A [`ContactRecorder`](@ref), passed as the `recorder` of a `ModelSpec`, does
this, through one method:
[`records_contacts`](@ref EpiBranch.records_contacts)`(recorder, parent, contact, state, t)`
says whether the recorder wants the simulation to keep drawing this pair. It is
asked every time a permanent block would end the pair's draws (not only the
first), with `t` the time of the proposal that would be dropped. The default
[`NoContactRecorder`](@ref) answers `false` for every pair, so a model with no
recorder drops the pair at no extra cost.

A sketch:

```julia
struct CountingRecorder <: ContactRecorder
    events::Vector{NTuple{3, Float64}}  # (parent_id, contact_id, t)
end
CountingRecorder() = CountingRecorder(NTuple{3, Float64}[])

function EpiBranch.records_contacts(r::CountingRecorder, parent, contact, state, t)
    push!(r.events, (parent.id, contact.id, t))
    return true
end
```

Usage: `ModelSpec(HouseholdProcess(...); recorder = CountingRecorder())`. Every
later proposal on a permanently blocked pair is logged before the simulation is
told to continue, so `rec.events` ends up with every contact event of every
such pair.

Answering `true` restricts which models can run: drawing again puts the pair
back under the rule for blocked contacts above, so a model whose window never
ends is refused, as if the block had never been declared permanent.

## Likelihood compatibility

The network and household infection likelihoods condition on infection times,
the start and end of each infectious period, which cases were index cases, and
the contact structure, and sum over the possible infectors. They do not include
the probability of the clinical timeline, of who received an intervention, of
population characteristics or of observation. [`progression_loglik`](@ref)
gives the clinical-timeline term separately; the others have no likelihood
function in the package.

`loglikelihood(data, spec)` refuses a model part unless it is declared
compatible with these likelihoods. The built-in parts declare:

| Model part | Fitted exactly by the infection likelihood |
|---|---|
| `clinical_presentation`, no population characteristics | yes |
| Other population characteristics (your own functions) | no, unless declared |
| `Transition`, `Reporting`, `Hospitalisation`, `Recovery`, `Death` | yes |
| `Isolation` | only with `post_isolation_transmission = 0` (complete isolation) |
| `ContactTracing` with `Quarantine` or `FlagOnly` | yes |
| `MassVaccination`, `GroupVaccination` | only with the default `dose_label` |
| `RingVaccination` | only with the default `dose_label`, and no `onward_efficacy` or `post_exposure_efficacy` |
| `Scheduled`, `CapacityConstrained` | as the intervention they hold |
| Interventions and transitions of your own | no, unless declared |

Removal from transmission, including complete isolation, is represented by the
recorded removal times. Partial reduction of transmission, or a susceptibility
multiplier, also changes the rate of infection within the infectious period. A
change to a person's susceptibility is declared through
`susceptibility_components` (below); other effects are not recorded in the
infection data.

A part of your own that only changes the infectious period can declare itself
compatible:

```julia
EpiBranch.infection_likelihood_compatible(::MyWindowRemoval) = true
```

Its `infectious_removal_time` method must describe the removal used in
simulation. The same declaration is available for custom clinical transitions
and for population characteristics written as types. Unknown population
characteristic functions are refused, to be safe. Declaring compatibility
promises that the part's only effects on the rate of infection are those the
recorded infectious periods represent; the package does not inspect what your
functions do.

For other effects, extract the infection data and call
`pairwise_surv_loglik(effective_kernel, data; external_hazard)` directly. The
contact-interval distribution you pass must represent the full rate of
transmission between each pair, including any per-person or intervention
effects, and the external hazard must represent introductions from outside.
This route can still be differentiated with respect to the kernel's
parameters. Extracting the data does not check that a bare contact interval
reproduces the full model.

### Susceptibility effects

A part that changes how susceptible a person is, such as a vaccination, enters
the likelihood through [`EpiBranch.susceptibility_components`](@ref). It is
asked about each susceptible person, given as a [`LayerHost`](@ref) that reads
the infection data's `host_times`, and returns `nothing` when it leaves that
person's rate of infection unchanged. Otherwise it returns `weight => modifier`
pairs, each modifier an [`EpiBranch.HazardScaling`](@ref) that multiplies every
rate of infection the person faces by a factor from a given time on, or
`nothing` for no change. The person's contribution to the likelihood is the
mixture over these components, with the escape from all of their possible
infectors inside each one. A property drawn once per person and never observed,
such as whether a vaccinee responded, then applies to all of that person's
exposures together.

`VaccineEffect` gives one component under `LeakyMode` and two under
`AllOrNothingMode`, starting at the person's immunity time, and every
`AbstractVaccination` answers with its `VaccineEffect`. An intervention of your
own declares its effect the same way, next to the `competing_risk` that applies
it in simulation, and names the per-person times it reads with
[`EpiBranch.susceptibility_host_times`](@ref):

```julia
struct Prophylaxis <: EpiBranch.AbstractIntervention
    reduction::Float64
end

function EpiBranch.susceptibility_components(p::Prophylaxis, host)
    t = get(host.state, :prophylaxis_time, Inf)
    isfinite(t) || return nothing
    return (1 => EpiBranch.HazardScaling(t, 1 - p.reduction),)
end
EpiBranch.susceptibility_host_times(::Prophylaxis) = (:prophylaxis_time,)
EpiBranch.infection_likelihood_compatible(::Prophylaxis) = true
```

`household_infections` and `network_infections` then record
`:prophylaxis_time` for every person, and `loglikelihood(data, spec)` evaluates
the effect. The same object, or any other effect, can be passed to
`pairwise_surv_loglik(kernel, data; susceptibility = effect)` directly, which
is how a candidate efficacy is evaluated when fitting.

### Removals that end and resume

A removal that takes a person out of transmission for a stretch and then lets
them back leaves them infectious on both sides of that stretch. The infectious
period has one end and cannot reopen, so such a removal leaves it alone and
blocks each contact over its own stretches instead, through `competing_risk`.
For the likelihood to agree with the simulation it has to remove the same
stretches from each pair's exposure, so it reads them from the infection data's
`host_times`.

Record each stretch with [`EpiBranch.record_removal!`](@ref EpiBranch.record_removal!)
and name the key you recorded it under with
[`EpiBranch.removal_gap_host_times`](@ref EpiBranch.removal_gap_host_times):

```julia
struct Shielding <: EpiBranch.AbstractIntervention
    duration::Float64
end
const SHIELDING_STRETCHES = :shielding_stretches

function EpiBranch.resolve_individual!(s::Shielding, ind, state)
    t = EpiBranch.onset_time(ind)
    isfinite(t) || return nothing
    EpiBranch.record_removal!(ind, t, t + s.duration; key = SHIELDING_STRETCHES)
    return nothing
end

function EpiBranch.competing_risk(::Shielding, parent, contact, state)
    stretches = EpiBranch.removal_stretches(parent, SHIELDING_STRETCHES)
    isempty(stretches) && return nothing
    return Tuple(
        EpiBranch.Risk(event_time = a, block_probability = 1.0, release_time = b)
            for (a, b) in stretches
    )
end

EpiBranch.removal_gap_host_times(::Shielding) = (SHIELDING_STRETCHES,)
EpiBranch.binding_release(::Shielding) = true
EpiBranch.infection_likelihood_compatible(::Shielding) = true
```

`household_infections` and `network_infections` then record the key and merge
it with every other removal's stretches, and `loglikelihood(data, spec)` leaves
out exactly the days the simulation blocked.

`EpiBranch.binding_release` lets the continuous-time models read the release.
Leaving it out is unsafe where a pair's contact interval allows unboundedly
many contacts within the window, as a truncated generation interval does: one
part that has not declared it stops the simulation reading any release for
that pair, and a model that ran without this intervention then stops with an
error with it. Under an unbounded contact interval leaving it out changes
nothing, so the failure appears only on the models where it matters most.

A removal that sometimes never lets its person back should also define
[`infectious_removal_time`](@ref EpiBranch.infectious_removal_time), returning
the start of such a removal as the built-in interventions do. Its default is
`Inf`, which leaves the infectious period ending with the natural history, and
the likelihood then ends the exposure at the start of the stretch, matching
what the simulation blocked. The continuous-time models read a `Risk`'s
`release_time` only from a part that declares `binding_release`, because
`competing_risk` reads the current state and a block that looks certain at one
proposal may have ended by the next. Recorded stretches are only ever added
to, so their releases are binding. A block that comes and goes with the state,
such as a ward that reopens, is not, and the simulation stops with an error
instead of ending a pair whose contacts could still transmit.

!!! warning "Name the key, or the fit is biased"
    A removal that declares `infection_likelihood_compatible` must name its key
    with `removal_gap_host_times`. If it names none, nothing is recorded, and
    the likelihood fits on the whole exposure while the simulation blocked part
    of it. This biases the fit without any sign of it.

The one part for which naming no key is right is one that holds another
intervention and can withdraw its block partway through a recorded stretch: a
[`Scheduled`](@ref) with an end does that. `InterventionWrapper` instead
narrows the infectious period to the first removal, but only when the
intervention it holds never releases anyone by itself. One that does, such as an
`Isolation` or `Quarantine` with a finite `duration`, stays a per-contact
competing risk, which checks the wrapper's on/off condition again at every
contact. That narrowing belongs to the wrapping type and is not passed to plain interventions, whose default
`infectious_removal_time` is `Inf`.

## Custom clinical transitions: further details

These details concern clinical transitions written outside the package;
[Custom clinical transitions](@ref) shows how to write one.

### Reusing clinical event sampling

A custom clinical transition can call `EpiBranch.transition_time` after reading
the event it starts from. The function checks that the starting time is
finite, evaluates the probability and samples the delay, and returns `nothing`
when the event does not happen (a sketch; `FollowupVisit` is a subtype of
`AbstractClinicalTransition` with `delay` and `probability` fields):

```julia
function EpiBranch.resolve_individual!(visit::FollowupVisit, ind, state)
    time = EpiBranch.transition_time(state.rng, ind, ind.infection_time,
        visit.delay; probability = visit.probability)
    time === nothing || (ind.state[:followup_time] = time)
    return nothing
end
```

`FollowupVisit` writes its own keys and can define `initialise_individual!` to
give them defaults. A terminal transition also defines `is_terminal` and
`terminal_event`, so it takes part in deciding which terminal event comes
first.

A probability you supply always uses one random draw, even when it is zero or
one. Leave out `probability` for an event that always happens, as `Recovery`
does, to avoid that draw. A missing starting event uses no draws. `Transition`
sets its flag before calling its delay function; `Reporting` and
`Hospitalisation` set theirs afterwards.

### The likelihood of a custom transition

To let [`progression_loglik`](@ref) evaluate `FollowupVisit`, add a
[`EpiBranch.transition_loglik`](@ref) method that reads back the same keys: the
log-density of the delay if the event happened, the log-probability of whether
it happened either way, and `0.0` when the starting event never happened. See
the `AntiviralTreatment` example in [A transition that does not end the
case](@ref).

Two cases need more than reading the keys back, and
[`EpiBranch.transition_term`](@ref) handles both: call it for the probability
term instead of reading `probability` yourself, then add the delay density.

The first is an infection ended early. A post-exposure dose undoes every
transition that would have taken effect at or after
[`infection_aborted_time`](@ref EpiBranch.infection_aborted_time), restoring its
flag and clearing its time, which looks the same as a transition that did not
happen. Read that way, a transition with probability one would give `-Inf`, so
it is instead treated as censored at the abort: the probability that it would
have happened no earlier than the abort,
`log1p(-p * cdf(delay, aborted - anchor))`, or `logccdf` when `p` is 1. This
matters most for a transition timed from before onset, since an infection
ended early has no onset to time from.

The second is a probability from [`exclusive_probabilities`](@ref). Its
alternatives share one draw, so each returns the 0 or 1 that draw produced,
while the likelihood needs the probability of the alternative the draw chose.

### Terminal transitions and the end of transmission

A terminal transition also defines [`EpiBranch.terminal_target`](@ref): the
state it writes, known without a person (unlike `terminal_event`, which needs
one to work out the *time*). EpiBranch warns when a progression can reach a
terminal state that no infectious window ends at, so that a case who dies or is
lost to follow-up does not go on infecting people. The check runs for a
fixed-size population, for every `RouteWindow`, and for the network and
household models, and it reads `terminal_target` to know the state exists. A
terminal transition without it is skipped by the check without a warning, so a
case reaching its state keeps transmitting for the rest of the run. `Death` and
`Recovery` define it, as does this terminal transition written outside the
package, for loss to follow-up timed from infection (the version in [Competing clinical
outcomes](@ref) is timed from onset):

```@example reference
using EpiBranch
using Distributions
using StableRNGs

struct LostToFollowUp <: AbstractClinicalTransition
    probability::Float64
    delay::Float64
end
EpiBranch.is_terminal(::LostToFollowUp) = true
EpiBranch.terminal_target(::LostToFollowUp) = :lost
function EpiBranch.resolve_individual!(t::LostToFollowUp, ind, state)
    time = EpiBranch.transition_time(
        state.rng, ind, ind.infection_time, t.delay; probability = t.probability
    )
    time === nothing || (ind.state[:lost_time] = time)
    return nothing
end
function EpiBranch.terminal_event(::LostToFollowUp, individual)
    t = get(individual.state, :lost_time, Inf)
    return isfinite(t) ? (t, :lost) : nothing
end

lost_pool = HomogeneousProcess(;
    transmission_rate = 0.0, population_size = 20,
    until = (:recovered, :died, :isolated, :lost)
)
lost_model = ModelSpec(
    lost_pool;
    progression = [
        Transition(:recovered; from = :infection, delay = 10.0, terminal = true),
        LostToFollowUp(0.6, 5.0),
    ]
)
lost_state = simulate(lost_model; n_initial = 20, rng = StableRNG(7))
count(ind -> get(ind.state, :outcome, :none) == :lost, lost_state.individuals)
```

Each of the 20 cases is lost to follow-up on day 5 with probability 0.6, before
it would recover on day 10, so on average 12 of them end as `:lost`. Leaving `:lost` out of
`lost_pool`'s `until` would still run, with a warning that a case reaching
`:lost` never has its infectious period ended, as for `Death` and `Recovery`
when `until` leaves out `:died` or `:recovered`.

### Event dates for uninfected people

Line lists leave out event dates that depend on an infection that did not
happen. For an event that does not, such as an appointment, declare its
line-list column and that it does not need infection. `Val{:appointment_time}`
lets the method apply to that one key:

```julia
EpiBranch.event_time_metadata(::Val{:appointment_time}) =
    (column = :date_appointment, requires_infection = false)
```

The intervention writes the simulation time to `ind.state[:appointment_time]`.
`linelist(state; infected_only = false)` then includes the date for uninfected
people too. Missing and infinite times stay missing. The default line list,
of infected people only, still includes cases only.

Keys ending in `_time` that have no declaration are taken to need infection.
Other keys stay ordinary columns. The built-in tracing, vaccination and
immunity dates do not need infection; labelled doses use columns such as
`date_vaccination_booster` and `date_immunity_booster`. Isolation output keeps
quarantine dates when a provisional onset time was used during the simulation.

## Proposing, admitting and keeping actions

An intervention that proposes actions (see [Intervention actions](@ref))
defines `EpiBranch.intervention_actions(iv, state, candidates)` and returns
`EpiBranch.InterventionAction(individual, time, effect!)` values.
`effect!(individual, time, state)` records an action once admitted. Proposing
can add people beyond the input, as group vaccination does when it finds every
member of a group in which a case was found. The intervention's generation hook
calls `EpiBranch.apply_actions!`.

`Scheduled` checks the proposed action time before delivery. A condition
function sees that time as `state.max_infection_time`, with the current case
count and generation. `CapacityConstrained` uses the simulation's own clock for
counting admissions. Both orders of the two follow these rules. The budget
counts admissions, including deliveries dated in the future; it is not a count
of doses given per calendar day. An intervention using a resource of your own
defines `capacity_key` and `capacity_time_key` and sets the resource flag only
after successful delivery. An action on a person who already has that flag
passes the capacity check without another charge, though the schedule still
applies. Ring vaccination uses this for protection from an existing dose at a
new exposure, with the current simulation time as its action time.

Recorded protection follows its effect date even when the delivery schedule is
no longer active. Vaccination declares this with
`EpiBranch.persistent_competing_risks(iv) = true`. An intervention of your own
can use the same method when its `competing_risk` reads recorded effects and
returns `nothing` before delivery. The default is `false`, for risks that apply
only while the scheduled policy is active. `Scheduled` and
`CapacityConstrained` pass this declaration on from the intervention they hold.

Use `EpiBranch.action_draw!(sample, individual, key)` to keep a delay or
acceptance draw when the same action is proposed again. Keys identify an action
or visit, so distinct visits need distinct keys. Ring and group delivery store
these draws per policy and person. A refused admission may be considered again
when it is proposed again, but is not queued. An earlier trigger can bring
forward an action not yet admitted, using the same delay. Admission fixes the
recorded date and effect draws, and later triggers do not change completed
actions, with one exception below: on a continuous-time model, a group dose for
a person whose infection is not yet final moves to a trigger found later that
turns out to be earlier than the one the dose was first given from. Dose
prerequisites are checked against the proposed date before admission.

For example, draw one visit time and reuse it if admission is attempted again:

```@example cached_visit
using EpiBranch, Distributions, Random

person = Individual(id = 1)
rng = Xoshiro(42)
draw_time() = rand(rng, Uniform(9.0, 11.0))
first_time = EpiBranch.action_draw!(draw_time, person, :clinic_visit)
next_time = EpiBranch.action_draw!(draw_time, person, :clinic_visit)
first_time == next_time
```

The result is `true`: the second call uses the stored draw. Give a new visit a
different key, and keep the same key when the same visit is proposed again.

On network and household models, supported actions run after a newly final
case is traced. Interventions of your own opt in with
`EpiBranch.continuous_actions(iv) = true`, and must work without the infection
time of a contact whose infection is not yet final. Ring delivery supports an
eligibility window with no end and zero post-exposure efficacy; group delivery
uses known triggering cases. Scheduling and capacity work as on branching
processes. Mass vaccination and the homogeneous pool are not supported here.
With several households, capacity needs `period = Inf` for one budget over the
whole run. Finite periods are refused because each household is simulated
separately, with its clock reset for the next. A single household supports
finite periods.

On continuous-time models, admission affects people whose infection is not yet
final and the current case. Earlier cases and their clinical outcomes are not
revised. An action dated before the current simulation time has expired and is
skipped. Selection and delay functions must use only information available
when the action is proposed. Protection still uses the risks at the time of
each proposal and the recorded delivery and immunity dates.

Cases become final in order of infection, not in order of eligibility, so a
case can be found eligible earlier than the case that infected it: a secondary
case confirmed by a laboratory before its infector, or the first case in a
group that is never confirmed. Group vaccination's trigger for a person whose
infection is not yet final therefore moves earlier whenever a later search
finds an earlier one, keeping the dose at the group's true earliest trigger
instead of the first one found. The dose of a person whose infection is final,
and a dose another vaccination gave, keep their date.

This moving earlier is itself a method rather than a rule written into the
intervention. `EpiBranch.may_revise(iv, prior_trigger, new_trigger)` says
whether a dose already admitted under `prior_trigger` may move to
`new_trigger`. The default is `false` (an admitted dose keeps its date), and
group vaccination defines it to allow a genuine improvement. An intervention of
your own wanting the same behaviour defines this method for its type.
`EpiBranch.is_settled(state, ind)` says whether a person can still be moved:
`true` once a continuous-time model has finished its own round of proposals for
`ind`, `false` for a person whose infection is not yet final (and for the case
being made final, during its own round).

## Rules as types with parameters

Offspring functions take `(rng, individual)` or `(rng, individual, state)`.
When both methods exist, the simulation uses the one with `state`. A generation
time function takes the individual and returns a distribution. These can be
types with parameters as well as functions:

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

The multi-type constructor that takes a contact matrix also accepts such a type
as its distribution family. Distributions and offspring specifications with
their own `draw_offspring` method work as before. Closed-form results need an
offspring distribution with the corresponding analytical methods; a rule that
works for simulation has no closed form.

## All extension points

| To add | You write | Used |
|---|---|---|
| An intervention | A type `<: AbstractIntervention` and hook methods | Each generation |
| Ending an infection early | `EpiBranch.abort_infection!(ind, time)` from an intervention hook | In that hook |
| A vaccination | A type `<: AbstractVaccination` holding a `VaccineEffect`, with `vaccine_effect` and `apply_post_transmission!` | Each generation |
| A vaccine effect mode | A type `<: AbstractEffectMode` and `realised_efficacy`; `realise_prior_dose!` only to change what a dose recorded before the run gets | When a dose is recorded, or when a person is created |
| A start date for an intervention | `Scheduled(iv; start_time = ...)`, with `intervention_time` and `reset!` for `iv` | After each hook |
| A capacity limit | `CapacityConstrained(iv; budget_per_period = ...)`, with `capacity_key` and `capacity_time_key` for `iv` | `apply_post_transmission!` |
| A stopping rule | A type `<: AbstractStoppingRule` and `should_stop` | Each step |
| A contact recorder | A type `<: ContactRecorder` and `records_contacts` | Continuous-time models, each proposal on a permanently blocked pair |
| A terminal clinical transition | A type `<: AbstractClinicalTransition` with `is_terminal`, `terminal_event`, `terminal_target` | When a case is created |
| Population characteristics | A function `(rng, ind) -> nothing` | When a person is created |
| Several population characteristics | `[f1, f2, ...]` | When a person is created |
| Offspring as a function | A function `(rng, ind) -> Int` | Offspring draw |
| Multi-type offspring as a function | A function `(rng, ind) -> Vector{Int}` | Offspring draw |
| An offspring specification | A type with `draw_offspring` and `chain_size_distribution` | Offspring draw and closed-form results |
| A transmission model | A type `<: TransmissionModel` with `generate_offspring` (offspring-driven) or `initialise_state`, `contacts_of` and `gather_by_target` (structure-driven); optionally `single_type_offspring` and accessors | Simulation and closed-form results |
| A transmission route | `RouteWindow(name; from, until, kernel, reach)` on a model that reads routes | Continuous-time models, each case |
| Structured mixing in a closed population | `mixing_by` (a tuple of characteristic names) and `force(group, counts)` for the internal `_sellke_pool!` | Simulation |
| A clinical transition | A type `<: AbstractClinicalTransition` with `initialise_individual!` and `resolve_individual!`; `is_terminal`, `terminal_event` and `terminal_target` if terminal; `transition_loglik` to evaluate it | When a case is created |
| A calendar schedule for a contact interval | A type with `calendar_multiplier`, and `next_calendar_break` or `calendar_shape(::YourSchedule) = SmoothCalendar()` | Simulation and likelihood |
| A pairwise likelihood for a contact structure | A type `<: InfectionLayer` with `contact_structure`; `compile_contact_pairs` and `pairwise_surv_loglik` then apply | Likelihood |
| A grouping of pairwise likelihood terms | A type `<: EpiBranch.PairwiseReduction` with `EpiBranch.ngroups` and `EpiBranch.group`, run with `EpiBranch.pairwise_reduce` | Likelihood |
| The likelihood of the natural history | `progression_loglik(spec, individuals)`; built-in transitions work as they are, one of your own needs `transition_loglik` | Likelihood |
| An observation model | A type `<: ObservationModel` with `observe(base, ::YourObs)` (closed form) and/or `apply_observation!(::YourObs, state, rng)` (simulation) | Closed-form results and fitting |
| Per-cluster information | Either computed into existing `ChainSizes` fields, or a new data type with a `loglikelihood` method that passes each group to `loglikelihood(ChainSizes(sizes; seeds), offspring)` | Likelihood |
