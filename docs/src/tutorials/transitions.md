# Clinical transitions

Where interventions are *policy* applied to a case (isolation, contact
tracing, vaccination), clinical transitions are *biology*: the case
progresses through symptom onset, reporting, possibly hospitalisation,
and ultimately recovery or death. **EpiBranch.jl** treats these as a
composable case-state Markov chain layered on top of the transmission
process.

Transitions share the hook shape of interventions
([`initialise_individual!`](@ref EpiBranch.initialise_individual!),
[`resolve_individual!`](@ref EpiBranch.resolve_individual!)) — but live
under a sibling abstract type, [`AbstractClinicalTransition`](@ref), so
the public API keeps `interventions=` and `transitions=` namespaces
distinct.

## Built-in transitions

Four transitions ship with the package:

- [`Reporting`](@ref) — symptomatic cases are reported with a probability
  after a delay from symptom onset.
- [`Hospitalisation`](@ref) — symptomatic cases are admitted to hospital
  with a probability after a delay from symptom onset.
- [`Death`](@ref) — terminal: probability of dying, delay from onset.
- [`Recovery`](@ref) — terminal: always draws a recovery delay from
  onset for symptomatic cases.

Build the vector explicitly. Each transition takes its own delay
distribution and probability, so there is no shared parameter to worry
about:

```@example transitions
using EpiBranch
using Distributions
using StableRNGs

model = BranchingProcess(Poisson(2.0), Exponential(5.0))
clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

transitions = [
    Reporting(delay = LogNormal(1.0, 0.3)),
    Hospitalisation(delay = LogNormal(2.0, 0.5), probability = 0.2),
    Death(delay = LogNormal(2.5, 0.4), probability = 0.05),
    Recovery(delay = LogNormal(2.0, 0.4)),
]

rng = StableRNG(42)
state = simulate(model;
    attributes = clinical,
    transitions = transitions,
    sim_opts = SimOpts(max_cases = 200),
    rng = rng,
)

ind = state.individuals[end]
println("onset = ", ind.state[:onset_time])
println("reported = ", ind.state[:reported], " at ", ind.state[:reporting_time])
println("outcome = ", ind.state[:outcome], " at ", ind.state[:outcome_time])
```

The gate that decides which cases see clinical transitions is
`isnan(:onset_time)` — cases without a recorded symptom onset are
skipped because they were never clinically observed. For diseases
with asymptomatic cases, use
[`clinical_presentation`](@ref)`(prob_asymptomatic = 0.x)`; it sets
onset to `NaN` for asymptomatic cases and the transitions skip them
naturally. For diseases without an asymptomatic concept, the default
`prob_asymptomatic = 0.0` is enough — or, if you don't need the
`:asymptomatic` flag for anything else, a minimal attributes function
that only sets `:onset_time` works too:

```julia
attributes = (rng, ind) -> ind.state[:onset_time] =
    ind.infection_time + rand(rng, LogNormal(1.5, 0.5))
```

No `:asymptomatic` infrastructure is required by the transitions
themselves.

## Heterogeneity via callables

`probability` and `delay` on every transition accept three shapes:

- a `Real` / `Distribution`: constant or one shared distribution across
  the population.
- a `Function (rng, ind) -> value`: an arbitrary per-individual rule.

The function form covers anything you might otherwise have asked for as
a bespoke field — age-conditional CFRs, vulnerability-dependent
delays, risk-group-specific reporting:

```@example transitions
attrs = compose(
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Uniform(0, 90)),
)

# Age-conditional CFR: 30% for 80+, 2% otherwise.
death_age = Death(
    delay = LogNormal(2.5, 0.4),
    probability = (rng, ind) -> ind.state[:age] >= 80 ? 0.3 : 0.02,
)

# Age-conditional admission delay: faster for under-30s.
hosp_age = Hospitalisation(
    delay = (rng, ind) -> ind.state[:age] < 30 ? 1.0 : 5.0,
    probability = 0.2,
)

rng = StableRNG(42)
state = simulate(model;
    attributes = attrs,
    transitions = [hosp_age, death_age, Recovery(delay = LogNormal(2.0, 0.4))],
    sim_opts = SimOpts(max_cases = 300),
    rng = rng,
)

n_died = count(ind -> ind.state[:outcome] == :died, state.individuals)
n_died_80plus = count(state.individuals) do ind
    ind.state[:outcome] == :died && ind.state[:age] >= 80
end
println("Deaths: $n_died, of which 80+: $n_died_80plus")
```

There is no special-cased age dependence anywhere in the package — the
closure that reads `ind.state[:age]` is the *only* mechanism. The same
idiom carries over to risk groups, comorbidities, or any user-defined
state field set via [`compose`](@ref).

## Gated transitions

Sometimes a transition should only fire when a prerequisite is met
(admit only if reported, give antivirals only if tested, etc.). The
same callable mechanism expresses these gates — return `0.0` from
`probability` when the gate is closed:

```@example transitions
gated_hosp = Hospitalisation(
    delay = LogNormal(2.0, 0.5),
    probability = (rng, ind) -> get(ind.state, :reported, false) ? 0.2 : 0.0,
)

rng = StableRNG(42)
state = simulate(model;
    attributes = clinical,
    transitions = [
        Reporting(delay = LogNormal(1.0, 0.3), probability = 0.5),
        gated_hosp,
    ],
    sim_opts = SimOpts(max_cases = 200),
    rng = rng,
)

# Admitted ⊆ Reported, by construction.
for ind in state.individuals
    ind.state[:admitted] && @assert ind.state[:reported]
end
println("No bespoke `requires_*` field needed — gate lives in the closure.")
```

The same pattern covers composite conditions — admit if reported *and*
not yet vaccinated, fire a transition only after the case is traced,
etc. The closure has access to the full `ind.state` dict so any
combination of upstream events or user-set attributes is reachable.

## Sequential transitions: chaining via `from`

The delay anchor on every built-in transition defaults to
`:onset_time`, but `from` is a kwarg that accepts any `Symbol` (looked
up in `ind.state`) or `Function (ind) -> Real`. That makes chained
timelines first-class: anchor each step on the *previous* event's
time, not always on onset.

Suppose reporting depends on testing — a positive test is what
generates the report, and the reporting delay is measured from the
test, not from symptom onset. Define a `Testing` transition that
writes `:test_time`, then anchor the built-in `Reporting` on it:

```@example transitions
struct Testing <: AbstractClinicalTransition
    delay::Distribution
    sensitivity::Float64
end

EpiBranch.required_fields(::Testing) = [:onset_time]

function EpiBranch.initialise_individual!(::Testing, ind, state)
    ind.state[:tested] = false
    ind.state[:test_time] = Inf
    return nothing
end

function EpiBranch.resolve_individual!(t::Testing, ind, state)
    ot = onset_time(ind); isnan(ot) && return nothing
    rand(state.rng) < t.sensitivity || return nothing
    ind.state[:tested] = true
    ind.state[:test_time] = ot + rand(state.rng, t.delay)
    return nothing
end

testing = Testing(LogNormal(0.5, 0.3), 0.9)
reporting_post_test = Reporting(
    delay = LogNormal(0.0, 0.2),
    from = :test_time,
)

rng = StableRNG(42)
state = simulate(model;
    attributes = clinical,
    transitions = [testing, reporting_post_test],
    sim_opts = SimOpts(max_cases = 100),
    rng = rng,
)

ind = state.individuals[end]
println("onset = ", onset_time(ind))
println("tested = ", ind.state[:tested], " at ", ind.state[:test_time])
println("reported at ", ind.state[:reporting_time])
```

`Reporting` skips cases whose anchor (`:test_time`) is still `Inf`
because `Testing` didn't fire — exactly the same NaN/Inf check that
gates asymptomatic cases under the default `:onset_time` anchor. The
ordering of transitions in the vector matters: `resolve_individual!`
is called in order, so any downstream transition that anchors on an
upstream key needs the upstream transition to come first.

### Anchoring on infection time (no onset modelled)

For diseases where you don't want to model symptom onset at all,
anchor on `ind.infection_time` via the function form. No
`clinical_presentation` is required — `from` as a function bypasses
the validator's `:onset_time` check entirely:

```julia
Reporting(
    delay = LogNormal(2.0, 0.3),
    from = ind -> ind.infection_time,
)
```

The same applies to `Death`, `Recovery`, `Hospitalisation`, and any
user-defined transition that adopts the same convention. The choice
of anchor is per-transition, so mixed timelines are fine: hospitalise
from onset, but draw outcome times from admission.

## Terminal transitions and competing arbitration

[`Death`](@ref) and [`Recovery`](@ref) are terminal — they end the
case. The framework is open: any transition whose
[`is_terminal`](@ref)`(t) == true` and that defines
[`terminal_event`](@ref)`(t, ind) -> (time, label)` participates. After
all transitions resolve for an individual, the engine takes the
earliest terminal candidate across the entire vector and writes
`:outcome` (the label) and `:outcome_time`.

Adding a third terminal state — say "lost to follow-up" — is just
another struct and two methods:

```@example transitions
struct LostToFollowUp <: AbstractClinicalTransition
    delay::Distribution
    probability::Float64
end

EpiBranch.required_fields(::LostToFollowUp) = [:onset_time]
EpiBranch.is_terminal(::LostToFollowUp) = true

function EpiBranch.initialise_individual!(::LostToFollowUp, ind, state)
    ind.state[:lost_candidate_time] = Inf
    return nothing
end

function EpiBranch.resolve_individual!(t::LostToFollowUp, ind, state)
    ot = onset_time(ind)
    isnan(ot) && return nothing
    rand(state.rng) < t.probability || return nothing
    ind.state[:lost_candidate_time] = ot + rand(state.rng, t.delay)
    return nothing
end

function EpiBranch.terminal_event(::LostToFollowUp, ind)
    t = get(ind.state, :lost_candidate_time, Inf)
    return isfinite(t) ? (t, :lost) : nothing
end

# Slot it alongside Death and Recovery — competing arbitration picks
# the earliest.
rng = StableRNG(42)
state = simulate(model;
    attributes = clinical,
    transitions = [
        Death(delay = LogNormal(2.5, 0.4), probability = 0.05),
        Recovery(delay = LogNormal(2.0, 0.4)),
        LostToFollowUp(LogNormal(1.5, 0.5), 0.1),
    ],
    sim_opts = SimOpts(max_cases = 200),
    rng = rng,
)

outcomes = [ind.state[:outcome] for ind in state.individuals if haskey(ind.state, :outcome)]
println("Outcome counts: ",
    (died = count(==(:died), outcomes),
     recovered = count(==(:recovered), outcomes),
     lost = count(==(:lost), outcomes)))
```

No engine changes were needed. The same pattern covers ICU
admission as a sub-state of hospitalisation, treatment-conditional
outcomes, or any disease-specific timeline.

## Writing a non-terminal custom transition

Non-terminal transitions follow the same pattern minus `is_terminal`
and `terminal_event`. They mark milestones on the timeline that other
transitions or downstream observation can read:

```@example transitions
struct AntiviralTreatment <: AbstractClinicalTransition
    delay::Distribution
    probability::Float64
end

EpiBranch.required_fields(::AntiviralTreatment) = [:onset_time]

function EpiBranch.initialise_individual!(::AntiviralTreatment, ind, state)
    ind.state[:treated] = false
    ind.state[:treatment_time] = Inf
    return nothing
end

function EpiBranch.resolve_individual!(t::AntiviralTreatment, ind, state)
    ot = onset_time(ind)
    isnan(ot) && return nothing
    # Only treat reported cases (read upstream state).
    get(ind.state, :reported, false) || return nothing
    rand(state.rng) < t.probability || return nothing
    ind.state[:treated] = true
    ind.state[:treatment_time] = ot + rand(state.rng, t.delay)
    return nothing
end
```

A downstream `Death` transition can read `ind.state[:treated]` in its
own callable probability to encode "treated cases have lower mortality":

```julia
Death(
    delay = LogNormal(2.5, 0.4),
    probability = (rng, ind) -> ind.state[:treated] ? 0.02 : 0.08,
)
```

That's the full extension surface. The combination of (a) shared
`ind.state` dict, (b) callable probability/delay, and (c) optional
terminal arbitration covers everything from simple parallel timelines
to richly conditional disease progression with treatment effects.
