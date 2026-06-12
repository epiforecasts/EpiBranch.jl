# Clinical transitions

Interventions are policy: isolation, contact tracing, vaccination.
Clinical transitions are the case's own progression: symptoms,
reporting, maybe admission, and recovery or death. **EpiBranch.jl**
models them as a Markov chain on case state. The natural history is
part of the model: set it as the [`BranchingProcess`](@ref)'s
`progression`.

Transitions use the same two hooks as interventions
([`initialise_individual!`](@ref EpiBranch.initialise_individual!),
[`resolve_individual!`](@ref EpiBranch.resolve_individual!)) but sit
under their own abstract type, [`AbstractClinicalTransition`](@ref).
This keeps the concerns tidy: the model's `interventions` forcing for
policy, its `progression` for biology.

## Built-in transitions

Four transitions are included in the package:

- [`Reporting`](@ref): reports symptomatic cases after a delay from
  onset, with optional probability below 1.
- [`Hospitalisation`](@ref): admits a fraction of cases after a delay
  from onset.
- [`Death`](@ref): terminal. Cases die with a given probability; delay
  from onset.
- [`Recovery`](@ref): terminal. Always draws a recovery time for
  symptomatic cases.

Build the vector explicitly; each transition has its own delay
distribution and probability, with no shared parameters underneath:

```@example transitions
using EpiBranch
using Distributions
using StableRNGs

clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

progression = [
    Reporting(delay = LogNormal(1.0, 0.3)),
    Hospitalisation(delay = LogNormal(2.0, 0.5), probability = 0.2),
    Death(delay = LogNormal(2.5, 0.4), probability = 0.05),
    Recovery(delay = LogNormal(2.0, 0.4)),
]

model = BranchingProcess(Poisson(2.0), Exponential(5.0); progression = progression)

rng = StableRNG(42)
state = simulate(model;
    attributes = clinical,
    max_cases = 200,
    rng = rng,
)

ind = state.individuals[end]
println("onset = ", ind.state[:onset_time])
println("reported = ", ind.state[:reported], " at ", ind.state[:reporting_time])
println("outcome = ", ind.state[:outcome], " at ", ind.state[:outcome_time])
```

The check that decides which cases see clinical transitions is
`isnan(:onset_time)`. Cases without a recorded onset are skipped;
they were never clinically observed. For diseases with asymptomatic
cases, [`clinical_presentation`](@ref)`(prob_asymptomatic = 0.x)`
sets onset to `NaN` for the asymptomatic fraction and the transitions
skip them. For diseases without an asymptomatic concept, the default
`prob_asymptomatic = 0.0` works. Or, if you don't need the
`:asymptomatic` flag for anything else, a minimal attributes function
that only sets `:onset_time` is enough:

```julia
attributes = (rng, ind) -> ind.state[:onset_time] =
    ind.infection_time + rand(rng, LogNormal(1.5, 0.5))
```

No `:asymptomatic` flag needed.

## Heterogeneity via callables

`probability` and `delay` on every transition accept three shapes:

- `Real` / `Distribution`: a constant or one shared distribution.
- `Function (rng, ind) -> value`: a per-individual rule.

The function form covers age-conditional CFRs, vulnerability-dependent
delays, risk-group-specific reporting, and similar cases.

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

progression = [hosp_age, death_age, Recovery(delay = LogNormal(2.0, 0.4))]
model = BranchingProcess(Poisson(2.0), Exponential(5.0); progression = progression)

rng = StableRNG(42)
state = simulate(model;
    attributes = attrs,
    max_cases = 300,
    rng = rng,
)

n_died = count(ind -> ind.state[:outcome] == :died, state.individuals)
n_died_80plus = count(state.individuals) do ind
    ind.state[:outcome] == :died && ind.state[:age] >= 80
end
println("Deaths: $n_died, of which 80+: $n_died_80plus")
```

Nothing in the package special-cases age. The closure reading
`ind.state[:age]` is the only mechanism, and the same pattern works
for risk groups, comorbidities, or any user-defined state field set
via [`compose`](@ref).

## Gated transitions

Sometimes a transition should only happen when a prerequisite is
met: admit only if reported, treat only if tested. Use the same
callable form for `probability`, returning `0.0` when the gate is
closed:

```@example transitions
gated_hosp = Hospitalisation(
    delay = LogNormal(2.0, 0.5),
    probability = (rng, ind) -> get(ind.state, :reported, false) ? 0.2 : 0.0,
)

progression = [
    Reporting(delay = LogNormal(1.0, 0.3), probability = 0.5),
    gated_hosp,
]
model = BranchingProcess(Poisson(2.0), Exponential(5.0); progression = progression)

rng = StableRNG(42)
state = simulate(model;
    attributes = clinical,
    max_cases = 200,
    rng = rng,
)

# Admitted ⊆ Reported, by construction.
for ind in state.individuals
    ind.state[:admitted] && @assert ind.state[:reported]
end
println("No bespoke `requires_*` field needed. The gate lives in the closure.")
```

Composite conditions work the same way: admit if reported *and* not
vaccinated, only happen after contact tracing, and so on. The
closure sees the full `ind.state` dict, so any upstream event or
user-set attribute is reachable.

## Sequential transitions: chaining via `from`

The delay anchor on every built-in transition defaults to
`:onset_time`, but `from` is a kwarg accepting any `Symbol` (looked up
in `ind.state`) or `Function (ind) -> Real`. That lets you chain
transitions: anchor each step on the previous event's time instead of
always on onset.

Suppose reporting depends on testing. The positive test triggers the
report, and the reporting delay is measured from the test rather
than from onset. Define a `Testing` transition that writes
`:test_time`, then anchor the built-in `Reporting` on it:

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

progression = [testing, reporting_post_test]
model = BranchingProcess(Poisson(2.0), Exponential(5.0); progression = progression)

rng = StableRNG(42)
state = simulate(model;
    attributes = clinical,
    max_cases = 100,
    rng = rng,
)

ind = state.individuals[end]
println("onset = ", onset_time(ind))
println("tested = ", ind.state[:tested], " at ", ind.state[:test_time])
println("reported at ", ind.state[:reporting_time])
```

`Reporting` skips cases whose anchor (`:test_time`) is still `Inf`
because the case was never tested. This is the same NaN/Inf check
that excludes asymptomatic cases under the default `:onset_time`
anchor. Order matters: `resolve_individual!` is called in vector
order, so any downstream transition that reads an upstream key needs
the upstream transition to come first.

### Anchoring on infection time (no onset modelled)

For diseases where you don't want to model symptom onset at all,
anchor on `ind.infection_time` via the function form. `from` as a
function bypasses the validator's `:onset_time` check, so no
`clinical_presentation` is required:

```julia
Reporting(
    delay = LogNormal(2.0, 0.3),
    from = ind -> ind.infection_time,
)
```

`Death`, `Recovery`, `Hospitalisation`, and any user-defined
transition take the same `from` kwarg. The choice is per-transition,
so mixed timelines are fine: hospitalise from onset but draw outcome
times from admission.

## Composing with multi-type models and demographics

Transitions only read and write `ind.state`. They don't know about
model topology. So a multi-type branching process (age strata, risk
groups, spatial patches) and transitions compose without coordination:
each closure reads whichever state keys it needs. The engine sets
`ind.state[:type]` to the type index for multi-type models, and
[`demographics`](@ref) sets `:age` and `:sex` when included in
`attributes`. A transition closure can read any of these.

```@example transitions
attrs_age = compose(
    clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
    demographics(age_distribution = Uniform(0, 90)),
)

# CFR depends on both type and age.
death_type_age = Death(
    delay = LogNormal(2.5, 0.4),
    probability = (rng, ind) -> begin
        base = ind.state[:age] >= 65 ? 0.15 : 0.01
        ind.state[:type] == 1 ? 0.5 * base : base  # children: half the CFR
    end,
)

# Type-conditional admission delay.
hosp_type = Hospitalisation(
    delay = (rng, ind) -> ind.state[:type] == 1 ? 1.0 : 3.0,
    probability = 0.2,
)

progression = [
    hosp_type,
    death_type_age,
    Recovery(delay = LogNormal(2.0, 0.4)),
]

# Two-type model with asymmetric mixing between children and adults.
multitype = BranchingProcess(
    [2.0 0.5; 0.8 1.5],
    R -> NegBin(R, 0.5),
    LogNormal(1.6, 0.5),
    type_labels = ["children", "adults"],
    progression = progression,
)

rng = StableRNG(42)
state = simulate(multitype;
    attributes = attrs_age,
    max_cases = 500,
    rng = rng,
)

n_died_kids = count(state.individuals) do ind
    ind.state[:outcome] == :died && ind.state[:type] == 1
end
n_died_adults = count(state.individuals) do ind
    ind.state[:outcome] == :died && ind.state[:type] == 2
end
println("Deaths. Children: $n_died_kids. Adults: $n_died_adults.")
```

Populations of different vulnerability work the same way:
[`transmission_traits`](@ref) sets per-individual `susceptibility` and
`infectiousness`, demographics or a custom builder sets risk
indicators, and transitions read whatever keys they need. The layers
stack because they share one state dict; nothing inside the package
hard-codes which keys mean what beyond a small set used by the
transmission engine itself (`susceptibility`, `infectiousness`,
`infection_time`).

## Terminal transitions and competing arbitration

[`Death`](@ref) and [`Recovery`](@ref) are terminal: they end the
case. Any transition with `is_terminal(t) == true` and a
`terminal_event(t, ind) -> (time, label)` method participates. After
every transition resolves, the engine picks the earliest terminal
candidate across the vector and writes `:outcome` (the label) and
`:outcome_time`.

Adding a third terminal state, say "lost to follow-up", is another
struct plus two methods:

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

# Slot it alongside Death and Recovery. The competing arbitration picks
# the earliest candidate time across all terminal transitions.
progression = [
    Death(delay = LogNormal(2.5, 0.4), probability = 0.05),
    Recovery(delay = LogNormal(2.0, 0.4)),
    LostToFollowUp(LogNormal(1.5, 0.5), 0.1),
]
model = BranchingProcess(Poisson(2.0), Exponential(5.0); progression = progression)

rng = StableRNG(42)
state = simulate(model;
    attributes = clinical,
    max_cases = 200,
    rng = rng,
)

outcomes = [ind.state[:outcome] for ind in state.individuals if haskey(ind.state, :outcome)]
println("Outcome counts: ",
    (died = count(==(:died), outcomes),
     recovered = count(==(:recovered), outcomes),
     lost = count(==(:lost), outcomes)))
```

No engine changes were needed. The same pattern works for ICU
admission as a sub-state of hospitalisation, treatment-conditional
outcomes, or whatever else your disease timeline needs.

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

That's the whole extension surface. Three ingredients (the shared
`ind.state` dict, callable probability/delay, optional terminal
arbitration) are enough for the cases I've worked through:
independent parallel draws, sequential chains, treatment-conditional
outcomes, capacity-dependent rates.
