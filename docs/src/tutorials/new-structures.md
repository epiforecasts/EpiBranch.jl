# New transmission structures

This page covers changes to *who can infect whom, and when*: a latent period
before a case becomes infectious, transmission over several routes at once
(community, household, funeral), seasonal transmission, a contact structure the
package lacks, a closed population with structured mixing, and new ways in
which cases are observed. It assumes [Extending EpiBranch](extending.md) and its
[words used in these pages](@ref "Words used in these pages").

## Infectiousness windows

By default a case can transmit from the moment it is infected, with the timing
of transmission given by the generation time. To make transmission start after
a latent period and stop at recovery or death, build the
[`BranchingProcess`](@ref) from one or more `Infectiousness` windows. A window
is a source of secondary cases tied to a stretch of the case's natural history:

```julia
Infectiousness(offspring; from = :infection, until = (), kernel = NoGenerationTime())
```

- `offspring`: how many contacts this source makes (a `Distribution`, or a
  function `(rng, ind) -> n` of the case, giving one count per type in a
  multi-type model).
- `from`: the state at which the window opens. `:infection` (the default) is
  the infection time; any other state `s` is read from the person's
  `<s>_time` value, so `from = :infectious` reads `:infectious_time`. The window
  contributes nothing until that time is set.
- `kernel`: the contact interval, measured from the `from` time. Each contact
  happens at the `from` time plus a draw from `kernel`. `NoGenerationTime()`
  places every contact at the `from` time itself.
- `until`: the states that end the window, each read from `<s>_time` on the
  infector. The earliest of them closes it, and a contact at or after that
  time does not transmit, because the infector was removed first.

The state times come from clinical transitions: `Transition(:state; from,
delay)` records `:state` and `:state_time`. A latent period, an infectious
period and a window fit together like this (a sketch with placeholder
distributions):

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
end of the window is a risk like any intervention's, and isolation, tracing and
vaccination combine with it. Isolation comes from the `Isolation`
intervention, not from a state in `until`.

The default single window, with `from = :infection`, `until = ()` and a
generation-time `kernel`, is the plain branching process.

!!! warning "Do not count the infectious period twice"
    Whether `until` is empty changes what `kernel` means. With `until = ()`
    nothing removes the case, so `kernel` is the generation time. With states
    in `until`, `kernel` is the contact interval, and the generation time
    results from contacts racing against removal. Giving a generation time
    *and* a recovery state in `until` for the same window counts the
    infectious period twice; use one or the other.

A case can have several windows with different timing and ends, for example
community transmission and a separate source at the funeral of a case who
dies:

```julia
process = ModelSpec(BranchingProcess(
        (Infectiousness(NegBin(R, k); from = :infectious, until = (:recovered, :died), kernel = gt),
            Infectiousness(Poisson(λ); from = :died, until = (:buried,), kernel = funeral_kernel)));
    progression = progression)
```

Each window draws its own secondary cases, times them from its own `from`
state and is ended by its own `until` states. Two limits:

- The analytical functions (`single_type_offspring`, the chain-size
  distributions) need a single window. With several windows the number of
  secondary cases is a mixture (community contacts for everyone, plus funeral
  contacts for those who die) with no closed form, so such models are for
  simulation only.
- If no transition records a window's `from` state, the window never opens.
  Building the `ModelSpec`, or simulating the bare process, warns when this
  can be detected.

## Transmission routes

On a `RoutedNetwork`, or a continuous-time model of your own that passes
routes to the simulation, a case can transmit over several routes at once,
each open over a different stretch of its natural history and each ended by
different events: a household route that lasts the whole infectious period, a
community route cut short by isolation, a funeral route. A
[`RouteWindow`](@ref) describes one route:

```julia
RouteWindow(name; from = nothing, until, kernel, reach = name,
            contacts_from = :infection, traceable = 1.0)
```

- `from` is the state at which the route's infectiousness begins. `:infection`
  opens it at infection; any other state opens it at that state's
  `<state>_time`. The default, `nothing`, takes the start the model works out
  from its progression. A route whose `from` state is never reached
  contributes nothing: a case who recovers never opens a funeral route.
- `until` names the states that end the route, which closes at the earliest of
  their times. **A state listed by one route and not another ends only the
  first.** This is how a control measure cuts one route and leaves another.
- `kernel` is the route's contact-interval distribution, measured from when the
  route opens.
- `reach` says whom the route reaches; the model turns it into the route's
  contacts, since only the model knows its own structure.
- `contacts_from` is the state from which the people the route reaches count as
  the case's contacts for tracing. The default, `:infection`, suits a standing
  relationship such as a household. A funeral route sets
  `contacts_from = :died`, so its contacts are traced only if the funeral took
  place before the route was cut. This is separate from `from`: a route whose
  infectiousness starts at onset still reaches the same household from
  infection.
- `traceable` is the probability that a case can name a contact made on the
  route. People can name the people they live with but not strangers they
  stood next to. A household route therefore keeps the default `1.0`, and an
  anonymous community route might use `0.0`. `true` and `false` also work, as `1.0` and
  `0.0`.

### Traceability and contact tracing

Naming and tracing are two steps. A route's `traceable` is the chance that the
case can identify a contact at all. The tracing intervention's own probability
(its `TraceRate`) is the chance that the programme then reaches a contact it
has been told about. A contact is traced only if both succeed, and the
probabilities multiply: with a community route at `traceable = 0.5` and a
tracing probability of `0.8`, 40% of the contacts a case meets only in
the community are traced. Set each probability for what it describes. A limit
on naming belongs in `traceable` alone; counting it again in the tracing
probability would reduce tracing twice.

The model applies `traceable` when it assembles the contacts it passes to
[`trace_contacts!`](@ref EpiBranch.trace_contacts!); tracing interventions
never see the routes. Each model has its own rule for a contact reachable on
several routes. `RoutedNetwork` makes one draw per pair of case and contact,
names the contact with the highest of its routes' probabilities, and traces it
no earlier than the routes it was named on allow. The draws use the
simulation's random number generator, and runs are reproducible.

### Cutting a route by an intervention

Other states that end a route come from the natural history, but removal by an
intervention cannot be read from a single recorded value: complete isolation
takes a case out of transmission, while leaky isolation only reduces it, which
a window cannot express. A route that isolation and quarantine should end lists
[`EpiBranch.INTERVENTION_REMOVAL`](@ref) in its `until`, and the removal time
then comes from each intervention's `infectious_removal_time`. `:isolated` in
an `until` refers to a `Transition(:isolated, ...)` in the natural history
instead.

Self-isolation is then two routes that differ in one entry (a sketch;
`community_adjacency` and `household_adjacency` are lists of who is in contact
with whom):

```julia
community = RouteWindow(:community;
    until = (:recovered, EpiBranch.INTERVENTION_REMOVAL),
    kernel = Exponential(12.0), reach = community_adjacency)
household = RouteWindow(:household; until = (:recovered,),
    kernel = Weibull(1.5, 3.0), reach = household_adjacency)
```

A case who isolates stops transmitting in the community and goes on infecting
the people it lives with until the end of its infectious period.

### R and k keep their meaning

`R` remains the reproduction number a case would have if never removed, and
the realised number depends on which routes were cut and when. The dispersion
`k` remains the dispersion of the number of secondary cases, kept separate
from the length of the infectious period, so that R and k keep their usual
meaning.

### Reading routes in your own model

A model of your own that has routes passes them to the continuous-time
simulation as `(window, targets)` pairs in place of a single
`from`/`until`/`targets`, turning each route's `reach` into a function giving
`(target_id, kernel)` pairs. Passing both `routes` and the single-window form is
an error, because the routes would silently drop the single window's ends. A
model that passes no routes gets one window, ended by intervention removal.

The model also passes `watches`: one tuple of
[`watched_records`](@ref EpiBranch.watched_records) per route, in route order,
taken from each route's own kernel, and `projections`, each route kernel's
`EpiBranch.kernel_projection`, which lets the
simulation check that a kernel's rates move only when a record it watches does:

```julia
EpiBranch.sellke_race!(
    state, members, rng, sim_opts;
    routes = routes, interventions = interventions, seed!,
    watches = Tuple(EpiBranch.watched_records(w.kernel) for (w, _) in routes),
    projections = Tuple(EpiBranch.kernel_projection(w.kernel) for (w, _) in routes),
)
```

A model with one kernel passes a tuple with one entry for each, matching the
single-window form:

```julia
watches = (EpiBranch.watched_records(model.edge_kernel),)
projections = (EpiBranch.kernel_projection(model.edge_kernel),)
```

Without `watches`, every route's rates are treated as fixed for the whole run.
A kernel that reads per-person records then keeps contacts drawn before a
record changed, without any warning if `projections` is left out too; with
`projections`, the simulation stops with an error when a record the kernel
reads changes without being declared.

## Seasonal or time-varying transmission

A [`PairKernel`](@ref)'s `calendar` multiplies its contact rate by a function
of calendar time, for seasonal forcing or a contact rate that changes during
an outbreak. The package provides two schedules: [`Steps`](@ref), which is
piecewise constant, and [`Seasonal`](@ref), which is smooth. A plain callable
`t -> multiplier` works too, and is read as smooth. Any other schedule is a
type with a [`calendar_multiplier`](@ref EpiBranch.calendar_multiplier) method
returning the (non-negative) multiplier at a calendar time, and
[`calendar_shape`](@ref EpiBranch.calendar_shape) says what kind of schedule
it is:

- **Piecewise constant**, the default: also define
  [`next_calendar_break`](@ref EpiBranch.next_calendar_break), the first time
  strictly after `t` at which the multiplier changes (`Inf` if none). The
  cumulative rate is then summed exactly, one constant stretch at a time.
- **Smooth**: declare `calendar_shape(::YourSchedule) = EpiBranch.SmoothCalendar()`,
  and the cumulative rate is integrated numerically.

A schedule of your own is for a shape neither built-in one gives. This one has
two peaks a year:

```julia
struct TwoPeakSeasonal{T <: Real}
    amplitude::T
    first_peak::Float64
end
function EpiBranch.calendar_multiplier(s::TwoPeakSeasonal, t)
    return 1 + s.amplitude * cos(4π * (t - s.first_peak) / 365)
end
EpiBranch.calendar_shape(::TwoPeakSeasonal) = EpiBranch.SmoothCalendar()

kernel = PairKernel(context -> Exponential(4.0); calendar = TwoPeakSeasonal(0.5, 15.0))
```

Simulation and the likelihood read a schedule only through these methods, so
both use the same rate. Give the schedule's fields a type parameter, as
`TwoPeakSeasonal{T}` does for its amplitude, so the likelihood can be
differentiated with respect to them. A worked seasonal example is in
[Covariates and time-varying transmission](covariate-transmission.md).

## Offspring distributions of your own

A function in place of the offspring distribution ([Change how many people
each case infects](@ref)) is enough for simulation. For the closed-form
results as well, write a distribution.

Often an existing one will do. Distributions.jl combines distributions with
`MixtureModel`, `truncated` and others, and [`BranchingProcess`](@ref) accepts
the result. Here 40% of cases infect nobody and the rest follow a negative
binomial (mean 3, dispersion 0.5); `NegativeBinomial(r, p)` is Distributions.jl's
own parameterisation:

```@example structures
using EpiBranch
using Distributions
using StableRNGs

zero_inflated = MixtureModel([Dirac(0), NegativeBinomial(0.5, 0.5 / (0.5 + 3.0))], [0.4, 0.6])
mixture_runs = simulate(BranchingProcess(zero_inflated, Exponential(5.0)), 200;
    max_cases = 200, rng = StableRNG(1))
(mean_R = mean(zero_inflated), contained = containment_probability(mixture_runs))
```

The mean R is 0.6 × 3 = 1.8. A mixture like this works for simulation, but has
no closed-form chain-size distribution.

A new distribution subtypes `Distribution` and defines `Distributions.rand`
(a sketch):

```julia
using Distributions
using Random: AbstractRNG

struct MyOffspring <: Distribution{Univariate, Discrete}
    # ... your parameters
end

function Distributions.rand(rng::AbstractRNG, d::MyOffspring)
    # ... return an integer number of secondary cases
end

model = BranchingProcess(MyOffspring(...), Exponential(5.0))
```

The simulation calls `rand(rng, offspring)`, so any distribution with a `rand`
method works. To get the chain-size distribution and the closed-form
chain-size likelihood as well, add a method of `chain_size_distribution` for
your type, returning a distribution over chain sizes, for example by iterating
the offspring distribution's probability generating function:

```julia
function EpiBranch.chain_size_distribution(d::MyOffspring)
    # return a Distribution over chain sizes
end
```

Simulation works without it; only the closed-form results need it.
[`extinction_probability`](@ref) needs its own method for the type, as
`src/analytical/cluster_mixed.jl` and `src/analytical/multi_type.jl` define.

### Offspring specifications

An offspring specification replaces what `BranchingProcess` draws for each
case with something that is not a single distribution.
`ClusterMixed(build, mixing)` is the built-in one: the offspring parameters
vary from chain to chain, as when R differs between clusters. A new
specification needs:

1. For simulation, a method `draw_offspring(rng, offspring, individual, state)`
   returning the number of secondary cases.
2. For closed-form results (optional but recommended),
   `chain_size_distribution(offspring)` returning the chain-size distribution.
   Without it, the likelihood falls back to simulation.
3. For the threshold and extinction (optional), methods of
   [`reproduction_number`](@ref)`(offspring)` and
   [`extinction_probability`](@ref)`(offspring)`, so the model-level functions
   answer for models built from the type. `src/analytical/cluster_mixed.jl` and
   `src/analytical/multi_type.jl` are examples.
4. For use without a generation time, a one-argument `BranchingProcess`
   constructor that wraps the type in an `Infectiousness` window.
   `BranchingProcess(offspring, generation_time)` already accepts any type.

`src/analytical/cluster_mixed.jl` shows the full pattern, including how
`ClusterMixed` stores each chain's parameters on its index case and passes them
down to every case in the chain through `parent_id`.

## Adding a transmission model

Most models stay within `BranchingProcess` and change the offspring
distribution (a function, `ClusterMixed`, or multi-type). For a transmission
process that is different in kind (density dependence, a contact structure, a
continuous-time model on a fixed population), subtype `TransmissionModel` and
reuse the rest of the package: interventions, natural history, line lists and
fitting.

### What your model must provide

For **simulation** there are two routes, depending on whether your model can
give each case's contacts one case at a time.

An **offspring-driven** model (a branching process and its variants) defines
one method:

- [`generate_offspring`](@ref)`(model, parent, state)`: the number of contacts
  the case `parent` makes this generation (a single number, or one per type for
  a multi-type model). The simulation calls it once per active case, creates
  that many contacts, gives each an infection time from your model's
  `generation_time`, and works out which are infected.

Return the number of contacts the case could infect if no control measures
existed, without looking at whether the case is isolated or vaccinated.
Interventions act afterwards, when the simulation decides which contacts are
infected; that is the only place they affect transmission. The simulation also
runs `initialise_individual!` on the new contacts and the interventions' hooks
on each case's contacts, the clinical transitions and `resolve_individual!` on
each new case, and keeps the running totals (`cumulative_cases`,
`current_generation`, `active_ids`, `extinct`, `max_infection_time`).

A **structure-driven** model has contacts a count cannot describe: a contact
network, or a household or metapopulation model in which a susceptible can be
exposed by several infectious people at once and infections use up a fixed
population. It defines instead:

- [`contacts_of`](@ref)`(model, node, state)`: the people an infectious person
  `node` reaches this generation, as `(contact, infection_time)` pairs. Return
  existing people (on a network), or create new ones with
  [`make_contact!`](@ref). Do not set `:infected` yourself.
- [`collect_exposures`](@ref), returning [`gather_by_target`](@ref): a person
  exposed by several cases in the same generation then has all exposures
  considered together, once.
- [`initialise_state`](@ref EpiBranch.initialise_state), which sets up the
  fixed population with [`new_state`](@ref EpiBranch.new_state),
  [`add_individuals!`](@ref EpiBranch.add_individuals!) and
  [`seed!`](@ref EpiBranch.seed!).

`contacts_of` follows the same rule as `generate_offspring`: return every
potential contact and let the interventions decide infection. If the model has
its own probability of transmission per pair (a per-pair probability on a
network, a coupling between patches), do not filter on it in `contacts_of`.
Return the contact and let the probability decide infection through
[`transmission_risks`](@ref EpiBranch.transmission_risks)`(model)`, which
returns a tuple of risks, each with a `competing_risk` method. The contact is then still
created and passed to the interventions' hooks for its infector, where contact
tracing and ring vaccination reach it, and the probability is weighed together with
susceptibility, infectiousness and interventions. The example below does this.

A model can also run **its own simulation loop**, as the continuous-time
household and network models do, stepping through cases in order of infection
time instead of by generation. Such a model receives the other model parts as
arguments: define [`EpiBranch.simulate_once`](@ref)`(m::MyModel, sim_opts; interventions,
attributes, progression, observation, recorder, rng)`, read the parts from the
arguments, and work out anything you need there from `progression`, such as
the start of the infectious period with
[`EpiBranch.infectious_from`](@ref)`(progression)`. `simulate` calls it once,
or again until the outcome matches a `condition`; the method itself takes no
`condition`. Run the continuous-time simulation with
[`EpiBranch.sellke_race!`](@ref), passing it `sim_opts`: it applies the
`max_time` limit, settles each case as it is infected (its clinical
transitions and the interventions' `resolve_individual!`), and updates the
running totals for you. For contact tracing and ring or group vaccination to
act, the model also declares `EpiBranch.supplies_contacts(::MyModel) = true`
and passes `sellke_race!` a `contacts` function naming each case's contacts;
without them those interventions are reported as having no effect. A loop that does not use the race instead calls
[`resolve_transitions!`](@ref EpiBranch.resolve_transitions!)`(state, individual)`
once per case, after its population characteristics and intervention values
are set, and then each intervention's
[`resolve_individual!`](@ref EpiBranch.resolve_individual!)`(intervention,
individual, state)`. The first runs the clinical transitions and records the
timeline values (`:onset_time`, `:outcome_time`, ...) that the line list, the
likelihoods and interventions such as `Isolation` read. `simulate` and `loglikelihood` on a `ModelSpec` call `simulate_once`
with the spec's parts.

A model whose population splits naturally into groups that can be simulated
independently, as a household model splits into households, defines
[`EpiBranch.race_groups`](@ref)`(model, kernel)` to say how it splits for a
given kernel, and its `simulate_once` runs one race per group. Each
`sellke_race!` call sets `state.extinct` from its own group alone, so such a
loop reads `state.extinct` after each call, keeps whether every group died out
(`extinct &= sellke_race!(...).extinct`), and sets `state.extinct = extinct`
after the last race. `HouseholdProcess` in the `EpiHouseholds` package does
the same through the internal `_sellke_race!`, which skips the per-call
bookkeeping. A new kernel type can define the method for a given model to choose a
different split:

```julia
# One race per household for your kernel, whatever it watches. Typing the
# kernel is what makes this a method of your own rather than a replacement
# for the one `HouseholdProcess` already has for every other kernel.
EpiBranch.race_groups(model::HouseholdProcess, kernel::MyKernel) = model.members
```

For the **analytical functions** that work from the offspring distribution
(`reproduction_number`, `extinction_probability`, `epidemic_probability`,
`probability_contain`, `proportion_transmission`, `chain_size_distribution`),
define one method:

- [`single_type_offspring`](@ref)`(model)`, returning the offspring
  distribution (or anything with a `chain_size_distribution` method). The
  analytical functions then work for your model.

For **likelihoods** of data that do not go through the offspring
distribution, define methods of `loglikelihood` directly.

If your model has values for them, also define `population_size` and
`n_types`; the defaults (`NoPopulation()` and `1`) are fine otherwise. If your
generation-time distribution is not stored in a field called
`generation_time`, define [`model_generation_time`](@ref EpiBranch.model_generation_time)`(m)`
to return it.

Your model describes transmission alone. The natural history (`progression`),
interventions, population characteristics and observation are added by the user
with a [`ModelSpec`](@ref), and an offspring-driven model gets them without
storing or defining anything:
`simulate(ModelSpec(MyModel(...); progression = [...], interventions = [...], attributes = attr))`
applies each one. The clinical transitions are applied to each new contact, as
for `BranchingProcess`.

### Offspring-driven: a minimal model

A minimal offspring-driven model, with the closed-form likelihood of chain
sizes working through `single_type_offspring`:

```@example structures
struct MyModel{O, G} <: TransmissionModel
    offspring::O
    generation_time::G
end

# How many people this case could infect. The package creates them, gives
# each an infection time, and applies interventions and natural history.
EpiBranch.generate_offspring(model::MyModel, parent, state) =
    rand(state.rng, model.offspring)

# Optional: lets the analytical functions and likelihoods use the offspring distribution.
EpiBranch.single_type_offspring(m::MyModel) = m.offspring

my_model = MyModel(NegBin(0.8, 0.5), Exponential(5.0))
loglikelihood(ChainSizes([1, 2, 3, 1, 5]), my_model)
```

The log-likelihood comes from the closed-form chain-size distribution of the
negative binomial (R = 0.8, k = 0.5), with no simulation.

### Structure-driven: patients on a hospital bay

Here is a complete structure-driven model. Patients lie in beds arranged in a
ring around a bay, and an infectious patient can infect only the patients in
the two neighbouring beds, each with probability `p`. The bay has a fixed
number of patients, and a patient can be exposed by both neighbours.

```@example structures
using Random: AbstractRNG

struct HospitalBay <: TransmissionModel
    beds::Int
    p::Float64      # probability that an exposure of a neighbour infects
end

# Patients run out because each is infected at most once, so no
# population-wide susceptible fraction is applied on top.
EpiBranch.population_size(::HospitalBay) = EpiBranch.NoPopulation()

# Create every patient at the start and infect the first `n_initial`.
function EpiBranch.initialise_state(m::HospitalBay, sim_opts::EpiBranch.SimOpts,
        interventions, transitions, attributes, rng::AbstractRNG)
    state = EpiBranch.new_state(m, transitions, attributes, rng)
    EpiBranch.add_individuals!(state, m.beds, interventions)
    EpiBranch.seed!(state, 1:(sim_opts.n_initial), interventions, transitions)
    return state
end

# A patient exposed by both neighbours is considered once.
EpiBranch.collect_exposures(m::HospitalBay, state::EpiBranch.SimulationState) =
    EpiBranch.gather_by_target(m, state)

# The neighbours in the beds either side, each with a potential infection time
# drawn from a generation time with mean 5 days.
function EpiBranch.contacts_of(m::HospitalBay, patient, state::EpiBranch.SimulationState)
    i = patient.id
    neighbours = (i == 1 ? m.beds : i - 1, i == m.beds ? 1 : i + 1)
    result = Tuple{eltype(state.individuals), Float64}[]
    for bed in neighbours
        neighbour = state.individuals[bed]
        EpiBranch.is_infected(neighbour) && continue
        push!(result, (neighbour, patient.infection_time + rand(state.rng, Gamma(2.0, 2.5))))
    end
    return result
end

# The per-exposure probability, as a risk alongside the interventions.
struct BedTransmission
    p::Float64
end
EpiBranch.transmission_risks(m::HospitalBay) = (BedTransmission(m.p),)
EpiBranch.competing_risk(r::BedTransmission, parent, contact, state) =
    Risk(block_probability = 1.0 - r.p)
```

Simulate 200 outbreaks on a bay of 30 patients, each started by one infected
patient, first without interventions and then with isolation one day after
symptom onset on average:

```@example structures
bay = HospitalBay(30, 0.6)
stop = [Extinction(), MaxGenerations(50)]
n_infected(s) = count(is_infected, s.individuals)

no_control = simulate(ModelSpec(bay), 200; n_initial = 1, stopping_rules = stop,
    rng = StableRNG(11))

iso = Isolation(onset_to_isolation_delay = Exponential(1.0), duration = 14.0)
isolating = ModelSpec(bay; interventions = [iso],
    attributes = clinical_presentation(incubation_period = LogNormal(1.0, 0.4)))
with_isolation = simulate(isolating, 200; n_initial = 1, stopping_rules = stop,
    rng = StableRNG(11))

(no_control = mean(n_infected.(no_control)),
 with_isolation = mean(n_infected.(with_isolation)))
```

Isolation, natural history and the line list work on the new model without any
further code, and isolation reduces the mean number of patients infected per
outbreak. The model is a version of the one the package's own tests use for
this route (`test/test_structure_driven.jl`).

### Fitting a structure-driven model to infection times

A structure-driven model simulated in continuous time can use the pairwise
survival likelihood, whose generative model is that continuous-time
simulation. Beyond the infection times, it needs to know who could have
infected whom. Define a type for the infection data that subtypes
[`InfectionLayer`](@ref) and give it a
[`contact_structure`](@ref EpiBranch.contact_structure) method returning either
a vector of group memberships, for groups whose members all mix, or a list of
each person's contacts. [`compile_contact_pairs`](@ref) and
[`pairwise_surv_loglik`](@ref) then work on it with no further methods,
including the per-pair, covariate and community-hazard terms, and
`loglikelihood` needs one method passing on to them (a sketch):

```julia
struct MyInfections{T <: Real} <: InfectionLayer
    contacts::Vector{Vector{Int}}    # contacts[i]: who host i can infect
    infection_time::Vector{T}        # NaN if never infected
    infectious_time::Vector{T}       # the infectious window opens
    removal_time::Vector{T}          # and closes (Inf if still open)
    is_index::Vector{Bool}           # introduced from outside
    obs_end::T                       # community introductions stop
    followup_end::T                  # observation ends (optional; Inf if absent)
    host_times::NamedTuple           # per-host times a live kernel reads (optional; empty if absent)
end
EpiBranch.contact_structure(d::MyInfections) = d.contacts

# MyContactModel: your structure-driven model, with `kernel` and
# `external_hazard` fields
Distributions.loglikelihood(d::MyInfections, m::MyContactModel) =
    pairwise_surv_loglik(m.kernel, d; external_hazard = m.external_hazard)
```

`HouseholdInfections` in `EpiHouseholds` and `NetworkInfections` in
`EpiNetwork` are worked examples. "Layer" here means the infection data for a
contact structure.

The lower-level `PairwiseSurvivalData` holds the rows of the likelihood but not
who the possible infectors were or when they were infected. A kernel given as a
function still receives a row index there. Use an `InfectionLayer` with a
[`PairKernel`](@ref) instead, or supply the information yourself through a
function of the row index.

`pairwise_surv_loglik` and `pairwise_surv_loglik_by_component` sum the same
terms in different groupings: the first gives one total, the second one per
connected component. A different grouping (by stratum, by spatial patch) is an
[`EpiBranch.PairwiseReduction`](@ref) subtype with `EpiBranch.ngroups` and
`EpiBranch.group` methods, run with
[`EpiBranch.pairwise_reduce`](@ref)`(reduction, kernel, data, layout)`:

```julia
struct ByStratum <: EpiBranch.PairwiseReduction
    stratum::Vector{Int}
    nstrata::Int
end
EpiBranch.ngroups(r::ByStratum) = r.nstrata
EpiBranch.group(r::ByStratum, host) = r.stratum[host]

EpiBranch.pairwise_reduce(ByStratum(stratum, nstrata), kernel, data, layout)
```

### Observation on a new model

An observation model is added to your model with a `ModelSpec`, like the other
parts, and your model stores and defines nothing for it:

```julia
simulate(ModelSpec(MyModel(...); observation = MinimumSize(2)))
```

The observation is applied after the simulation, and `loglikelihood(data,
spec)` reads it from the `ModelSpec`, as for `BranchingProcess`. A model with
its own `simulate_once` is the exception: it calls
`EpiBranch.apply_observation!(observation, state, rng)` itself at the end of
the run, as `MixingProcess` does.

## Age- or group-structured mixing in a closed population

The [Homogeneous models](homogeneous.md) tutorial covers `HomogeneousProcess`,
a closed population where everyone mixes with everyone else at the same rate.
Mixing is often uneven: age bands, sex, income groups or spatial patches
contact each other at different rates. [`MixingProcess`](@ref) simulates such a
population in the same way as `HomogeneousProcess`, with no simulation code of
your own. You supply two things:

1. **Which characteristics define the mixing groups**: `mixing_by`, a tuple of
   names of values each person already has (`:age_band`, `:ses`, `:patch`;
   real characteristics set by the model's `attributes`, not an artificial
   group number). A person's group is the tuple of those values. With
   `mixing_by = ()` everyone is in one group, which gives back the homogeneous
   model.
2. **The contact rate between groups**, `rate(infector_group, target_group)`:
   the rate at which one infectious person in the first group contacts one
   person in the second. Homogeneous mixing is `β/N` for every pair. With a
   contact matrix `M[b, a]`, the rate at which one infectious person in group
   `a` contacts the `n[b]` people of group `b` between them, each pair meets at
   rate `M[b, a] / n[b]`.

!!! note "Groups are always tuples"
    A mixing group is the *tuple* of `mixing_by` values, even when there is a
    single characteristic. Under `mixing_by = (:age_band,)` a person in band
    `b` has group `(b,)`, not `b`. Inside `rate`, read the band with
    `group[1]`.

Below is a population in two age bands with an asymmetric contact matrix: the
younger, more socially active band mixes more than the older one. The
`:age_band` value is an ordinary population characteristic:

```@example mixing
using EpiBranch, Distributions, Random

# A closed population of N in two age bands of equal size.
N = 2000
n = [N ÷ 2, N ÷ 2]                     # band sizes
band = (rng, ind) -> (ind.state[:age_band] = ind.id <= n[1] ? 1 : 2)

# M[b, a] is the mean rate at which one infectious person in band a contacts
# the people of band b. Band 1 mixes far more.
M = [3.0 0.5;
     0.5 0.5]

# The rate at which one infectious person in group `from` contacts one person
# in group `to`. A group is a tuple of :age_band values: read the band with [1].
rate = (from, to) -> M[to[1], from[1]] / n[to[1]]

model = ModelSpec(
    MixingProcess(; population_size = N, mixing_by = (:age_band,), rate);
    progression = [Transition(:recovered; from = :infection,
        delay = Exponential(1.0), terminal = true)],
    attributes = band)
state = simulate(model; n_initial = 5, rng = MersenneTwister(1))

attack_rate(b) = count(ind -> ind.state[:age_band] == b && is_infected(ind),
    state.individuals) / n[b]
(band_1 = attack_rate(1), band_2 = attack_rate(2))
```

People in band 1 are contacted more often and have a higher attack rate. To
check the set-up, give every pair the same rate: the two bands should then
behave as a single homogeneous population with the SIR final size. The same
pattern extends to further groups: give people an `:ses` value and pass
`mixing_by = (:age_band, :ses)`, and `rate` then receives `(band, ses)` tuples.
From there you can write any contact structure, such as a full matrix over
every `(band, ses)` combination or one where band and SES contacts multiply
independently.

Because every contact is between two named people, interventions, per-person
susceptibility and infectiousness, and risks that read the infector, such as a
leaky isolation, apply as they do on the homogeneous model. The rate is fixed
for the whole run: a force of infection that is not proportional to the number
infectious cannot be written this way.

## Choosing initial cases in a fixed population

Network and household simulations accept the ids of the people to infect first
through `initial_cases`. How to choose them is up to the calling code:

```@example initial_cases
using EpiBranch, EpiNetwork, Distributions, Random

adjacency = [Int[] for _ in 1:5]
process = NetworkProcess(adjacency, Exponential(2.0))
chosen = [2, 4]
state = simulate(ModelSpec(process); initial_cases = chosen, rng = Xoshiro(42))
findall(is_infected, state.individuals)
```

With no contacts, only people 2 and 4 are infected. The same keyword works with
`RoutedNetwork`, `HouseholdProcess` and repeated or parallel simulations. Ids
refer to the whole population, across households. An empty vector starts with
no infections. The vector is copied and checked for duplicates and for ids
outside the population.

Leaving out `initial_cases` keeps the default choice of index cases. A vector
of ids replaces that rule and cannot be combined with `n_initial`, but it can
be combined with an `external_hazard`: the chosen cases are infected at time
zero and the hazard acts on everyone else from the same moment, so an outbreak
with known index cases can also receive background introductions. When the
choice itself is random, make it in your own code with an explicit random
number generator.

## Adding an observation model

An observation model describes how cases come to be observed. It is a type
subtyping `ObservationModel` with up to two methods:

1. [`observe`](@ref)`(base_distribution, ::YourObservation)`, for closed-form
   results: return the distribution of observed chain sizes, given the
   distribution of true chain sizes. This is often a small new
   `DiscreteUnivariateDistribution`.
2. `apply_observation!(::YourObservation, state, rng)`, for simulation: mark
   the observed cases on a finished simulation, by setting `:reported` on each
   case. Every simulation calls it, and the simulation-based chain-size
   likelihood counts only the cases it marks (`chain_statistics` still
   reports every infected case).

Here chains larger than a cap are never observed, so the observed sizes follow
the true chain-size distribution truncated at the cap and renormalised. This is
truncation (chains above the cap are absent from the data), not censoring
(where they would be recorded as "at least the cap").

```@example structures
struct TruncatedAtSize <: ObservationModel
    cap::Int
end

struct CappedChainSize{D} <: DiscreteUnivariateDistribution
    base::D
    cap::Int
end
Distributions.minimum(::CappedChainSize) = 1
Distributions.maximum(d::CappedChainSize) = d.cap
Distributions.insupport(d::CappedChainSize, n::Integer) = 1 <= n <= d.cap

function Distributions.logpdf(d::CappedChainSize, n::Integer)
    1 <= n <= d.cap || return -Inf
    Z = sum(pdf(d.base, m) for m in 1:d.cap)
    return logpdf(d.base, n) - log(Z)
end

EpiBranch.observe(base, o::TruncatedAtSize) = CappedChainSize(base, o.cap)
# A simulated case is observed if its whole chain is no larger than the cap.
function EpiBranch.apply_observation!(o::TruncatedAtSize, state, rng)
    cases = filter(is_infected, state.individuals)
    chain_size = Dict{Int, Int}()
    for case in cases
        chain_size[case.chain_id] = get(chain_size, case.chain_id, 0) + 1
    end
    for case in cases
        case.state[:reported] = chain_size[case.chain_id] <= o.cap
    end
    return state
end

sizes = ChainSizes([1, 2, 3, 1, 5])
process = BranchingProcess(NegBin(0.8, 0.5))
(all_observed = loglikelihood(sizes, process),
 truncated_at_10 = loglikelihood(sizes, ModelSpec(process; observation = TruncatedAtSize(10))))
```

Allowing for truncation raises the log-likelihood of these small chains,
because chains above 10 cases could never have appeared in the data. No
`loglikelihood` method is needed for the observation: `observe` returns a
distribution and the package evaluates `logpdf` on it.

For the common case of a lower limit instead of an upper cap, the built-in
[`MinimumSize`](@ref) and [`TruncatedChainSize`](@ref) do this already.

### Checking simulation against the closed form

When contributing to EpiBranch, the helper in
`test/testutils/sim_analytical_consistency.jl` compares simulation with a new
observation distribution. It reads the model's observation and filters the
simulated true sizes accordingly; add a method for your observation type to its
`_observe_sizes`:

```julia
# Turn simulated true sizes into observed ones
_observe_sizes(o::TruncatedAtSize, true_sizes, ::AbstractRNG) =
    filter(n -> n <= o.cap, true_sizes)
```

`sim_analytical_consistent(model; n_chains=5000, rng=StableRNG(1))` then
returns simulated and closed-form probabilities of each chain size, which
should agree within sampling error. The helper is part of the test suite, not
of the package.

## Extra information about each cluster

[`ChainSizes`](@ref) has one value per cluster besides its size: `seeds`, the
number of index cases in a cluster with several. It needs nothing further:

```@example clusters
using EpiBranch, Distributions
loglikelihood(ChainSizes([4]; seeds = [2]), NegBin(0.8, 0.3))
```

A second per-cluster choice,
the probability that a cluster has finished growing, is passed when the
likelihood is evaluated, through the `prob_concluded` keyword of
`loglikelihood`, because it is decided at that point and not stored on the
data. Other per-cluster information follows one
of two patterns.

### Compute it in advance

If the information reduces to a value the existing likelihood already uses,
compute it first and pass it in. The rule of Endo et al. that a cluster with no
new case for 7 days is finished looks like censoring in time, but only
computes each cluster's `prob_concluded` (`1.0` for a finished cluster, `0.0`
for an ongoing one). The dot in `is_ongoing.(...)` applies the function to each
element, and `.!` negates each result:

```julia
using Dates
is_ongoing(latest_case, cutoff; window_days = 7) =
    cutoff - latest_case < Day(window_days)

prob_concluded = Float64.(.!is_ongoing.(last_case_dates, cutoff_date))
data = ChainSizes(sizes; seeds = imports_per_cluster)
loglikelihood(data, offspring; prob_concluded = prob_concluded)
```

No new type is needed, and the rule stays with the rest of the analysis.

### Define a new data type

If the likelihood itself needs new information per cluster, define a type for
the data and a `loglikelihood` method for it. Here each cluster belongs to a
strain, patch or group with its own offspring distribution:

```julia
struct MultiTypeChainSizes
    sizes::Vector{Int}
    seeds::Vector{Int}  # index cases per cluster
    type::Vector{Int}   # which strain/patch/group
end

function Distributions.loglikelihood(data::MultiTypeChainSizes,
        offsprings::Vector{<:Distribution})
    total = 0.0
    for (k, offspring) in enumerate(offsprings)
        in_type = data.type .== k
        any(in_type) || continue
        clusters = ChainSizes(data.sizes[in_type]; seeds = data.seeds[in_type])
        total += loglikelihood(clusters, offspring)
    end
    return total
end
```

Each type's clusters go to the existing `loglikelihood(ChainSizes(sizes; seeds), offspring)`,
which handles clusters with several index cases and uses the same closed forms
for `Borel`, `GammaBorel` and `PoissonGammaChainSize` as any other chain-size fit.
