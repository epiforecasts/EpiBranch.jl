"""
    simulate(model::TransmissionModel;
             max_cases=10_000, max_generations=100, max_time=nothing,
             n_initial=nothing, initial_cases=nothing, index_types=nothing,
             stopping_rules=nothing,
             rng=Random.default_rng(), condition=nothing, max_attempts=10_000)

Simulate one outbreak and return it as a [`SimulationState`](@ref). Pass the
result to [`linelist`](@ref), [`contacts`](@ref), [`chain_statistics`](@ref)
or [`weekly_incidence`](@ref) to get tables.

`model` is a transmission model such as a [`BranchingProcess`](@ref), or a
[`ModelSpec`](@ref) that adds natural history, interventions, population
characteristics and reporting to it. A model on its own runs with none of
these. Times are in the units of the model's delays; the documentation uses
days throughout.

# Keywords
- `n_initial`: number of index cases (default 1).
- `max_cases`, `max_generations`, `max_time` (days): stop the run once any of
  these is reached. Set one to `nothing` to remove that limit. The run always
  stops when transmission dies out. Every model settles cases in order of
  infection time, so a run stops at exactly `max_cases` cases (or with the
  index cases, if there are more of them), leaves out any infection after
  `max_time`, and draws no contacts for a case at generation
  `max_generations`. The last cases of a run stopped by `max_cases` or
  `max_time` have not made all their contacts, so per-case quantities such as
  offspring counts or realised generation intervals are right-truncated;
  `max_generations` alone leaves every case below the bound complete. On a
  branching process, a contact whose infection was still to be decided when
  the run stopped is left out, unless tracing or an intervention's actions
  had already reached it; it is then kept with no infection time.
- `stopping_rules`: a vector of [`AbstractStoppingRule`](@ref)s for finer
  control over when the run ends. When given, `max_cases`,
  `max_generations` and `max_time` are ignored.
- `rng`: the random number generator; pass a seeded one, such as
  `Xoshiro(42)`, for reproducible runs. Pass [`CommonRandomNumbers`](@ref)
  instead to compare scenarios on shared random numbers: two runs built from
  the same seed then share every draw until one of them changes what happens
  to a particular person or pair.
- `condition`: a range of final outbreak sizes, such as `10:1000`. The
  simulation is repeated until the total number of cases falls inside it (at
  most `max_attempts` tries), for example to leave out outbreaks that died
  out after a few cases. An error is raised if no run qualifies.
- `initial_cases`: for `NetworkProcess`, `RoutedNetwork` and
  `HouseholdProcess` only, the IDs of the people infected at time
  zero (an empty vector is allowed). It replaces the random choice of index
  cases. Give either `initial_cases` or `n_initial`, not both; IDs outside
  the population raise an error. With an `external_hazard`, everyone else can
  still be infected from outside from time zero. Other models raise an error
  when given a vector here.
- `index_types`: for a multi-type [`BranchingProcess`](@ref) only, the types of
  the index cases, in place of the default uniform random choice among
  `1:n_types`. A vector gives each index case's type directly and sets their
  number, so `index_types = [1]` starts the outbreak with a single type-1
  case (give either `index_types` as a vector or `n_initial`, not both); a
  `Distribution` draws each of the `n_initial` index cases' types
  independently, such as `Categorical([0.9, 0.1])` for a 9:1 mix. A type
  outside `1:n_types` raises an error. A single-type model raises an error
  when given either here.

The probability and delay of each step in a case's natural history are set
on the transitions in the `progression` of a [`ModelSpec`](@ref). Each takes
a constant or a function of the random number generator and the individual,
`(rng, ind) -> value`, so rates and delays can depend on age or risk group.

!!! note
    On a model over a fixed population ([`HomogeneousProcess`](@ref),
    [`MixingProcess`](@ref), `NetworkProcess`, `RoutedNetwork` and
    `HouseholdProcess`) the `max_cases` and `max_generations` keyword
    *defaults* have no effect, so a run over a large population is not capped
    unless you ask for it. A `stopping_rules` entry, or a non-default
    `max_cases` or `max_generations`, still ends such a run.

# Examples
```julia
using EpiBranch, Distributions, Random

model = BranchingProcess(NegBin(2.5, 0.16), Gamma(2.5, 2.0))
state = simulate(model; max_cases = 500, rng = Xoshiro(1))
linelist(state)

# keep only outbreaks with between 10 and 500 cases
state = simulate(model; max_cases = 500, condition = 10:500, rng = Xoshiro(1))
```
"""
function simulate(
        model::TransmissionModel;
        n_initial::Union{Int, Nothing} = nothing,
        initial_cases::Union{AbstractVector{<:Integer}, Nothing} = nothing,
        index_types::Union{AbstractVector{<:Integer}, Distribution, Nothing} = nothing,
        max_cases::Union{Int, Nothing} = _DEFAULT_MAX_CASES,
        max_generations::Union{Int, Nothing} = _DEFAULT_MAX_GENERATIONS,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng(),
        condition::Union{UnitRange{Int}, Nothing} = nothing,
        max_attempts::Int = 10_000
    )
    # A bare process is composed with no ModelSpec, so run the checks the spec
    # constructor would otherwise do (a ModelSpec routes here via its own
    # `simulate`, already validated at composition, so it never double-warns).
    _validate_process_windows(model, progression(model))
    _warn_incomplete_terminal_coverage(progression(model))
    sim_opts = _build_sim_opts(
        model, n_initial, initial_cases, index_types, max_cases, max_generations,
        max_time, stopping_rules
    )
    _validate_initial_cases(model, sim_opts)
    _validate_index_types(model, sim_opts)
    return _run_once_or_retry(
        model, sim_opts, condition, max_attempts;
        interventions = interventions(model), attributes = attributes(model),
        progression = progression(model), observation = observation(model),
        recorder = recorder(model), rng
    )
end

# Shared by both `simulate` methods (a bare process and a `ModelSpec`): run
# `simulate_once` once, or repeat it through `_retry_for_condition` when a
# `condition` is set. Pulled out so the two entry points don't each duplicate
# the retry branch.
function _run_once_or_retry(model, sim_opts, condition, max_attempts; kwargs...)
    run = () -> simulate_once(model, sim_opts; kwargs...)
    condition === nothing && return run()
    return _retry_for_condition(run, condition, max_attempts)
end

"""
    simulate_once(model, sim_opts; interventions, attributes, progression,
                  observation, recorder, rng)

Run a single outbreak simulation against a built [`SimOpts`](@ref), with the
modelling layers (`interventions`, `attributes`, `progression`, `observation`,
`recorder`) supplied explicitly rather than read off `model`. `simulate` reads
them off a bare process, or off a [`ModelSpec`](@ref), and wraps this in a
retry loop when a `condition` is given; this method itself takes no
`condition`.

The built-in models run on the continuous-time race and each define a method,
reading the layers off the arguments and deriving any window state they need
(the infectious window's `from`, say, via [`infectious_from`](@ref)) from the
composed `progression`. The default method runs a model written against
[`contacts_of`](@ref) or [`generate_offspring`](@ref) on the same race.
"""
function simulate_once(
        model::TransmissionModel, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng
    )
    layers = (; interventions, attributes, progression, observation, recorder, rng)
    _has_own_method(contacts_of, typeof(model), TransmissionModel) &&
        return _simulate_contacts_of(model, sim_opts; layers...)
    return _simulate_generated(model, sim_opts; layers...)
end

"""
    simulate(model, n::Int; parallel=false, kwargs...)

Simulate `n` independent outbreaks from the same model and return them as a
vector of [`SimulationState`](@ref)s. It takes the same stopping keywords
(`max_cases`, `max_generations`, `max_time`, `stopping_rules`), `n_initial`,
`initial_cases`, `index_types` and `rng` as the single-outbreak method;
`condition` is not available here. The usual next step is
[`containment_probability`](@ref), the share of these outbreaks that died out;
`containment_probability(model, n)` gives it directly, without keeping the
outbreaks.

With `parallel = true` the runs are spread over the CPU threads Julia was
started with (for example `julia --threads 4`). The results are reproducible
for a given seeded `rng`. When Julia has more than one thread, they are not
the same outbreaks as with `parallel = false`.

# Examples
```julia
using EpiBranch, Distributions, Random

model = BranchingProcess(NegBin(2.5, 0.16), Gamma(2.5, 2.0))
states = simulate(model, 1000; max_cases = 5000, rng = Xoshiro(1))
containment_probability(states; max_cases = 5000)
```
"""
simulate(model::TransmissionModel, n::Int; kwargs...) =
    _simulate_each(identity, SimulationState, model, n; kwargs...)

# `simulate(model, n)`, with `each` applied to every run as it finishes and only
# its result (of type `T`) kept, so a summary over `n` runs holds one
# `SimulationState` at a time (one per thread with `parallel`).
function _simulate_each(
        each, ::Type{T}, model::TransmissionModel, n::Int;
        n_initial::Union{Int, Nothing} = nothing,
        initial_cases::Union{AbstractVector{<:Integer}, Nothing} = nothing,
        index_types::Union{AbstractVector{<:Integer}, Distribution, Nothing} = nothing,
        max_cases::Union{Int, Nothing} = _DEFAULT_MAX_CASES,
        max_generations::Union{Int, Nothing} = _DEFAULT_MAX_GENERATIONS,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng(),
        parallel::Bool = false
    ) where {T}
    _validate_process_windows(model, progression(model))
    _warn_incomplete_terminal_coverage(progression(model))
    sim_opts = _build_sim_opts(
        model, n_initial, initial_cases, index_types, max_cases, max_generations,
        max_time, stopping_rules
    )
    _validate_initial_cases(model, sim_opts)
    _validate_index_types(model, sim_opts)
    return _simulate_n(
        each, T, model, n, sim_opts; interventions = interventions(model),
        attributes = attributes(model), progression = progression(model),
        observation = observation(model), recorder = recorder(model), rng,
        parallel
    )
end

_simulate_n(model::TransmissionModel, n::Int, sim_opts::SimOpts; kwargs...) =
    _simulate_n(identity, SimulationState, model, n, sim_opts; kwargs...)

function _simulate_n(
        each, ::Type{T}, model::TransmissionModel, n::Int, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng,
        parallel::Bool = false
    ) where {T}
    if parallel && Threads.nthreads() > 1
        seeds = [rand(rng, UInt64) for _ in 1:n]
        results = Vector{T}(undef, n)
        Threads.@threads for i in 1:n
            local_rng = Random.Xoshiro(seeds[i])
            results[i] = each(
                simulate_once(
                    model, sim_opts; interventions, attributes,
                    progression, observation, recorder, rng = local_rng
                )
            )
        end
        return results
    else
        return [
            each(
                simulate_once(
                    model, sim_opts; interventions, attributes, progression,
                    observation, recorder, rng
                )
            )
                for _ in 1:n
        ]
    end
end

# ── Shared helpers for the race's models ─────────────────────────────
# The race writes each case's state directly, so the outbreak's aggregate
# bookkeeping is reconciled from it afterwards, which the public `sellke_race!`
# wrapper does through `_reconcile_sellke_bookkeeping!` below.

# Whether a model's simulation honours the `max_cases`/`max_generations`
# keyword shortcuts' *defaults* out of the box: a branching process and a model
# of its own do. The built-in models over a fixed population leave these two
# shortcuts' defaults off (`_drop_default_shortcuts`), so a run over a large
# population keeps running to extinction or `max_time` unless a cap is
# explicitly asked for. They override this to `false`.
_honours_termination_controls(::TransmissionModel) = true

# The time at which a structure-driven run ends: the earliest `time_bound`
# among the stopping rules, or `Inf`.
function _max_time(sim_opts)
    return minimum(time_bound(r) for r in sim_opts.stopping_rules; init = Inf)
end

# The generation beyond which a structure-driven run drops a settled case's
# own proposals: the smallest `generation_bound` among the stopping rules, or
# `typemax(Int)`.
function _max_generation(sim_opts)
    return minimum(
        generation_bound(r) for r in sim_opts.stopping_rules; init = typemax(Int)
    )
end

# The stopping rules a structure-driven race must poll `should_stop` for
# after each settled case: those not already fully captured by `time_bound`
# and `generation_bound` alone.
_poll_rules(sim_opts) = filter(!honoured_without_should_stop, sim_opts.stopping_rules)

# Drop the `MaxCases`/`MaxGenerations` rules `SimOpts` built from the
# `max_cases`/`max_generations` keywords, on a model whose race does not
# honour their defaults (`_honours_termination_controls` false), when each was
# left at its default — a cap set through a non-default keyword value, or
# through `stopping_rules` directly, is never touched here.
function _drop_default_shortcuts(model, sim_opts, max_cases, max_generations, stopping_rules)
    (stopping_rules !== nothing || _honours_termination_controls(model)) && return sim_opts
    rules = sim_opts.stopping_rules
    max_cases == _DEFAULT_MAX_CASES &&
        (rules = filter(r -> !(r isa MaxCases && r.n == max_cases), rules))
    max_generations == _DEFAULT_MAX_GENERATIONS &&
        (rules = filter(r -> !(r isa MaxGenerations && r.n == max_generations), rules))
    return SimOpts(sim_opts.n_initial, sim_opts.initial_cases, sim_opts.index_types, rules)
end

# Shared by all four `simulate` entry points (a bare process or a `ModelSpec`,
# each for one outbreak or `n`): build the `SimOpts` the termination keywords
# describe, with the defaults dropped per `_drop_default_shortcuts`.
function _build_sim_opts(
        model, n_initial, initial_cases, index_types, max_cases, max_generations,
        max_time, stopping_rules
    )
    sim_opts = SimOpts(;
        n_initial, initial_cases, index_types, max_cases, max_generations, max_time,
        stopping_rules
    )
    return _drop_default_shortcuts(model, sim_opts, max_cases, max_generations, stopping_rules)
end

"""
    _retry_for_condition(run, condition, max_attempts)

Call `run()` (one simulation) until its `cumulative_cases` fall in `condition`,
up to `max_attempts` times; error if none does.
"""
function _retry_for_condition(run, condition, max_attempts)
    for _ in 1:max_attempts
        state = run()
        state.cumulative_cases in condition && return state
    end
    throw(
        ErrorException(
            "No simulation produced an outbreak of size $condition within $max_attempts attempts"
        )
    )
end

"""
    _reconcile_sellke_bookkeeping!(state, extinct) -> state

Set `cumulative_cases`, `max_infection_time` and `current_generation` (the
latest generation reached) from the per-individual state a
continuous-time (Sellke) loop writes directly, and `extinct` to the given
`extinct`. `extinct` is whether the Sellke loop(s) that built `state` ran
until no candidate infection remained, rather than being cut off at
`max_time` with candidates still pending.
"""
function _reconcile_sellke_bookkeeping!(state::SimulationState, extinct::Bool)
    cases = 0
    max_time = 0.0
    max_generation = 0
    for ind in state.individuals
        get(ind.state, :infected, false) || continue
        cases += 1
        max_time = max(max_time, ind.infection_time)
        max_generation = max(max_generation, ind.generation)
    end
    state.cumulative_cases = cases
    state.max_infection_time = max_time
    state.current_generation = max_generation
    state.extinct = extinct
    return state
end

"""
    contacts_of(model, node, state) -> iterable of (contact, infection_time)

The people the case `node` exposes, each with the time they would be infected,
as `(contact, infection_time)` pairs. The simulation asks once for each case,
when the case's infection is decided.

Only needed when writing a new transmission model that spreads over a fixed
set of people, such as a contact network (the case's network neighbours) or
households (the other household members). A contact may also be a new person,
created with [`make_contact!`](@ref). Someone exposed by several cases is
infected at most once, at the time of the earliest exposure that transmits; an
exposure that is blocked, by an intervention or by the contact's
susceptibility, is not tried again later. A ring of contact tracing reaches the
contacts of a traced person who has not settled only if the model declares
[`reads_settled_state`](@ref EpiBranch.reads_settled_state)`(::YourModel) = false`.

Models in which every contact is a new person, such as a branching process,
define [`generate_offspring`](@ref) instead. In both cases the simulation
applies interventions, natural history and the infection decision itself.
"""
function contacts_of(model::TransmissionModel, parent, state::SimulationState)
    throw(
        ArgumentError(
            "$(typeof(model)) defines no contacts_of method. A structure-driven " *
                "model must implement contacts_of(model, parent, state) returning " *
                "(contact, infection_time) pairs; see the EpiNetwork subpackage for a " *
                "worked example."
        )
    )
end

"""
    model_generation_time(model)

The generation time a model written against [`generate_offspring`](@ref) uses
to time each case's contacts. Returns the model's
`generation_time` field. A new model that times its contacts another way (for
example with a separate delay per transmission route) defines this function
for its own type instead of having a `generation_time` field.
"""
model_generation_time(model::TransmissionModel) = model.generation_time

# Whether an offspring draw has at least one contact to time: a count
# (single-type) or a count per type (multi-type).
_has_contacts(n::Int) = n > 0
_has_contacts(counts::Vector{Int}) = any(>(0), counts)

# Wraps the per-contact offset an `OffspringTimes` already gives, letting
# `offspring_route.jl` pass it here in place of a kernel without
# `_draw_contact!` branching on where the offset came from. It is kept
# distinct from a bare `Real` because a kernel that mistakenly returns a
# number (one must return a `Distribution`) should still raise a
# `MethodError` instead of being silently treated as a fixed offset.
struct _FixedOffset{T <: Real}
    offset::T
end

# A contact's infection time is the window's `from`-state time plus a draw
# from its kernel (the contact interval measured from `from`), or the time
# an `OffspringTimes` already gave.
_window_infection_time(::NoGenerationTime, from_t::Real, state) = from_t
_window_infection_time(offset::_FixedOffset, from_t::Real, state) = from_t + offset.offset
function _window_infection_time(kernel::Distribution, from_t::Real, state)
    return from_t + rand(state.rng, kernel)
end
# A window whose `calendar` scales its kernel draws from the resulting
# calendar-scaled kernel directly, exactly as a `PairKernel` with a calendar
# does on the structure-driven routes.
function _window_infection_time(kernel::_CalendarScaledKernel, from_t::Real, state)
    return from_t + rand(state.rng, kernel)
end

# Tag a contact with its window's `until` states so `WindowCensor` can
# block it after the infector is removed. Skipped for windows with no
# `until` (the default), so single-window models write no extra state.
_tag_window!(contact, until::Tuple{}) = nothing
_tag_window!(contact, until) = (contact.state[:censor_until] = until; nothing)

# Offer `case`'s own infector to interventions tracing backward, as the sole
# contact in `trace_contacts!`'s usual four-argument form. An index case or a
# community introduction has no infector to offer. The race calls it once a
# case is settled.
function _trace_source!(
        state::SimulationState, case, interventions,
        tracing_source = any(traces_source, interventions)
    )
    tracing_source || return nothing
    1 <= case.parent_id <= length(state.individuals) || return nothing
    infector = state.individuals[case.parent_id]
    contacts = (infector,)
    for intervention in interventions
        traces_source(intervention) || continue
        trace_contacts!(intervention, state, case, contacts)
    end
    return nothing
end

"""Step 3 of a generation: decide which exposed people are infected, and
record the new cases. An exposure need not lead to infection. The infector's
infectiousness, the contact's susceptibility, anything the model adds and any
interventions can each prevent it, and infection follows only if none of them
does. A person exposed by several cases is infected if any of the exposures
transmits, at the time of the earliest one that does. When someone who was
infected before is reinfected (possible only once their immunity has waned,
see [`HostImmunity`](@ref EpiBranch.HostImmunity)), the earlier infection is
kept as a past episode; see [`close_episode!`](@ref)."""

"""
    _settle_case!(state, case, interventions, count)

Run the hooks that settle one case on the continuous-time race, once its
infection time is fixed and before it transmits: its clinical transitions
([`resolve_transitions!`](@ref)), then each intervention's
[`resolve_individual!`](@ref), with the state at the case as
[`_at_case!`](@ref) sets it.
"""
function _settle_case!(state::SimulationState, case, interventions, count)
    _at_case!(state, case, count)
    resolve_transitions!(state, case)
    _resolve_interventions!(state, case, interventions)
    return nothing
end

"""
    _at_case!(state, case, count)

Set the state a case's settling hooks read: the clock
(`state.max_infection_time`) at the case's infection time,
`state.cumulative_cases` to `count` and `state.current_generation` to the case's
generation.
"""
function _at_case!(state::SimulationState, case, count)
    state.max_infection_time = case.infection_time
    state.cumulative_cases = count
    state.current_generation = case.generation
    return nothing
end

function _resolve_interventions!(state::SimulationState, case, interventions)
    # A reinfection (`case.episodes` already holds the earlier one, closed
    # before this call) gets a fresh `initialise_individual!` first. That
    # keeps per-episode draws such as `Isolation`'s `:test_positive` specific
    # to the new episode, rather than left over from the one before it.
    if !isempty(case.episodes)
        for intervention in interventions
            initialise_individual!(intervention, case, state)
        end
    end
    for intervention in interventions
        resolve_individual!(intervention, case, state)
    end
    return nothing
end

# ── Internal helpers ───────────────────────────────────────────────

# Discard an abort that the resolved infection does not bear out: the
# individual escaped infection, or its infection started at or after the abort
# and so is a later infection the abort never ended. The onset the abort
# suppressed comes back. Without this, an abort recorded on a pre-created node
# that escapes one exposure would end the infection it gets in a later
# generation.
function _drop_stale_abort!(ind::Individual)
    _infection_aborted(ind) || return nothing
    is_infected(ind) && ind.infection_time < infection_aborted_time(ind) &&
        return nothing
    delete!(ind.state, :infection_aborted_time)
    _set_onset_from_incubation!(ind)
    return nothing
end

# ── Population-building helpers for a model's `initialise_state` ──────
# A model defines `initialise_state` to set up its starting population.
# These three helpers carry the shared boilerplate so a model never
# touches the `SimulationState` constructor or the engine's bookkeeping
# fields directly: `new_state` opens an empty state, `add_individuals!`
# builds its members, `seed!` infects the index cases.

# The real element type carrying timing and hazard values through a run, read
# from the model's timing parameters. Float64 unless the model was built with a
# dual (or other Real) parameter type — e.g. under ForwardDiff — in which case
# the whole simulation carries that type so gradients flow through the timing.
_time_type(::TransmissionModel) = Float64
_kernel_time_type(::NoGenerationTime) = Float64
_kernel_time_type(k::Distribution) = float(Distributions.partype(k))
_kernel_time_type(::Any) = Float64
function _time_type(m::BranchingProcess)
    return mapreduce(
        w -> _kernel_time_type(w.kernel), promote_type, m.infectiousness;
        init = Float64
    )
end

"""
    new_state(model, transitions, attributes, rng) -> SimulationState

An empty outbreak for `model`, before anyone exists or is infected: a
[`SimulationState`](@ref) at generation 0 holding the model's population
size, the population characteristics (`attributes`) and the natural-history
`transitions`. Used when writing a new transmission model: its
`initialise_state` starts from this and adds people with
[`add_individuals!`](@ref) and index cases with [`seed!`](@ref).
"""
function new_state(
        model::TransmissionModel, transitions, attributes,
        rng::AbstractRNG
    )
    T = _time_type(model)
    return SimulationState(
        Individual{T}[], Int[], 0, rng, 0, false,
        population_size(model), zero(T), _fresh_attributes(attributes),
        convert(Vector{AbstractClinicalTransition}, transitions)
    )
end

"""
    add_individuals!(state, n, interventions; n_types = 1, setup = (ind, i) -> nothing,
                     infection_time = NaN, types = nothing)

Add `n` people to the outbreak's population and return them. Used by a new
transmission model's `initialise_state` to build its population.

Each person starts with the given `infection_time`: `NaN` by default, since
nobody is infected yet. [`seed!`](@ref) sets the index cases' infection time,
and the model sets everyone else's if they are infected later. Pass
`infection_time = 0` when everyone added is an index case, so that population
characteristics drawn at creation see the time of their infection.

For each person, a type is drawn first for multi-type models, then the
population characteristics are drawn, then `setup(ind, i)` runs (to record
model-specific information such as a network node or household id), and
finally each intervention sets up its own information on that person. By
default each person's type is drawn uniformly from `1:n_types`; pass `types`
(a vector giving the `i`-th person's type, or a `Distribution` to draw each
one from independently) to choose otherwise, as [`simulate`](@ref)'s
`index_types` does for the index cases.
"""
function add_individuals!(
        state::SimulationState, n::Integer, interventions;
        n_types::Integer = 1, setup = (ind, i) -> nothing, infection_time::Real = NaN,
        types = nothing
    )
    base = length(state.individuals)
    added = eltype(state.individuals)[]
    for i in 1:n
        type_idx = n_types > 1 ? _draw_type(state.rng, types, i, n_types) : NoTypeLabels()
        ind = _create_individual(
            state, 0, base + i, base + i, infection_time; type_idx
        )
        setup(ind, i)
        for intervention in interventions
            initialise_individual!(intervention, ind, state)
        end
        push!(state.individuals, ind)
        push!(added, ind)
    end
    return added
end

"""The `i`-th person's type, out of `n_types`: uniform at random by default,
the `i`-th entry of `types` when it is a vector, or an independent draw from
`types` when it is a `Distribution` (checked against `1:n_types` and for
integrality, since a `Distribution`'s support cannot be checked upfront in
general)."""
_draw_type(rng, ::Nothing, i, n_types) = rand(rng, 1:n_types)
_draw_type(rng, types::AbstractVector, i, n_types) = types[i]
function _draw_type(rng, types::Distribution, i, n_types)
    t = rand(rng, types)
    isinteger(t) || throw(ArgumentError("drew non-integer type $t"))
    1 <= t <= n_types || throw(
        ArgumentError("drew type $t, which is outside 1:$n_types")
    )
    return Int(t)
end

"""
    seed!(state, ids, interventions, transitions) -> state

Make the people with the given `ids` the index cases of the outbreak,
infected at time 0. Used by a new transmission model's `initialise_state`
after [`add_individuals!`](@ref).

It sets their infection time to 0, marks them infected, sets their symptom
onset from any incubation period, and sets the outbreak's case count and list
of infectious cases. It also checks, on the first index case, that the
information the interventions and natural history need is present, so a
missing population characteristic gives a clear error.
"""
function seed!(state::SimulationState, ids, interventions, transitions)
    T = _timetype(state)
    for id in ids
        ind = state.individuals[id]
        ind.infection_time = zero(T)
        ind.state[:infected] = true
        _set_onset_from_incubation!(ind)
    end
    if !isempty(ids)
        first_ind = state.individuals[first(ids)]
        _validate_required_fields(first_ind, interventions)
        # Building the probe costs a copy of the case's state on every outbreak,
        # so it is built only when a transition has a field to check.
        if any(t -> !isempty(required_fields(t)), transitions)
            _validate_required_fields(_transition_probe(first_ind, transitions, state), transitions)
        end
    end
    state.cumulative_cases = length(ids)
    state.active_ids = collect(ids)
    state.extinct = isempty(ids)
    return state
end

# A throwaway individual that holds the first seed case's attribute-derived
# state plus every transition's own `initialise_individual!` keys. A `from`
# anchored on another transition's output is not yet written on any real
# individual at seed time, because the transitions themselves haven't run
# yet. `_validate_required_fields` is checked against this probe instead,
# which makes every step's own keys visible without running any step for
# real.
function _transition_probe(first_ind, transitions, state)
    probe = Individual(;
        id = first_ind.id, infection_time = first_ind.infection_time,
        state = copy(first_ind.state)
    )
    for transition in transitions
        initialise_individual!(transition, probe, state)
    end
    return probe
end

"""
    initialise_state(model, sim_opts, interventions, transitions, attributes, rng) -> SimulationState

Set up the outbreak at time 0, before any transmission: the population and
its index cases, as a [`SimulationState`](@ref). The default, used by
branching processes, creates `sim_opts.n_initial` index cases. A model with a
fixed population that is used up as people are infected (a network,
households) defines its own method, usually by building the population with
[`new_state`](@ref) and [`add_individuals!`](@ref) and infecting the index
cases with [`seed!`](@ref).
"""
function initialise_state(
        model::TransmissionModel, sim_opts::SimOpts,
        interventions, transitions, attributes, rng::AbstractRNG
    )
    state = new_state(model, transitions, attributes, rng)
    # Every individual created here is an index case, so attributes that read
    # the infection time at creation must see the time of 0 they are seeded at.
    add_individuals!(
        state, sim_opts.n_initial, interventions;
        n_types = n_types(model), infection_time = 0, types = sim_opts.index_types
    )
    seed!(state, 1:(sim_opts.n_initial), interventions, transitions)
    return state
end

# The rules a run has by default are tested for by type so their calls compile
# statically; any other rule goes through dispatch as usual. Asked once per
# generation, so on many small outbreaks a dynamic call here is a visible
# share of the run.
@inline function _should_stop(rule, state)
    rule isa Extinction && return should_stop(rule, state)
    rule isa MaxCases && return should_stop(rule, state)
    rule isa MaxGenerations && return should_stop(rule, state)
    return should_stop(rule, state)
end

"""
    susceptible_fraction(state::SimulationState, extra_infected::Int = 0) -> Float64

Share of the population still susceptible at this point of the outbreak:
`1.0` when the population is unbounded (no `population_size`), otherwise one
minus the share already infected. `extra_infected` counts people infected in
the current generation who are not yet in the case count.

For extension authors: a package can add its own kind of structured
population by defining a new type for `population_size` and a method of this
function for it.
"""
function susceptible_fraction(
        state::SimulationState{<:Any, <:Any, NoPopulation},
        extra_infected::Int = 0
    )
    return 1.0
end

function susceptible_fraction(
        state::SimulationState{<:Any, <:Any, Int},
        extra_infected::Int = 0
    )
    n_susceptible = state.population_size - state.cumulative_cases - extra_infected
    n_susceptible <= 0 && return 0.0
    return n_susceptible / state.population_size
end

"""Create a new person, give them their type (if any) and draw their
population characteristics. They start as not infected; only the infection
decision later in the generation marks them infected. Interventions set up
their information on the person later in the generation, so creating a
contact (in `make_contact!` or a model's [`contacts_of`](@ref)) involves no
intervention.

`type_idx` is set before the population characteristics are drawn, so an
attributes function sees each person's actual type rather than the type-1
default every new `IndividualState` starts with.
"""
function _create_individual(
        state::SimulationState, parent_id::Int,
        chain_id::Int, next_id::Int, inf_time::Real;
        type_idx::Union{Int, NoTypeLabels} = NoTypeLabels()
    )
    T = _timetype(state)
    s = IndividualState{T}()
    s[:infected] = false

    # Build `Individual{T}` directly (not the keyword constructor) so the type
    # matches `state.individuals`, whatever `T` is: seed cases pass a plain
    # `Float64` infection time but must still land as `Individual{T}` under AD.
    ind = Individual{T}(
        next_id, parent_id,
        state.current_generation + (parent_id == 0 ? 0 : 1),
        chain_id, convert(T, inf_time), one(T), one(T), Int[], s,
        InfectionEpisode{T}[]
    )

    _set_type!(ind, type_idx)
    _apply_attributes!(state.attributes, state.rng, ind)

    return ind
end

_set_type!(contact, ::NoTypeLabels) = nothing
_set_type!(contact, idx::Int) = (contact.state[:type] = idx)

"""
    make_contact!(state, parent, infection_time; type_idx = NoTypeLabels())

Add one new contact of the case `parent`, exposed at `infection_time` (days),
to the outbreak, and return it. Its population characteristics are drawn
when it is created.

Only needed when writing a new transmission model. The simulation calls it
for every contact drawn from a branching process's offspring distribution, or
from [`generate_offspring`](@ref) in a model that defines it. A model that
creates new people inside its [`contacts_of`](@ref) calls it directly,
returning each contact with its infection time. A model whose contacts are
all new and share one generation time is simpler written as
`generate_offspring`; `contacts_of` earns its keep where the model times each
contact itself, as here, where every route out of a case has its own delay:

```julia
function EpiBranch.contacts_of(m::MyModel, parent, state::SimulationState)
    map(m.routes) do route
        t = parent.infection_time + rand(state.rng, route.delay)
        (make_contact!(state, parent, t), t)
    end
end
```

The simulation does everything else: it applies interventions to the case
and its contacts, decides whether each contact is infected, draws the natural
history of new cases and updates the case counts.
"""
function make_contact!(
        state::SimulationState, parent::Individual,
        infection_time::Real;
        type_idx::Union{Int, NoTypeLabels} = NoTypeLabels()
    )
    next_id = length(state.individuals) + 1
    contact = _create_individual(
        state, parent.id, parent.chain_id,
        next_id, infection_time; type_idx
    )
    push!(parent.secondary_case_ids, next_id)
    push!(state.individuals, contact)
    return contact
end

"""
    resolve_transitions!(state, individual)

Draw a case's natural history: the time of each step in the model's
`progression` (becoming infectious, symptom onset, hospitalisation, outcome)
and which outcome it reaches first. The results are recorded on
`individual.state` (`:infectious_time`, `:onset_time`, `:outcome`,
`:outcome_time` and whatever else each step records).

The built-in simulation calls this for every new case. Only a new
transmission model that runs its own simulation loop needs to call it, once
per case, after the case's population characteristics and intervention
information are set. The steps come from the model's `progression`, placed on
the outbreak when it is built with [`new_state`](@ref EpiBranch.new_state).

If the infection was stopped before symptom onset (see
[`abort_infection!`](@ref EpiBranch.abort_infection!)), its natural history
ends at that time. Steps that take effect before then stand; any step that
would take effect at or after it is undone, and steps that follow from an
undone one do not happen. Undoing restores each entry a step set on
`individual.state`, so a transition must record its results by setting
entries there rather than by changing a vector or dictionary it finds there.
"""
function resolve_transitions!(state::SimulationState, individual)
    transitions = state.transitions
    isempty(transitions) && return nothing
    for transition in transitions
        initialise_individual!(transition, individual, state)
    end
    # Branch once per case: a check inside the loop measurably slows every
    # progression model, although almost no case is aborted.
    if _infection_aborted(individual)
        _resolve_before_abort!(
            transitions, individual, state, infection_aborted_time(individual)
        )
    else
        for transition in transitions
            resolve_individual!(transition, individual, state)
        end
    end
    _finalise_terminal!(individual, transitions)
    return nothing
end

# Resolve the transitions of an aborted infection in order, undoing each one
# that takes effect at or after the abort. Transitions record when they happen
# under `_time` keys (`:hospitalised_time`, `:admission_time`,
# `:death_candidate_time`, ...), so reading those keys applies the same check to
# every transition, built-in or user-defined, whatever state it is timed from.
#
# `kept` holds the state as the transitions that stood left it: copied once per
# case, then brought up to date with only the keys each such transition
# changed.
function _resolve_before_abort!(transitions, individual, state, aborted_t)
    kept = copy(individual.state)
    for transition in transitions
        resolve_individual!(transition, individual, state)
        if _writes_time_from(individual.state, kept, aborted_t)
            # The uniform a group of siblings shares belongs to the group
            # rather than to whichever of them first needed it, so it joins the
            # state the undo restores: the next sibling then reads the same
            # value instead of drawing a fresh one, and the group still
            # partitions a case the abort cut short.
            for key in _shared_draw_keys(transition)
                haskey(individual.state, key) && (kept[key] = individual.state[key])
            end
            empty!(individual.state)
            merge!(individual.state, kept)
        else
            _catch_up!(kept, individual.state)
        end
    end
    return nothing
end

# Marks a key that was absent before a transition ran. A private instance
# compares unequal to anything a transition could store.
struct _Absent end
const _ABSENT = _Absent()

function _writes_time_from(after, before, t)
    for (key, value) in after
        # Identity first: it needs no dispatch on the stored value and rules
        # out almost every key.
        get(before, key, _ABSENT) === value && continue
        value isa Real && isfinite(value) && value >= t || continue
        isequal(get(before, key, nothing), value) && continue
        # Only a changed value at or after the abort gets this far, so building the
        # key's name here costs nothing on the other keys.
        endswith(String(key), "_time") || continue
        return true
    end
    return false
end

# Bring `kept` up to date with `current` after a transition that stood.
function _catch_up!(kept, current)
    for (key, value) in current
        get(kept, key, _ABSENT) === value || (kept[key] = value)
    end
    # Every key of `current` is now in `kept`, so a surplus means the
    # transition deleted keys.
    length(kept) > length(current) && filter!(kv -> haskey(current, first(kv)), kept)
    return nothing
end

_iter_risks(::Nothing) = ()
_iter_risks(r::Risk) = (r,)
_iter_risks(rs) = rs

# ── Susceptibility and infectiousness as default risk sources ────────
#
# The host's susceptibility and the infector's infectiousness are not
# special engine rules. They are default risk sources on the same
# [`competing_risk`](@ref) surface interventions use: the engine composes
# them with the user's interventions and privileges neither, and a user
# could replace or extend them the same way.

"""Partial susceptibility of the exposed person: a contact whose
`susceptibility` is below 1 escapes infection with probability
`1 - susceptibility`. Applied to every exposure in a branching process, like
an intervention's [`competing_risk`](@ref)."""
struct HostSusceptibility end
function competing_risk(::HostSusceptibility, parent, contact, state)
    return contact.susceptibility < 1.0 ?
        Risk(block_probability = 1.0 - contact.susceptibility) : nothing
end

"""Reduced infectiousness of the infector: a case whose `infectiousness` is
below 1 fails to infect each contact with probability `1 - infectiousness`.
Applied to every exposure in a branching process, like an intervention's
[`competing_risk`](@ref)."""
struct InfectorInfectiousness end
function competing_risk(::InfectorInfectiousness, parent, contact, state)
    return parent.infectiousness < 1.0 ?
        Risk(block_probability = 1.0 - parent.infectiousness) : nothing
end

"""Only infected people transmit. A contact who was not infected can still
be followed up for its own contacts (for example to trace contacts of
contacts, see [`keep_active`](@ref)), but none of those contacts is infected
through them. Has no effect when everyone being followed is infected."""
struct InfectiousSource end
function competing_risk(::InfectiousSource, parent, contact, state)
    return is_infected(parent) ? nothing : Risk(block_probability = 1.0)
end

"""Immunity after infection: a person who has been infected cannot be
infected again until [`susceptible_again_time`](@ref), which is `Inf` (never)
unless a natural-history step sets it. Every model over a fixed population,
built in or written against [`contacts_of`](@ref), applies it to every settled
member once its `progression` can reach `:susceptible_again`, with no change
needed to how it names contacts. A model allows reinfection by adding a step
to its `progression` that sets `:susceptible_again_time`; the earlier
infection is then kept as a past episode (see [`close_episode!`](@ref)) when
the person is reinfected."""
struct HostImmunity end
function competing_risk(::HostImmunity, parent, contact, state)
    is_infected(contact) || return nothing
    return Risk(block_probability = 1.0, release_time = susceptible_again_time(contact))
end

"""End of a transmission route's window: no infection happens at or after
the earliest of the route's `until` events for the infector (death, recovery,
burial and so on), because the infector stopped transmitting by that route
before the contact would have happened. Routes without `until` (the default)
are not affected."""
struct WindowCensor end
function competing_risk(::WindowCensor, parent, contact, state)
    until = get(contact.state, :censor_until, nothing)
    until === nothing && return nothing
    t_end = Inf
    for s in until
        st = get(parent.state, Symbol(s, :_time), Inf)
        st < t_end && (t_end = st)
    end
    isfinite(t_end) || return nothing
    return Risk(event_time = t_end, block_probability = 1.0)
end

"""Stopped infections: a case whose infection was stopped before symptom
onset ([`abort_infection!`](@ref EpiBranch.abort_infection!), as post-exposure
vaccination with [`RingVaccination`](@ref) does) infects nobody from that
time on, whether or not the intervention that stopped it is still active."""
struct AbortedInfection end
function competing_risk(::AbortedInfection, parent, contact, state)
    _infection_aborted(parent) || return nothing
    return Risk(event_time = infection_aborted_time(parent), block_probability = 1.0)
end

# The built-in risk sources, in the order they apply. The calls are written out
# because a loop over a tuple of more than four distinct types is not
# union-split and would dispatch dynamically on every edge, for every model.
function _builtin_risk_blocks(parent, contact, state, transmission_time)
    _risk_blocks(InfectiousSource(), parent, contact, state, transmission_time) &&
        return true
    _risk_blocks(HostImmunity(), parent, contact, state, transmission_time) &&
        return true
    _risk_blocks(WindowCensor(), parent, contact, state, transmission_time) &&
        return true
    _risk_blocks(AbortedInfection(), parent, contact, state, transmission_time) &&
        return true
    _risk_blocks(HostSusceptibility(), parent, contact, state, transmission_time) &&
        return true
    _risk_blocks(InfectorInfectiousness(), parent, contact, state, transmission_time) &&
        return true
    return false
end

# The built-in sources the models over a fixed population compose, where each
# pair draws its contact times from a kernel. Only two of the six apply there.
# Two can never apply: an infector has settled and so is infected by
# construction, and route censoring is the infectious window's job rather than
# a tag written on a contact. The other two, the per-individual
# susceptibility and infectiousness, are rate multipliers on those models
# rather than per-contact blocks: each one scales the hazard a pair meets at
# (`_traits_scaled_draw`) or the pressure a susceptible absorbs, so resolving
# them here again would count them twice. `HostImmunity` does apply: a
# candidate the race proposes can be a settled member offered again once the
# race allows reinfection (see `_allows_reinfection` in sellke.jl), and this
# is where that candidate is held back until `susceptible_again_time`.
function _sellke_builtin_risk_blocks(parent, contact, state, transmission_time)
    _risk_blocks(HostImmunity(), parent, contact, state, transmission_time) &&
        return true
    return _risk_blocks(AbortedInfection(), parent, contact, state, transmission_time)
end

"""Apply one risk source's [`competing_risk`](@ref)(s) to a transmission;
return `true` if any active risk blocks it. Built-in risk sources and
interventions share this single risk-evaluation path."""
function _risk_blocks(source, parent, contact, state, transmission_time)
    risks = _iter_risks(competing_risk(source, parent, contact, state))
    return _risks_block(risks, parent, contact, state, transmission_time)
end

# A call per kind of answer `competing_risk` gives, so a source that answers
# with one of a few types has each loop compiled for its own type.
function _risks_block(risks, parent, contact, state, transmission_time)
    rng = state.rng
    for risk in risks
        event_t = sample_value(risk.event_time, rng, parent, contact, state)
        event_t > transmission_time && continue
        release_t = sample_value(risk.release_time, rng, parent, contact, state)
        transmission_time < release_t || continue
        prob = sample_value(risk.block_probability, rng, parent, contact, state)
        prob <= 0.0 && continue
        prob >= 1.0 && return true
        rand(rng) < prob && return true
    end
    return false
end

"""
    transmission_risks(model) -> iterable of risk sources

Ways in which the transmission model itself can stop a contact becoming
infected, applied alongside susceptibility, infectiousness and the
interventions. Only needed when writing a new model. For example, a
metapopulation model in which transmission between two places succeeds only
with some probability returns a source here, so that probability acts on
every contact while the contact is still made (and can still be traced).
Each source defines [`competing_risk`](@ref). Returns none by default.
"""
transmission_risks(::TransmissionModel) = ()

"""Whether any risk blocks the `parent` → `contact` transmission at
`transmission_time`: the built-in sources (host susceptibility, infector
infectiousness, …) first, then any the model contributes through
[`transmission_risks`](@ref), then the interventions in stack order. The
first to block wins.

Every model resolves a transmission through this one function, on each
candidate infection time the race proposes, so an intervention writes one
[`Risk`](@ref) for all of them. The risk's effect on the epidemic still
differs: a single exposure (a branching-process contact, or one a model names
through [`contacts_of`](@ref)) is lost when it is blocked, while a pair that
draws its contacts from a kernel comes round again, as the race redraws its
contact interval, so blocking a fraction of the contacts thins the hazard by
the same fraction. `builtin_blocks` is where the two part company, dropping
four of the built-in sources for a kernel pair: two that cannot apply there,
and the per-individual susceptibility and infectiousness, which the
contact-interval draw already folds in.
Nothing is drawn from the rng unless a risk actually applies."""
function _composed_risks_block(
        state::SimulationState, parent, contact,
        transmission_time, model_risks, interventions,
        builtin_blocks = _builtin_risk_blocks
    )
    builtin_blocks(parent, contact, state, transmission_time) && return true
    _any_risk_blocks(model_risks, parent, contact, state, transmission_time) &&
        return true
    return _any_risk_blocks(interventions, parent, contact, state, transmission_time)
end

# The sources in order, the first to block winning. A tuple is walked
# element by element so each call is resolved when the method is compiled;
# a call through a vector of mixed types is resolved at run time instead,
# which boxes `transmission_time` on every call.
_any_risk_blocks(::Tuple{}, parent, contact, state, transmission_time) = false
@inline function _any_risk_blocks(sources::Tuple, parent, contact, state, transmission_time)
    _risk_blocks(first(sources), parent, contact, state, transmission_time) &&
        return true
    return _any_risk_blocks(Base.tail(sources), parent, contact, state, transmission_time)
end
function _any_risk_blocks(sources, parent, contact, state, transmission_time)
    for source in sources
        _risk_blocks(source, parent, contact, state, transmission_time) && return true
    end
    return false
end

"""Apply attributes function to an individual. No-op for NoAttributes."""
_apply_attributes!(::NoAttributes, rng, ind) = nothing
_apply_attributes!(f, rng, ind) = f(rng, ind)
function _apply_attributes!(builders::Union{Tuple, AbstractVector}, rng, ind)
    for build! in builders
        _apply_attributes!(build!, rng, ind)
    end
    return nothing
end

"""
A population characteristic drawn once per group and shared by every member
of that group, such as a household's reporting probability. The group is
whatever the individual has under `group_key`, set by [`groups`](@ref) or an
earlier entry in `attributes`. Each group's value is drawn when its first
member is created and kept for the rest of the outbreak.

Create one with [`group_attribute`](@ref).
"""
struct GroupAttribute{D}
    key::Symbol
    group_key::Symbol
    propensity::D
    cache::Dict{Any, Any}
end

function _apply_attributes!(attribute::GroupAttribute, rng, ind)
    haskey(ind.state, attribute.group_key) || throw(
        ArgumentError(
            "group_attribute(:$(attribute.key)) needs :$(attribute.group_key) set on an individual " *
                "before it runs; list `groups(n; key = :$(attribute.group_key))`, or " *
                "another attributes function setting that key, ahead of it."
        )
    )
    ind.state[attribute.key] = get!(attribute.cache, ind.state[attribute.group_key]) do
        sample_value(attribute.propensity, rng, ind)
    end
    return nothing
end

"""Attributes for one run. An element that caches per-run draws (a
[`GroupAttribute`](@ref EpiBranch.GroupAttribute)) gets a fresh, empty cache,
so one attributes object can be reused across runs and across threads, each
run drawing its own values. Everything else passes through."""
_fresh_attributes(x) = x
function _fresh_attributes(a::GroupAttribute)
    return GroupAttribute(a.key, a.group_key, a.propensity, Dict{Any, Any}())
end
_fresh_attributes(xs::Union{Tuple, AbstractVector}) = map(_fresh_attributes, xs)

# ── Attributes function constructors ─────────────────────────────────

"""
    clinical_presentation(; incubation_period, prob_asymptomatic = 0.0)

Give each case a symptom onset time and decide whether it is asymptomatic.
Pass the result as `attributes` to [`ModelSpec`](@ref).

- `incubation_period`: distribution of the time from infection to symptom
  onset, in days. Each symptomatic case's onset time is its infection time
  plus a draw from it.
- `prob_asymptomatic`: probability that a case never develops symptoms
  (default 0). Asymptomatic cases have no onset time (`NaN`). Give a number,
  a distribution (one probability drawn per case) or a function of the random
  number generator and the individual, `(rng, ind) -> probability`, for
  example to make it depend on age.

[`Isolation`](@ref) needs these onset times, and [`linelist`](@ref) reports
them as `date_onset`.

# Examples

Symptomatic-only with a log-normal incubation period:

```julia
attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5))
```

With 30% asymptomatic:

```julia
attributes = clinical_presentation(
    incubation_period = LogNormal(1.6, 0.5),
    prob_asymptomatic = 0.3,
)
```

Per-individual asymptomatic probability drawn from a Beta:

```julia
attributes = clinical_presentation(
    incubation_period = LogNormal(1.6, 0.5),
    prob_asymptomatic = Beta(2, 8),
)
```

Age-conditional (children much more likely to be asymptomatic; list
after `demographics` so `:age` is set first):

```julia
attributes = [
    demographics(age_distribution = Uniform(0, 90)),
    clinical_presentation(
        incubation_period = LogNormal(1.6, 0.5),
        prob_asymptomatic = (rng, ind) -> ind.state[:age] < 18 ? 0.6 : 0.2,
    ),
]
```

See also [`demographics`](@ref).
"""
function clinical_presentation(;
        incubation_period::Distribution,
        prob_asymptomatic = 0.0
    )
    return ClinicalPresentation(incubation_period, prob_asymptomatic)
end

struct ClinicalPresentation{D, P}
    incubation_period::D
    prob_asymptomatic::P
end

function (clinical::ClinicalPresentation)(rng, ind)
    pa = sample_value(clinical.prob_asymptomatic, rng, ind)
    is_asymp = rand(rng) < pa
    ind.state[:asymptomatic] = is_asymp
    ind.state[:incubation_period] = is_asymp ? NaN : rand(rng, clinical.incubation_period)
    return _set_onset_from_incubation!(ind)
end

"""
    _set_onset_from_incubation!(ind)

Set `:onset_time` to `infection_time + :incubation_period` from the
stored host incubation period. Asymptomatic individuals (`NaN`
incubation) get a `NaN` onset, and so does an infection aborted before
onset, which never reaches it. Applies when `:incubation_period` is
present on the individual.
"""
function _set_onset_from_incubation!(ind::Individual)
    haskey(ind.state, :incubation_period) || return nothing
    inc = _state_real(ind.state, :incubation_period, NaN)
    ind.state[:onset_time] = isnan(inc) || _infection_aborted(ind) ? NaN :
        ind.infection_time + inc
    return nothing
end

"""
    demographics(; age_distribution=NoAgeDistribution(), age_range=(0, 90), prob_female=0.5)

Give each person an age (`:age`, whole years) and a sex (`:sex`, `:female`
or `:male`). Pass the result as `attributes` to [`ModelSpec`](@ref); list it
before any population characteristic that reads the age.

- `age_distribution`: distribution of ages. Draws are rounded down and kept
  within `age_range`. If it is left out, ages are drawn uniformly from
  `age_range`.
- `age_range`: youngest and oldest age as a tuple (default `(0, 90)`).
- `prob_female`: probability that a person is female (default 0.5).

# Examples
```julia
attributes = demographics(age_distribution = Gamma(4.0, 9.0), prob_female = 0.52)

# age-dependent asymptomatic fraction
attributes = [
    demographics(age_range = (0, 85)),
    clinical_presentation(
        incubation_period = LogNormal(1.6, 0.5),
        prob_asymptomatic = (rng, ind) -> ind.state[:age] < 18 ? 0.6 : 0.2,
    ),
]
```
"""
function demographics(;
        age_distribution::Union{Distribution, NoAgeDistribution} = NoAgeDistribution(),
        age_range::Tuple{Int, Int} = (0, 90),
        prob_female::Real = 0.5
    )
    return Demographics(age_distribution, age_range, float(prob_female))
end

"""Callable population characteristic `demographics` returns. Has a type,
rather than being a bare closure, so [`EpiBranch.offspring_law_unchanged`](@ref)
can recognise it as leaving the offspring law unchanged."""
struct Demographics{D}
    age_distribution::D
    age_range::Tuple{Int, Int}
    prob_female::Float64
end

function (d::Demographics)(rng, ind)
    ind.state[:age] = _sample_age(rng, d.age_distribution, d.age_range)
    return ind.state[:sex] = rand(rng) < d.prob_female ? :female : :male
end

_sample_age(rng, ::NoAgeDistribution, age_range) = rand(rng, age_range[1]:age_range[2])
function _sample_age(rng, dist::Distribution, age_range)
    return clamp(floor(Int, rand(rng, dist)), age_range...)
end

"""
    groups(n_groups::Integer; key::Symbol = :group)

Assign each person at random to one of `n_groups` equally likely groups,
numbered `1` to `n_groups` and recorded under `key` (default `:group`). A
group can stand for a community, a health area or a household; it is what
[`GroupVaccination`](@ref) and [`group_attribute`](@ref) work on.

Groups are assigned independently of who infected whom, so transmission does
not cluster within them. To make it cluster, use a multi-type
[`BranchingProcess`](@ref) with one type per group and a next-generation
matrix with most transmission within types, and use the type itself as the
group in `GroupVaccination(...; group_key = :type)`. [`group_attribute`](@ref)
and [`vaccine_acceptance`](@ref) cannot group by type, since population
characteristics are drawn before a person's type is set.

# Examples

Twenty equally likely communities:

```julia
attributes = groups(20)
```

A named unit, for a builder that also sets other fields:

```julia
attributes = [groups(10; key = :household), clinical_presentation(...)]
```

See also [`GroupVaccination`](@ref), [`vaccine_acceptance`](@ref).
"""
function groups(n_groups::Integer; key::Symbol = :group)
    n_groups >= 1 || throw(ArgumentError("n_groups must be at least 1, got $n_groups"))
    return Groups(n_groups, key)
end

"""Callable population characteristic `groups` returns. Has a type, rather
than being a bare closure, so [`EpiBranch.offspring_law_unchanged`](@ref) can
recognise it as leaving the offspring law unchanged."""
struct Groups
    n_groups::Int
    key::Symbol
end

function (g::Groups)(rng, ind)
    return ind.state[g.key] = rand(rng, 1:g.n_groups)
end

"""
    transmission_traits(; susceptibility = 1.0, infectiousness = 1.0)

Give each person a susceptibility and an infectiousness. Pass the result as
`attributes` to [`ModelSpec`](@ref).

- `susceptibility`: probability that an exposed person is infected.
- `infectiousness`: probability that each contact the person makes once
  infected leads to infection, so 0.5 halves the number they infect.

Both default to 1 (no change). In the continuous-time models
([`HomogeneousProcess`](@ref), [`MixingProcess`](@ref), network and household
models) both instead
multiply the rate of transmission, which lowers the chance of infection only
over an infectious period of limited length.

Each argument can be a number given to everyone, a distribution drawn once
per person, or a function of the random number generator and the individual,
`(rng, ind) -> value`, for example to make it depend on age (list
[`demographics`](@ref) first so the age is set).

# Examples

Constant per-contact infection probability:

```julia
attributes = transmission_traits(susceptibility = 0.3)
```

Per-individual heterogeneity:

```julia
attributes = transmission_traits(
    susceptibility = Beta(2, 5),
    infectiousness = Beta(8, 2),
)
```

Age-conditional susceptibility (list after `demographics` so `:age` is
set first):

```julia
attributes = [
    demographics(age_distribution = Uniform(0, 90)),
    transmission_traits(
        susceptibility = (rng, ind) -> ind.state[:age] >= 65 ? 0.8 : 0.3,
    ),
]
```

For rules this function does not cover, an `attributes` entry can set the
fields directly, as in `(rng, ind) -> (ind.susceptibility = ...)`. Such a
closure sends `loglikelihood(::ChainSizes, model)`, `chain_size_distribution`
and the closed-form incidence likelihood to simulation rather than their
analytical path, since [`EpiBranch.offspring_law_unchanged`](@ref) cannot
tell what a bare function does; wrap it in its own callable type with a
method on that trait to get the analytical path back.

See also [`clinical_presentation`](@ref), [`demographics`](@ref).
"""
function transmission_traits(;
        susceptibility = 1.0,
        infectiousness = 1.0
    )
    return TransmissionTraits(
        susceptibility, infectiousness,
        _trait_sampler(susceptibility), _trait_sampler(infectiousness)
    )
end

"""Callable population characteristic `transmission_traits` returns. Keeps
the original `susceptibility`/`infectiousness` arguments alongside the
samplers built from them, so [`EpiBranch.offspring_law_unchanged`](@ref) can
tell a neutral value (the default `1.0`) from one that thins transmission."""
struct TransmissionTraits{S, I, SF, IF}
    susceptibility::S
    infectiousness::I
    sus_sampler::SF
    inf_sampler::IF
end

function (traits::TransmissionTraits)(rng, ind)
    ind.susceptibility = traits.sus_sampler(stream(rng, ind.id, :susceptibility), ind)
    return ind.infectiousness = traits.inf_sampler(stream(rng, ind.id, :infectiousness), ind)
end

# Neutral only for a bare `1`; a distribution or function might also always
# give 1, but that can't be told without drawing from it, so it counts as a
# change, conservatively.
_trait_is_neutral(x::Real) = isone(x)
_trait_is_neutral(x) = false
function offspring_law_unchanged(t::TransmissionTraits)
    return _trait_is_neutral(t.susceptibility) && _trait_is_neutral(t.infectiousness)
end

_trait_sampler(x::Real) =
let v = float(x)
    (rng, ind) -> v
end
_trait_sampler(d::Distribution) = (rng, ind) -> float(rand(rng, d))
_trait_sampler(f) = (rng, ind) -> float(f(rng, ind))

"""
    group_attribute(key::Symbol; value, group_key = :group)

Give every member of a group the same value of a numeric population
characteristic `key`, drawn once per group, such as a household's reporting
probability or a community's acceptance of vaccination. The value lasts for
the whole outbreak.

`value` can be a number, a distribution or a function of the random number
generator and the individual, `(rng, ind) -> value`; a function is called for
the first member created in each group. List [`groups`](@ref), or another
entry that sets `group_key`, earlier in `attributes`; a person without it
raises an error. Each simulation draws its own group values, so the same
`group_attribute` can be reused across simulations.

For example, share a reporting probability within each household:

```julia
attributes = [groups(50; key = :household),
    group_attribute(:reporting_probability; value = Beta(6, 4),
        group_key = :household)]
progression = [Reporting(
    delay = Exponential(2.0),
    probability = (rng, ind) -> ind.state[:reporting_probability])]
```

[`vaccine_acceptance`](@ref) is a convenience constructor for this operation.
"""
function group_attribute(key::Symbol; value, group_key::Symbol = :group)
    return GroupAttribute(key, group_key, value, Dict{Any, Any}())
end

"""
    vaccine_acceptance(; propensity, group_key = :group, key = :vaccine_acceptance)

Give each group a probability of accepting vaccination, shared by all its
members, recorded under `key` (default `:vaccine_acceptance`). Acceptance
tends to cluster by household or community, and the contacts who avoid
tracing are often the ones who decline a dose. The group is whatever the
person has under `group_key` (default `:group`, as set by [`groups`](@ref)),
so refusal clusters in the same unit [`GroupVaccination`](@ref) vaccinates.
List the entry that sets `group_key` before this one; a person without it
raises an error.

Use the value as a vaccination's `coverage` with a function such as
`coverage = (rng, ind) -> ind.state[:vaccine_acceptance]`.

`propensity` is a number, a distribution or a function
`(rng, ind) -> probability`, drawn once per group when its first member is
created. Each member then accepts or declines at random with that
probability. With a single number every group has the same probability, which
is the same as no grouping. With a distribution, the average coverage is the
same but coverage varies more between groups, some mostly vaccinated and
others mostly not. For a whole group to accept or decline together, use a
propensity of exactly 0 or 1, such as
`(rng, ind) -> Float64(rand(rng, Bernoulli(p)))`.

# Examples

Each community's coverage is Beta-distributed around a mean of 60%:

```julia
attributes = [groups(20),
    vaccine_acceptance(propensity = Beta(6, 4))]
gv = GroupVaccination(efficacy = 0.8,
    coverage = (rng, ind) -> ind.state[:vaccine_acceptance])
```

Households, under a key of their own:

```julia
attributes = [groups(50; key = :household),
    vaccine_acceptance(propensity = Beta(2, 2), group_key = :household)]
```

See also [`groups`](@ref), [`clinical_presentation`](@ref),
[`demographics`](@ref).
"""
function vaccine_acceptance(;
        propensity,
        group_key::Symbol = :group,
        key::Symbol = :vaccine_acceptance
    )
    return group_attribute(key; value = propensity, group_key)
end

# ── Intervention field validation ────────────────────────────────────

"""The information an intervention needs recorded on each person, such as
`:onset_time` for isolation, checked when the outbreak starts. Default: none."""
required_fields(::AbstractIntervention) = Symbol[]

"""Check that a person has all the information the interventions or
natural-history steps need (their [`required_fields`](@ref EpiBranch.required_fields)),
and raise an error naming the missing one and how to provide it."""
function _validate_required_fields(individual, items)
    for item in items
        for field in required_fields(item)
            if !haskey(individual.state, field)
                itype = typeof(item)
                hint = _field_hint(field)
                error("$itype requires field :$field on individuals. $hint")
            end
        end
    end
    return
end

function _field_hint(field::Symbol)
    hints = Dict(
        :onset_time => "Provide attributes = clinical_presentation(incubation_period = ...).",
        :asymptomatic => "Provide attributes = clinical_presentation(incubation_period = ...).",
        :incubation_period => "Provide attributes = clinical_presentation(incubation_period = ...).",
        :age => "Provide attributes = demographics(age_distribution = ...).",
        :sex => "Provide attributes = demographics(...)."
    )
    return get(
        hints, field,
        "Set this field via an attributes function, or add the step that " *
            "produces it to `progression`."
    )
end
