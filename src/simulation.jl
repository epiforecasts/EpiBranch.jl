"""
    simulate(model::TransmissionModel;
             max_cases=10_000, max_generations=100, max_time=nothing,
             n_initial=nothing, initial_cases=nothing, stopping_rules=nothing,
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
  stops when transmission dies out. The limits are checked at the end of each
  generation, so the final number of cases can exceed `max_cases`.
- `stopping_rules`: a vector of [`AbstractStoppingRule`](@ref)s for finer
  control over when the run ends. When given, `max_cases`,
  `max_generations` and `max_time` are ignored.
- `rng`: the random number generator; pass a seeded one, such as
  `Xoshiro(42)`, for reproducible runs.
- `condition`: a range of final outbreak sizes, such as `10:1000`. The
  simulation is repeated until the total number of cases falls inside it (at
  most `max_attempts` tries), for example to keep only outbreaks that took
  off. An error is raised if no run qualifies.
- `initial_cases`: for `NetworkProcess`, `RoutedNetwork` and
  `HouseholdProcess` only, the IDs of the people infected at time
  zero (an empty vector is allowed). It replaces the random choice of index
  cases. Give either `initial_cases` or `n_initial`, not both; IDs outside
  the population raise an error. With an `external_hazard`, everyone else can
  still be infected from outside from time zero. Other models raise an error
  when given a vector here.

The probability and delay of each step in a case's natural history are set
on the transitions in the `progression` of a [`ModelSpec`](@ref). Each takes
a constant or a function of the random number generator and the individual,
`(rng, ind) -> value`, so rates and delays can depend on age or risk group.

!!! note
    [`HomogeneousProcess`](@ref), `NetworkProcess`,
    `RoutedNetwork` and `HouseholdProcess` run over a fixed
    population until transmission
    dies out or `max_time` is reached. They ignore `max_cases`,
    `max_generations` and most stopping rules, and warn when one is set.

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
    _validate_process_windows(model, _progression(model))
    _warn_incomplete_terminal_coverage(_progression(model))
    _warn_ignored_termination(
        model, max_cases, max_generations, max_time, stopping_rules
    )
    sim_opts = SimOpts(;
        n_initial, initial_cases, max_cases, max_generations, max_time,
        stopping_rules
    )
    _validate_initial_cases(model, sim_opts)
    return _simulate(
        model, sim_opts; interventions = interventions(model),
        attributes = attributes(model), progression = _progression(model),
        observation = observation(model), recorder = recorder(model), rng,
        condition, max_attempts
    )
end

# Internal single run against a built `SimOpts`. The forcing layers
# (interventions, attributes, progression, observation, recorder) are passed
# in explicitly, so a `ModelSpec` can supply its own while the process stays
# the dispatched model. The public methods read them off a bare process, or
# off the spec. `recorder` is read only by the continuous-time (Sellke) race;
# the generation-based engine below never drops a pair for a standing block,
# so it has nothing to ask one.
function _simulate(
        model::TransmissionModel, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng,
        condition, max_attempts
    )
    if condition !== nothing
        for _ in 1:max_attempts
            state = _simulate(
                model, sim_opts; interventions, attributes,
                progression, observation, recorder, rng, condition = nothing,
                max_attempts
            )
            state.cumulative_cases in condition && return state
        end
        throw(
            ErrorException(
                "No simulation produced an outbreak of size $condition within $max_attempts attempts"
            )
        )
    end

    state = initialise_state(
        model, sim_opts, interventions, progression, attributes, rng
    )
    _resolve_new_transitions!(state, 0)

    while !should_terminate(state, sim_opts)
        _advance_generation!(model, state, interventions)
    end

    apply_observation!(observation, state, rng)
    return state
end

"""
    simulate(model, n::Int; parallel=false, kwargs...)

Simulate `n` independent outbreaks from the same model and return them as a
vector of [`SimulationState`](@ref)s. It takes the same stopping keywords
(`max_cases`, `max_generations`, `max_time`, `stopping_rules`), `n_initial`,
`initial_cases` and `rng` as the single-outbreak method; `condition` is not
available here. The usual next step is
[`containment_probability`](@ref), the share of these outbreaks that died out.

With `parallel = true` the runs are spread over the CPU threads Julia was
started with (for example `julia --threads 4`), and the results are still
reproducible for a given seeded `rng`.

# Examples
```julia
using EpiBranch, Distributions, Random

model = BranchingProcess(NegBin(2.5, 0.16), Gamma(2.5, 2.0))
states = simulate(model, 1000; max_cases = 5000, rng = Xoshiro(1))
containment_probability(states; max_cases = 5000)
```
"""
function simulate(
        model::TransmissionModel, n::Int;
        n_initial::Union{Int, Nothing} = nothing,
        initial_cases::Union{AbstractVector{<:Integer}, Nothing} = nothing,
        max_cases::Union{Int, Nothing} = _DEFAULT_MAX_CASES,
        max_generations::Union{Int, Nothing} = _DEFAULT_MAX_GENERATIONS,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng(),
        parallel::Bool = false
    )
    _validate_process_windows(model, _progression(model))
    _warn_incomplete_terminal_coverage(_progression(model))
    _warn_ignored_termination(
        model, max_cases, max_generations, max_time, stopping_rules
    )
    sim_opts = SimOpts(;
        n_initial, initial_cases, max_cases, max_generations, max_time,
        stopping_rules
    )
    _validate_initial_cases(model, sim_opts)
    return _simulate_n(
        model, n, sim_opts; interventions = interventions(model),
        attributes = attributes(model), progression = _progression(model),
        observation = observation(model), recorder = recorder(model), rng,
        parallel
    )
end

function _simulate_n(
        model::TransmissionModel, n::Int, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng,
        parallel::Bool = false
    )
    if parallel && Threads.nthreads() > 1
        seeds = [rand(rng, UInt64) for _ in 1:n]
        results = Vector{SimulationState}(undef, n)
        Threads.@threads for i in 1:n
            local_rng = Random.Xoshiro(seeds[i])
            results[i] = _simulate(
                model, sim_opts; interventions, attributes,
                progression, observation, recorder, rng = local_rng,
                condition = nothing, max_attempts = 10_000
            )
        end
        return results
    else
        return [
            _simulate(
                model, sim_opts; interventions, attributes, progression,
                observation, recorder, rng, condition = nothing,
                max_attempts = 10_000
            )
                for _ in 1:n
        ]
    end
end

# ── Shared helpers for the structure-driven (continuous-time) models ────
# The homogeneous, household and network `_simulate` methods run their own
# Sellke loop rather than the generation engine, so they share the same two
# concerns: retrying until a `condition` is met, and reconciling the aggregate
# bookkeeping the engine would otherwise maintain.

# Whether a model's simulation honours the termination controls (`max_cases`,
# `max_generations`, `max_time`, `stopping_rules`). The generation-based engine
# does. The structure-driven pools run over their fixed population until
# extinction or `max_time`, and ignore the other controls; they override this
# to `false`.
_honours_termination_controls(::TransmissionModel) = true

# The time at which a structure-driven run ends: the earliest `time_bound`
# among the stopping rules, or `Inf`.
function _max_time(sim_opts)
    return minimum(time_bound(r) for r in sim_opts.stopping_rules; init = Inf)
end

_rule_names(rules) = join([string(nameof(typeof(r))) for r in rules], ", ")

# Warn when a termination control is set on a model that ignores it, so the
# silent no-op is discoverable. Compares against the keyword defaults, so only
# an explicitly-set control triggers the warning; `simulate` on a pool with no
# termination keywords stays quiet.
function _warn_ignored_termination(
        model, max_cases, max_generations, max_time, stopping_rules
    )
    _honours_termination_controls(model) && return nothing
    ignored = String[]
    max_cases != _DEFAULT_MAX_CASES && push!(ignored, "max_cases")
    max_generations != _DEFAULT_MAX_GENERATIONS && push!(ignored, "max_generations")
    # Each rule says for itself whether a run that never consults `should_stop`
    # applies it in full. One that does not is reported by name, and separately
    # when a time bound of its own was still applied, so a rule that is only
    # half honoured does not read as having done nothing.
    partial = ""
    if stopping_rules !== nothing
        unapplied = filter(r -> !honoured_without_should_stop(r), stopping_rules)
        inert = _rule_names(filter(r -> !isfinite(time_bound(r)), unapplied))
        partial = _rule_names(filter(r -> isfinite(time_bound(r)), unapplied))
        isempty(inert) || push!(ignored, "stopping_rules ($inert)")
    end
    (isempty(ignored) && isempty(partial)) && return nothing
    msg = "$(nameof(typeof(model))) runs to extinction or `max_time` over its " *
        "fixed population and ignores the other termination controls"
    isempty(ignored) || (msg *= "; $(join(ignored, ", ")) had no effect")
    # A rule that bounds time and tests something else had half an effect, so
    # it is reported apart from the controls that had none.
    isempty(partial) || (msg *= "; of $partial only the time bound applied")
    @warn msg * " (only n_initial, a time bound and condition apply)."
    return nothing
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

Set `cumulative_cases` and `max_infection_time` from the per-individual state a
continuous-time (Sellke) loop writes directly, and `extinct` to the given
`extinct`, keeping the returned state consistent with the generation engine's
bookkeeping. `extinct` is whether the Sellke loop(s) that built `state` ran
until no candidate infection remained, rather than being cut off at
`max_time` with candidates still pending.
"""
function _reconcile_sellke_bookkeeping!(state::SimulationState, extinct::Bool)
    state.cumulative_cases = count(
        ind -> get(ind.state, :infected, false), state.individuals
    )
    state.max_infection_time = maximum(
        (
            ind.infection_time
                for ind in state.individuals if get(ind.state, :infected, false)
        );
        init = 0.0
    )
    state.extinct = extinct
    return state
end

# ── Unified generation step ─────────────────────────────────────────
#
# Every model advances through this one step. The only thing a model
# varies is how it names this generation's contacts, and there are two
# extension paths for that:
#
#   * Offspring-driven (the tree case — BranchingProcess). The model
#     defines [`generate_offspring`](@ref): how *many* contacts each
#     infectious parent makes, as a pure count. It constructs nothing and
#     assigns no time. The engine creates that many fresh, never-seen
#     contacts and times each one — the default [`collect_exposures`](@ref).
#     The tree and the timing factorise, so the model stays a pure draw.
#
#   * Structure-driven (the graph case — a contact network; households later).
#     The contacts are *existing* nodes a count cannot name, and a
#     susceptible can be reached by several infectious neighbours in one
#     generation (a loop). The model defines [`contacts_of`](@ref) — the
#     actual nodes, each with its infection time — and overrides
#     `collect_exposures` with [`gather_by_target`](@ref), which
#     deduplicates so a node reached several times resolves once.
#
# Everything downstream — intervention hooks, competing-risks resolution,
# clinical transitions, bookkeeping — is shared.

"""
    contacts_of(model, node, state) -> iterable of (contact, infection_time)

The people an infectious case `node` can reach in this generation, each with
the time they would be infected, as `(contact, infection_time)` pairs.

Only needed when writing a new transmission model that spreads over a fixed
set of people, such as a contact network (the case's network neighbours) or
households (the other household members). Such a model defines this function
for its own type and uses [`gather_by_target`](@ref), so that a person
exposed by several cases in one generation is infected at most once.

Models in which every contact is a new person, such as a branching process,
define [`generate_offspring`](@ref) instead. In both cases the simulation
applies interventions, natural history and the infection decision itself.
Calling `contacts_of` on a model that does not define it raises an error
explaining what to define.
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

The generation time a model uses to time each case's secondary cases, read
by the default [`collect_exposures`](@ref). Returns the model's
`generation_time` field. A new model that times its contacts another way (for
example with a separate delay per transmission route) defines this function
for its own type instead of having a `generation_time` field.
"""
model_generation_time(model::TransmissionModel) = model.generation_time

"""
    collect_exposures(model, state) -> (targets, edges, minted, is_new)

The people each infectious case exposes in this generation, and when. Only
needed when writing a new transmission model.

Returns the distinct exposed people (`targets`); for each of them, the
`(infector_id, infection_time)` exposures reaching them (`edges`) and whether
they were created in this generation (`is_new`); and the contacts newly
created in this generation (`minted`).

The default suits models in which every contact is a new person: it asks
each infectious case how many contacts it makes via
[`generate_offspring`](@ref), then creates and times that many contacts.
Models whose contacts are people who already exist and can be exposed by
more than one case in a generation (networks, households) use
[`gather_by_target`](@ref) instead, which calls [`contacts_of`](@ref) and
groups the exposures by person.
"""
function collect_exposures(model::TransmissionModel, state::SimulationState)
    pre = length(state.individuals)   # contacts created this step get id > pre
    T = _timetype(state)
    targets = Individual{T}[]
    edges = Vector{Tuple{Int, T}}[]
    for idx in state.active_ids
        parent = state.individuals[idx]
        offspring = generate_offspring(model, parent, state)
        # the generation interval: how long after this parent was infected
        # each of its contacts occurs.
        gt_dist = get_generation_time(model_generation_time(model), parent)
        _materialise_offspring!(targets, edges, offspring, parent, state, gt_dist)
    end
    minted = view(state.individuals, (pre + 1):length(state.individuals))
    return targets, edges, minted, trues(length(targets))
end

# Engine half of the offspring-driven contract: the model returns a count
# from `generate_offspring`; the engine creates each contact with
# `make_contact!` and assigns it a generation time. Each fresh contact is
# its own target reached by a single edge (the tree case). Single-type
# offspring is a count; multi-type is a count per type.
function _materialise_offspring!(
        targets, edges, n_contacts::Int,
        parent::Individual, state::SimulationState,
        gt_dist::Union{Distribution, NoGenerationTime}
    )
    T = _timetype(state)
    for _ in 1:n_contacts
        t = transmission_time(gt_dist, parent, state)
        push!(targets, make_contact!(state, parent, t))
        push!(edges, Tuple{Int, T}[(parent.id, t)])
    end
    return nothing
end

function _materialise_offspring!(
        targets, edges, counts::Vector{Int},
        parent::Individual, state::SimulationState,
        gt_dist::Union{Distribution, NoGenerationTime}
    )
    T = _timetype(state)
    for (type_idx, n) in enumerate(counts)
        for _ in 1:n
            t = transmission_time(gt_dist, parent, state)
            push!(targets, make_contact!(state, parent, t; type_idx))
            push!(edges, Tuple{Int, T}[(parent.id, t)])
        end
    end
    return nothing
end

# Window-aware exposure collection for the branching process. Each
# infectiousness window draws its own offspring and times them from its
# `from` state. A window contributes contacts only once its `from` state
# has been reached for this parent (`:infection` always has, so the
# default single window reproduces the offspring-driven path above).
function collect_exposures(model::BranchingProcess, state::SimulationState)
    pre = length(state.individuals)
    T = _timetype(state)
    targets = Individual{T}[]
    edges = Vector{Tuple{Int, T}}[]
    for idx in state.active_ids
        parent = state.individuals[idx]
        for window in model.infectiousness
            from_t = _state_time(parent, window.from)
            isfinite(from_t) || continue
            kernel = get_generation_time(window.kernel, parent)
            counts = draw_offspring(state.rng, window.offspring, parent, state)
            _materialise_window!(
                targets, edges, counts, parent, state, from_t, kernel, window.until
            )
        end
    end
    minted = view(state.individuals, (pre + 1):length(state.individuals))
    return targets, edges, minted, trues(length(targets))
end

# A contact's infection time is the window's `from`-state time plus a draw
# from its kernel (the contact interval measured from `from`).
_window_infection_time(::NoGenerationTime, from_t::Real, state) = from_t
function _window_infection_time(kernel::Distribution, from_t::Real, state)
    return from_t + rand(state.rng, kernel)
end

# Tag a contact with its window's `until` states so `WindowCensor` can
# block it after the infector is removed. Skipped for windows with no
# `until` (the default), so single-window models write no extra state.
_tag_window!(contact, until::Tuple{}) = nothing
_tag_window!(contact, until) = (contact.state[:censor_until] = until; nothing)

# Single-type: one count. Multi-type: a count per type.
function _materialise_window!(
        targets, edges, n_contacts::Int,
        parent::Individual, state::SimulationState, from_t::Real,
        kernel::Union{Distribution, NoGenerationTime}, until
    )
    T = _timetype(state)
    for _ in 1:n_contacts
        t = _window_infection_time(kernel, from_t, state)
        contact = make_contact!(state, parent, t)
        _tag_window!(contact, until)
        push!(targets, contact)
        push!(edges, Tuple{Int, T}[(parent.id, t)])
    end
    return nothing
end

function _materialise_window!(
        targets, edges, counts::Vector{Int},
        parent::Individual, state::SimulationState, from_t::Real,
        kernel::Union{Distribution, NoGenerationTime}, until
    )
    T = _timetype(state)
    for (type_idx, n) in enumerate(counts)
        for _ in 1:n
            t = _window_infection_time(kernel, from_t, state)
            contact = make_contact!(state, parent, t; type_idx)
            _tag_window!(contact, until)
            push!(targets, contact)
            push!(edges, Tuple{Int, T}[(parent.id, t)])
        end
    end
    return nothing
end

"""
    gather_by_target(model, state) -> (targets, edges, minted, is_new)

A version of [`collect_exposures`](@ref) for models in which several cases
can expose the same person in one generation (networks, households). It
collects every exposure of a person and lets the simulation decide once
whether, and by whom, they are infected. Newly created contacts are each
treated separately, so a model may mix new people with existing ones.
"""
function gather_by_target(model::TransmissionModel, state::SimulationState)
    pre = length(state.individuals)
    T = _timetype(state)
    targets = Individual{T}[]
    edges = Vector{Tuple{Int, T}}[]
    is_new = Bool[]
    pos = Dict{Int, Int}()       # node id -> target index; shared nodes only
    for idx in state.active_ids
        parent = state.individuals[idx]
        for (target, time) in contacts_of(model, parent, state)
            if target.id > pre
                push!(targets, target)
                push!(edges, Tuple{Int, T}[(parent.id, time)])
                push!(is_new, true)
            else
                j = get(pos, target.id, 0)
                if j == 0
                    push!(targets, target)
                    push!(edges, Tuple{Int, T}[])
                    push!(is_new, false)
                    j = length(targets)
                    pos[target.id] = j
                end
                push!(edges[j], (parent.id, time))
            end
        end
    end
    minted = view(state.individuals, (pre + 1):length(state.individuals))
    return targets, edges, minted, is_new
end

"""Simulate one generation of transmission. Case-level interventions such as
isolation act on the infectious cases ([`_prepare_parents!`](@ref)); contacts
are made and timed ([`collect_exposures`](@ref)); tracing and vaccination act
on those contacts ([`_intervene!`](@ref)); then each contact is infected or
not, and the natural history of the new cases is drawn ([`_resolve!`](@ref)).
In a branching process a contact is created together with its infection
time; in a fixed population the contacts already exist."""
function _advance_generation!(
        model::TransmissionModel,
        state::SimulationState, interventions::Vector{<:AbstractIntervention}
    )
    _prepare_parents!(state, interventions)
    targets, edges, minted, is_new = collect_exposures(model, state)
    # Snapshotted before anything below moves a live field: `_intervene!`'s
    # provisional parent assignment overwrites `parent_id`/`infection_time`
    # unconditionally, on a pre-existing node offered again just as much as on
    # a fresh one.
    reinfections = _reinfection_episodes(targets, is_new)
    _intervene!(state, interventions, targets, edges, minted)
    _resolve!(model, state, interventions, targets, edges, is_new, reinfections)
    return nothing
end

# A pre-existing target already carrying a prior infection, keyed by its
# position in `targets` rather than by id, since two targets never share a
# position. Only a model that offers one past its first infection (through
# `contacts_of`) ever populates this; empty for every other model, which is
# every built-in one today.
function _reinfection_episodes(targets, is_new)
    episodes = Dict{Int, InfectionEpisode}()
    for i in eachindex(targets)
        is_new[i] && continue
        target = targets[i]
        is_infected(target) && (episodes[i] = InfectionEpisode(target))
    end
    return episodes
end

"""Step 1 of a generation: interventions act on the infectious cases (for
example, isolating them) before they transmit."""
function _prepare_parents!(
        state::SimulationState,
        interventions::Vector{<:AbstractIntervention}
    )
    for idx in state.active_ids
        individual = state.individuals[idx]
        for intervention in interventions
            resolve_individual!(intervention, individual, state)
        end
    end
    return nothing
end

"""Step 3 of a generation: interventions act on the people exposed in this
generation, before it is decided whether they are infected. Each new contact
is set up for every intervention, and each exposed person is provisionally
assigned to the case that exposed them earliest, so that tracing and ring
vaccination can reach them."""
function _intervene!(
        state::SimulationState,
        interventions::Vector{<:AbstractIntervention},
        targets::Vector{<:Individual}, edges::Vector{<:Vector{<:Tuple}},
        minted
    )
    # Newly created contacts (already appended to state by make_contact!)
    # get their intervention state initialised.
    for contact in minted
        for intervention in interventions
            initialise_individual!(intervention, contact, state)
        end
    end

    # Provisional parent = earliest exposing edge. With one edge (the tree
    # case) this is the contact's only parent.
    for i in eachindex(targets)
        es = edges[i]
        length(es) > 1 && sort!(es, by = last)
        targets[i].parent_id, targets[i].infection_time = es[1]
    end

    for intervention in interventions
        apply_post_transmission!(intervention, state, targets)
    end
    return nothing
end

"""Step 4 of a generation: decide which exposed people are infected, and
record the new cases. An exposure need not lead to infection. The infector's
infectiousness, the contact's susceptibility, anything the model adds and any
interventions can each prevent it, whichever acts first. A person exposed by
several cases is infected if any of the exposures transmits, at the time of
the earliest one that does. When someone who was infected before is
reinfected (possible only once their immunity has waned, see
[`HostImmunity`](@ref EpiBranch.HostImmunity)), the earlier infection is kept
as a past episode; see [`close_episode!`](@ref)."""
function _resolve!(
        model::TransmissionModel, state::SimulationState,
        interventions::Vector{<:AbstractIntervention},
        targets::Vector{<:Individual}, edges::Vector{<:Vector{<:Tuple}},
        is_new, reinfections = Dict{Int, InfectionEpisode}()
    )
    model_risks = transmission_risks(model)
    infected_so_far = 0
    newly_infected = eltype(targets)[]
    for i in eachindex(targets)
        target = targets[i]
        # A pre-existing node already carrying a prior infection, offered
        # again as a candidate contact by a model whose host has waned
        # enough (see `HostImmunity`); its live fields as `_advance_generation!`
        # snapshotted them, before `_intervene!`'s provisional parent
        # assignment or the trial edges below moved them. The earlier episode
        # is either archived (confirmed reinfection) or restored (every edge
        # failed) rather than lost either way.
        prior_episode = get(reinfections, i, nothing)
        reinfection = prior_episode !== nothing
        infected = false
        for (pid, t) in edges[i]
            target.parent_id = pid
            target.infection_time = t
            if _decide_infected(state, target, model_risks, interventions, infected_so_far)
                infected = true
                break
            end
        end
        if reinfection && !infected
            target.parent_id = prior_episode.parent_id
            target.infection_time = prior_episode.infection_time
            _drop_stale_abort!(target)
            continue
        end
        target.state[:infected] = infected
        if infected
            reinfection && close_episode!(target, prior_episode)
            infected_so_far += 1
            parent = state.individuals[target.parent_id]
            target.generation = parent.generation + 1
            target.chain_id = parent.chain_id
            # Onset follows from the *infection* time. A minted contact is
            # created at its infection time, so this is idempotent; a
            # pre-instantiated node was created with no infection time (NaN),
            # so this derives its onset from the time it was actually infected.
            _set_onset_from_incubation!(target)
            # Freshly created contacts were already registered on their
            # parent by `make_contact!`; shared network nodes are not.
            is_new[i] || push!(parent.secondary_case_ids, target.id)
            push!(newly_infected, target)
        elseif !is_new[i]
            # A pre-instantiated node exposed but not infected this
            # generation stays a clean susceptible; clear the provisional
            # parent and infection time left from the failed exposure. A NaN
            # infection time marks it as never infected; 0.0 would mean
            # infected at time 0. (Minted "contact-only" individuals keep their
            # parent because they are real contacts.)
            target.parent_id = 0
            target.infection_time = NaN
        end
        _drop_stale_abort!(target)
    end

    for target in newly_infected
        target.infection_time > state.max_infection_time &&
            (state.max_infection_time = target.infection_time)
    end
    state.cumulative_cases += length(newly_infected)
    state.current_generation += 1

    # Next active set: the cases that transmit, plus any nodes an
    # intervention asks to keep active (`keep_active`), such as uninfected
    # contacts a tracing depth wants to keep growing. Who stays active is
    # not a special built-in rule.
    next_active = [target.id for target in newly_infected]
    for intervention in interventions
        append!(next_active, keep_active(intervention, state, targets, is_new))
    end
    state.active_ids = next_active
    state.extinct = isempty(next_active)

    if !isempty(state.transitions)
        for target in newly_infected
            resolve_transitions!(state, target)
        end
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
                     infection_time = NaN)

Add `n` people to the outbreak's population and return them. Used by a new
transmission model's `initialise_state` to build its population.

Each person starts with the given `infection_time`: `NaN` by default, since
nobody is infected yet. [`seed!`](@ref) sets the index cases' infection time,
and the model sets everyone else's if they are infected later. Pass
`infection_time = 0` when everyone added is an index case, so that population
characteristics drawn at creation see the time of their infection.

For each person, the population characteristics are drawn first, then
`setup(ind, i)` runs (to record model-specific information such as a network
node or household id), then a random type is drawn for multi-type models, and
finally each intervention sets up its own information on that person.
"""
function add_individuals!(
        state::SimulationState, n::Integer, interventions;
        n_types::Integer = 1, setup = (ind, i) -> nothing, infection_time::Real = NaN
    )
    base = length(state.individuals)
    added = eltype(state.individuals)[]
    for i in 1:n
        ind = _create_individual(state, 0, base + i, base + i, infection_time)
        setup(ind, i)
        # Match the new-contact path's ordering (`make_contact!` sets
        # `:type` before the engine calls `initialise_individual!`) so an
        # intervention that reads `:type` at init sees the same state for
        # seed cases and downstream contacts.
        n_types > 1 && (ind.state[:type] = rand(state.rng, 1:n_types))
        for intervention in interventions
            initialise_individual!(intervention, ind, state)
        end
        push!(state.individuals, ind)
        push!(added, ind)
    end
    return added
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
        _validate_required_fields(first_ind, transitions)
    end
    state.cumulative_cases = length(ids)
    state.active_ids = collect(ids)
    state.extinct = isempty(ids)
    return state
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
        n_types = n_types(model), infection_time = 0
    )
    seed!(state, 1:(sim_opts.n_initial), interventions, transitions)
    return state
end

function should_terminate(state::SimulationState, sim_opts::SimOpts)
    for rule in sim_opts.stopping_rules
        should_stop(rule, state) && return true
    end
    return false
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

"""Create a new person and draw their population characteristics. They
start as not infected; only the infection decision later in the generation
marks them infected. Interventions set up their information on the person
later in the generation, so creating a contact (in `make_contact!` or a
model's [`contacts_of`](@ref)) involves no intervention.
"""
function _create_individual(
        state::SimulationState, parent_id::Int,
        chain_id::Int, next_id::Int, inf_time::Real
    )
    T = _timetype(state)
    s = Dict{Symbol, Any}(:infected => false)

    # Build `Individual{T}` directly (not the keyword constructor) so the type
    # matches `state.individuals`, whatever `T` is: seed cases pass a plain
    # `Float64` infection time but must still land as `Individual{T}` under AD.
    ind = Individual{T}(
        next_id, parent_id,
        state.current_generation + (parent_id == 0 ? 0 : 1),
        chain_id, convert(T, inf_time), one(T), one(T), Int[], s,
        InfectionEpisode{T}[]
    )

    _apply_attributes!(state.attributes, state.rng, ind)

    return ind
end

_set_type!(contact, ::NoTypeLabels) = nothing
_set_type!(contact, idx::Int) = (contact.state[:type] = idx)

"""
    make_contact!(state, parent, infection_time; type_idx = NoTypeLabels())

Add one new contact of the case `parent`, exposed at `infection_time` (days),
to the outbreak, and return it. Its population characteristics are drawn
when it is created; interventions act on it later in the generation.

Only needed when writing a new transmission model. The built-in models call
it for every secondary case [`generate_offspring`](@ref) asks for. A model
that creates new people inside its [`contacts_of`](@ref) calls it directly,
returning each contact with its infection time:

```julia
function contacts_of(m::MyModel, parent, state)
    map(1:rand(state.rng, m.offspring)) do _
        t = parent.infection_time + rand(state.rng, m.generation_time)
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
        next_id, infection_time
    )
    _set_type!(contact, type_idx)
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

"""Decide whether a single contact is infected along one edge by
composing competing risks. Built-in risks (per-individual
susceptibility, parent infectiousness, population-level susceptibility
for finite-population models) are applied first; any
[`competing_risk`](@ref) contributed by an intervention is then applied
in stack order. A risk whose event has occurred by transmission time blocks
transmission with its `block_probability`; transmission succeeds iff no
risk blocks it.

`infected_so_far` counts infections already resolved this generation, so
population susceptibility shrinks the pool as contacts get infected and
the cumulative case count cannot overshoot a finite `population_size`."""
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
unless a natural-history step sets it. None of the built-in models exposes an
infected person again, so this matters only for a new model that does,
through its own [`contacts_of`](@ref). Such a model allows reinfection by
adding a step to its `progression` that sets `:susceptible_again_time`; the
earlier infection is then kept as a past episode (see
[`close_episode!`](@ref)) when the person is reinfected."""
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
    until = get(contact.state, :censor_until, ())
    isempty(until) && return nothing
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

# The built-in sources the continuous-time models compose. Only one of the six
# applies there. Three are the generation engine's own and can never apply: an
# infector on those models has settled and so is infected by construction,
# route censoring is the infectious window's job rather than a tag written on a
# contact, and a candidate the race proposes is always one not yet settled,
# hence never already infected either — the race has nowhere yet to put a
# second episode even once a model wants to offer one. The other two, the
# per-individual susceptibility and infectiousness, are rate multipliers on
# those models rather than per-contact blocks: each one
# scales the hazard a pair meets at (`_traits_scaled_draw`) or the pressure a
# susceptible absorbs, so resolving them here again would count them twice.
function _sellke_builtin_risk_blocks(parent, contact, state, transmission_time)
    return _risk_blocks(AbortedInfection(), parent, contact, state, transmission_time)
end

"""Apply one risk source's [`competing_risk`](@ref)(s) to a transmission;
return `true` if any active risk blocks it. Built-in risk sources and
interventions share this single risk-evaluation path."""
function _risk_blocks(source, parent, contact, state, transmission_time)
    rng = state.rng
    for risk in _iter_risks(competing_risk(source, parent, contact, state))
        event_t = _sample_value(risk.event_time, rng, parent, contact, state)
        event_t > transmission_time && continue
        release_t = _sample_value(risk.release_time, rng, parent, contact, state)
        transmission_time < release_t || continue
        prob = _sample_value(risk.block_probability, rng, parent, contact, state)
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

"""
Decide whether `contact` is infected by the case that exposed it, and return
`true` if so. An index case is always infected. Otherwise, in a finite
population the contact first escapes with probability one minus the
[`susceptible_fraction`](@ref). Then each thing that can prevent infection
is checked in turn: the contact's susceptibility and the infector's
infectiousness, anything from the model's [`transmission_risks`](@ref), and
the interventions. The first one that prevents it decides.
"""
function _decide_infected(
        state::SimulationState, contact::Individual,
        model_risks, interventions, infected_so_far::Int
    )
    contact.parent_id == 0 && return true
    rng = state.rng
    parent = state.individuals[contact.parent_id]
    transmission_time = contact.infection_time

    # Finite-population depletion (dispatched on the population type).
    pop_suscept = susceptible_fraction(state, infected_so_far)
    pop_suscept <= 0.0 && return false
    pop_suscept < 1.0 && rand(rng) > pop_suscept && return false

    return !_composed_risks_block(
        state, parent, contact, transmission_time, model_risks, interventions
    )
end

"""Whether any risk blocks the `parent` → `contact` transmission at
`transmission_time`: the built-in sources (host susceptibility, infector
infectiousness, …) first, then any the model contributes through
[`transmission_risks`](@ref), then the interventions in stack order. The
first to block wins.

Both engines resolve a transmission through this one function — the
generation engine on each contact it created, the continuous-time models on
each candidate infection time they propose — so an intervention writes one
[`Risk`](@ref) for both. What that risk does to the epidemic still differs: the
generation engine blocks a contact and loses it, while on a continuous-time
model the contact comes round again — the race redraws the pair's contact
interval, the pool gives the susceptible a fresh resistance above the pressure
it has absorbed — so blocking a fraction of the contacts thins the hazard by
the same fraction. `builtin_blocks` is where the two part company, dropping
four of the five built-in sources on a continuous-time model: two that cannot
apply there, and the per-individual susceptibility and infectiousness, which
those models already carry in the contact-interval draw and in the pool's
threshold and force.
Nothing is drawn from the rng unless a risk actually applies."""
function _composed_risks_block(
        state::SimulationState, parent, contact,
        transmission_time, model_risks, interventions,
        builtin_blocks = _builtin_risk_blocks
    )
    builtin_blocks(parent, contact, state, transmission_time) && return true
    for source in model_risks
        _risk_blocks(source, parent, contact, state, transmission_time) && return true
    end
    for intervention in interventions
        _risk_blocks(intervention, parent, contact, state, transmission_time) &&
            return true
    end
    return false
end

"""Sweep newly added infected individuals (those at indices
`from_index+1:end`) and run clinical transitions on each. Called by
[`simulate`](@ref) after `initialise_state` to resolve transitions on
the seed cases; per-generation resolution happens inside
[`_advance_generation!`](@ref)."""
function _resolve_new_transitions!(state::SimulationState, from_index::Int)
    isempty(state.transitions) && return nothing
    @inbounds for i in (from_index + 1):length(state.individuals)
        ind = state.individuals[i]
        if get(ind.state, :infected, false)
            resolve_transitions!(state, ind)
        end
    end
    return nothing
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
        _sample_value(attribute.propensity, rng, ind)
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
    pa = _sample_value(clinical.prob_asymptomatic, rng, ind)
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
    inc = ind.state[:incubation_period]
    ind.state[:onset_time] = isnan(inc) || _infection_aborted(ind) ? NaN :
        ind.infection_time + inc
    return nothing
end

"""
    demographics(; age_distribution=nothing, age_range=(0, 90), prob_female=0.5)

Give each person an age (`:age`, whole years) and a sex (`:sex`, `:female`
or `:male`). Pass the result as `attributes` to [`ModelSpec`](@ref); list it
before any population characteristic that reads the age.

- `age_distribution`: distribution of ages. Draws are rounded down and kept
  within `age_range`. Without one, ages are drawn uniformly from `age_range`.
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
    pf = float(prob_female)
    return function (rng, ind)
        ind.state[:age] = _sample_age(rng, age_distribution, age_range)
        return ind.state[:sex] = rand(rng) < pf ? :female : :male
    end
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
matrix with most transmission within types, and set `:group` from the type
with your own function `(rng, ind) -> ...` in place of `groups`.

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
    return function (rng, ind)
        return ind.state[key] = rand(rng, 1:n_groups)
    end
end

"""
    transmission_traits(; susceptibility = 1.0, infectiousness = 1.0)

Give each person a susceptibility and an infectiousness. Pass the result as
`attributes` to [`ModelSpec`](@ref).

- `susceptibility`: probability that an exposed person is infected.
- `infectiousness`: probability that each contact the person makes once
  infected leads to infection, so 0.5 halves the number they infect.

Both default to 1 (no change). In the continuous-time models
([`HomogeneousProcess`](@ref), network and household models) both instead
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
fields directly, as in `(rng, ind) -> (ind.susceptibility = ...)`.

See also [`clinical_presentation`](@ref), [`demographics`](@ref).
"""
function transmission_traits(;
        susceptibility = 1.0,
        infectiousness = 1.0
    )
    sus = _trait_sampler(susceptibility)
    inf = _trait_sampler(infectiousness)
    return function (rng, ind)
        ind.susceptibility = sus(rng, ind)
        return ind.infectiousness = inf(rng, ind)
    end
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
observation = PerCaseObservation(
    detection_prob = (rng, ind) -> ind.state[:reporting_probability])
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
    return get(hints, field, "Set this field via an attributes function.")
end
