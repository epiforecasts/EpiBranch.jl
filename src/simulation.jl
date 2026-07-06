"""
    simulate(model::TransmissionModel;
             max_cases=10_000, max_generations=100, max_time=nothing,
             n_initial=1, stopping_rules=nothing,
             rng=Random.default_rng(), condition=nothing, max_attempts=10_000)

Run a single outbreak simulation.

The interventions, attributes and observation are carried by `model`, set
on the process constructor, so the model alone determines the generative
process. `simulate` takes only execution controls.

Termination is set by `max_cases`, `max_generations`, and `max_time` (any
of which may be `nothing` to drop that limit); the run always stops on
extinction. For finer control pass a `stopping_rules` vector of
[`AbstractStoppingRule`](@ref). `n_initial` is the number of seed cases.

The case's clinical timeline — the [`AbstractClinicalTransition`](@ref)s a
case moves through (latent, onset, severity, death/recovery, burial) — is
the model's `progression`, set on the [`BranchingProcess`](@ref). Each
transition's `probability` and `delay` accept constants or
`(rng, ind) -> value` functions, so age- or risk-conditional rates and
delays are configured per-transition.

If `condition` is provided (a `UnitRange{Int}`), simulations are repeated
until one produces an outbreak whose cumulative cases fall within the range,
up to `max_attempts`.
"""
function simulate(model::TransmissionModel;
        n_initial::Int = 1,
        max_cases::Union{Int, Nothing} = 10_000,
        max_generations::Union{Int, Nothing} = 100,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng(),
        condition::Union{UnitRange{Int}, Nothing} = nothing,
        max_attempts::Int = 10_000)
    sim_opts = SimOpts(; n_initial, max_cases, max_generations, max_time,
        stopping_rules)
    return _simulate(model, sim_opts; interventions = interventions(model),
        attributes = attributes(model), rng, condition, max_attempts)
end

# Internal single run against a built `SimOpts`. The public methods build
# the `SimOpts` from the flat termination keywords.
function _simulate(model::TransmissionModel, sim_opts::SimOpts;
        interventions, attributes, rng, condition, max_attempts)
    if condition !== nothing
        for _ in 1:max_attempts
            state = _simulate(model, sim_opts; interventions, attributes, rng,
                condition = nothing, max_attempts)
            state.cumulative_cases in condition && return state
        end
        throw(ErrorException(
            "No simulation produced an outbreak of size $condition within $max_attempts attempts"
        ))
    end

    state = initialise_state(
        model, sim_opts, interventions, _progression(model), attributes, rng)
    _resolve_new_transitions!(state, 0)

    while !should_terminate(state, sim_opts)
        _advance_generation!(model, state, interventions)
    end

    apply_observation!(observation(model), state, rng)
    return state
end

"""
    simulate(model, n::Int; parallel=false, kwargs...)

Run `n` independent outbreak simulations. Returns a
`Vector{SimulationState}`. Takes the same termination keywords as the
single-run method.

When `parallel=true`, simulations are distributed across available threads
using independent RNG streams derived from the provided `rng`. Use
`julia --threads N` to enable multi-threading.
"""
function simulate(model::TransmissionModel, n::Int;
        n_initial::Int = 1,
        max_cases::Union{Int, Nothing} = 10_000,
        max_generations::Union{Int, Nothing} = 100,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng(),
        parallel::Bool = false)
    sim_opts = SimOpts(; n_initial, max_cases, max_generations, max_time,
        stopping_rules)
    return _simulate_n(model, n, sim_opts; interventions = interventions(model),
        attributes = attributes(model), rng, parallel)
end

function _simulate_n(model::TransmissionModel, n::Int, sim_opts::SimOpts;
        interventions, attributes, rng, parallel::Bool = false)
    if parallel && Threads.nthreads() > 1
        seeds = [rand(rng, UInt64) for _ in 1:n]
        results = Vector{SimulationState}(undef, n)
        Threads.@threads for i in 1:n
            local_rng = Random.Xoshiro(seeds[i])
            results[i] = _simulate(model, sim_opts; interventions, attributes,
                rng = local_rng, condition = nothing, max_attempts = 10_000)
        end
        return results
    else
        return [_simulate(model, sim_opts; interventions, attributes, rng,
                    condition = nothing, max_attempts = 10_000) for _ in 1:n]
    end
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

The structure-driven transmission seam: the contacts an infectious `node`
reaches this generation, as `(contact, infection_time)` pairs. Used by
models whose contacts are *existing* nodes a plain count cannot name — a
network returns the node's graph neighbours (a graph with loops);
households would return household members. Pair it with
[`gather_by_target`](@ref) so a node reached by several neighbours in one
generation resolves once.

Offspring-driven models (a branching process, where every contact is
fresh) use [`generate_offspring`](@ref) instead and never define
`contacts_of`: they return a count and the engine creates and times the
contacts. Either way the shared engine ([`collect_exposures`](@ref),
[`_advance_generation!`](@ref)) handles interventions, competing-risks
resolution, clinical transitions, and bookkeeping.

No in-package model is structure-driven, so the abstract fallback states
the contract rather than failing with a bare `MethodError`. A structure-
driven extension (see the `EpiNetwork` subpackage) defines a method on its
own model type.
"""
function contacts_of(model::TransmissionModel, parent, state::SimulationState)
    throw(ArgumentError(
        "$(typeof(model)) defines no contacts_of method. A structure-driven " *
        "model must implement contacts_of(model, parent, state) returning " *
        "(contact, infection_time) pairs; see the EpiNetwork subpackage for a " *
        "worked example."))
end

"""
    model_generation_time(model)

The generation-time spec the default [`collect_exposures`](@ref) reads when
timing the contacts of an offspring-driven model. Defaults to the model's
`generation_time` field; a model that times its contacts differently (no such
field, or a per-window kernel) overrides this instead of being forced to carry
a `generation_time` field.
"""
model_generation_time(model::TransmissionModel) = model.generation_time

"""
    collect_exposures(model, state) -> (targets, edges, minted, is_new)

This generation's candidate exposures, grouped by target, gathered by
walking [`contacts_of`](@ref) for each active node. Returns parallel
vectors: distinct `targets`, the `edges` (`(parent_id, infection_time)`)
reaching each, the contacts `minted` this generation, and `is_new`
flagging which targets were created this step.

The default serves offspring-driven models: it asks each active parent
for an offspring count via [`generate_offspring`](@ref), then creates and
times that many fresh contacts itself. Every contact is its own target —
no two parents share one — so `is_new` is all `true`. Models whose
contacts are existing nodes that can be *shared* across parents in a
generation (networks, households, clustering) override this with
[`gather_by_target`](@ref), which walks [`contacts_of`](@ref) and
deduplicates so a node reached several times resolves once.
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
function _materialise_offspring!(targets, edges, n_contacts::Int,
        parent::Individual, state::SimulationState,
        gt_dist::Union{Distribution, NoGenerationTime})
    T = _timetype(state)
    for _ in 1:n_contacts
        t = transmission_time(gt_dist, parent, state)
        push!(targets, make_contact!(state, parent, t))
        push!(edges, Tuple{Int, T}[(parent.id, t)])
    end
    return nothing
end

function _materialise_offspring!(targets, edges, counts::Vector{Int},
        parent::Individual, state::SimulationState,
        gt_dist::Union{Distribution, NoGenerationTime})
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
                targets, edges, counts, parent, state, from_t, kernel, window.until)
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
function _materialise_window!(targets, edges, n_contacts::Int,
        parent::Individual, state::SimulationState, from_t::Real,
        kernel::Union{Distribution, NoGenerationTime}, until)
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

function _materialise_window!(targets, edges, counts::Vector{Int},
        parent::Individual, state::SimulationState, from_t::Real,
        kernel::Union{Distribution, NoGenerationTime}, until)
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

A [`collect_exposures`](@ref) implementation for models whose contacts
can be shared across parents within a generation (networks, households,
clustering). Exposures are deduplicated by id, so a node reached by
several infectious neighbours in one generation collects all its incoming
edges and resolves once. Freshly created contacts (id past the
generation's starting count) are each their own target, so a model that
mixes new and existing contacts — e.g. a clustering dial — works too. The
dedup map is only touched for shared nodes.
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

"""Advance the simulation by one generation through the unified engine, as
four phases: interventions act on the active infectives ([`_prepare_parents!`](@ref)),
the exposure phase ([`collect_exposures`](@ref)) builds and times this
generation's contacts, contact-level interventions act on the exposed
([`_intervene!`](@ref)), and the resolve phase ([`_resolve!`](@ref)) decides
infection under competing risks and updates bookkeeping and clinical
transitions. In the growing tree the build and time steps are fused in
`collect_exposures` (a contact is minted with its infection time); they
separate only in the fixed-population path, where contacts pre-exist."""
function _advance_generation!(model::TransmissionModel,
        state::SimulationState, interventions::Vector{<:AbstractIntervention})
    _prepare_parents!(state, interventions)
    targets, edges, minted, is_new = collect_exposures(model, state)
    _intervene!(state, interventions, targets, edges, minted)
    _resolve!(model, state, interventions, targets, edges, is_new)
    return nothing
end

"""Phase 1 — interventions act on the active infectives before they transmit."""
function _prepare_parents!(state::SimulationState,
        interventions::Vector{<:AbstractIntervention})
    for idx in state.active_ids
        individual = state.individuals[idx]
        for intervention in interventions
            resolve_individual!(intervention, individual, state)
        end
    end
    return nothing
end

"""Phase 3 — contact-level interventions act on this generation's exposures.
Newly minted contacts get their intervention state initialised; each target is
given a provisional parent (its earliest exposing edge) so contact-level
interventions (tracing, ring vaccination) act on the exposed target before
infection is resolved; then the interventions act."""
function _intervene!(state::SimulationState,
        interventions::Vector{<:AbstractIntervention},
        targets::Vector{<:Individual}, edges::Vector{<:Vector{<:Tuple}},
        minted)
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

"""Phase 4 — decide infection under competing risks and update bookkeeping.
Exposure is not infection. Each contact exposed this generation is decided
infected-or-not: the infector's infectiousness, the contact's susceptibility,
any risks the model contributes and any interventions all act as competing
risks on the same footing. A contact is infected if any of its exposing edges
transmits; the earliest successful edge fixes the infection time."""
function _resolve!(model::TransmissionModel, state::SimulationState,
        interventions::Vector{<:AbstractIntervention},
        targets::Vector{<:Individual}, edges::Vector{<:Vector{<:Tuple}},
        is_new)
    model_risks = transmission_risks(model)
    infected_so_far = 0
    newly_infected = eltype(targets)[]
    for i in eachindex(targets)
        target = targets[i]
        infected = false
        for (pid, t) in edges[i]
            target.parent_id = pid
            target.infection_time = t
            if _decide_infected(state, target, model_risks, interventions, infected_so_far)
                infected = true
                break
            end
        end
        target.state[:infected] = infected
        if infected
            infected_so_far += 1
            parent = state.individuals[target.parent_id]
            target.generation = parent.generation + 1
            target.chain_id = parent.chain_id
            # Onset follows from the *infection* time. A minted contact is
            # created at its infection time, so this is idempotent; a
            # pre-instantiated network node was created at t=0, so this
            # recomputes its onset from the time it was actually infected.
            _set_onset_from_incubation!(target)
            # Freshly created contacts were already registered on their
            # parent by `make_contact!`; shared network nodes are not.
            is_new[i] || push!(parent.secondary_case_ids, target.id)
            push!(newly_infected, target)
        elseif !is_new[i]
            # A pre-instantiated node exposed but not infected this
            # generation stays a clean susceptible; clear the provisional
            # parent left from the failed exposure. (Minted "contact-only"
            # individuals keep their parent — they are real contacts.)
            target.parent_id = 0
            target.infection_time = 0.0
        end
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
_kernel_time_type(::Function) = Float64
function _time_type(m::BranchingProcess)
    mapreduce(w -> _kernel_time_type(w.kernel), promote_type, m.infectiousness;
        init = Float64)
end

"""
    new_state(model, transitions, attributes, rng) -> SimulationState

An empty [`SimulationState`](@ref) for `model`: no individuals yet,
generation 0, carrying the model's population size, the run's `attributes`,
and the clinical `transitions`. The starting point a model's
`initialise_state` builds on with [`add_individuals!`](@ref) and
[`seed!`](@ref).
"""
function new_state(model::TransmissionModel, transitions, attributes,
        rng::AbstractRNG)
    T = _time_type(model)
    SimulationState(Individual{T}[], Int[], 0, rng, 0, false,
        population_size(model), zero(T), attributes,
        convert(Vector{AbstractClinicalTransition}, transitions))
end

"""
    add_individuals!(state, n, interventions; n_types = 1, setup = (ind, i) -> nothing)

Create `n` individuals, append them to `state`, and return them. For each,
`setup(ind, i)` runs first (to stamp model-specific state such as a node or
household id), then a random type is assigned for multi-type models, then
each intervention's `initialise_individual!` runs. Used by a model's
`initialise_state` to build its population.
"""
function add_individuals!(state::SimulationState, n::Integer, interventions;
        n_types::Integer = 1, setup = (ind, i) -> nothing)
    base = length(state.individuals)
    added = eltype(state.individuals)[]
    for i in 1:n
        ind = _create_individual(state, 0, base + i, base + i, 0.0)
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

Mark the individuals with the given `ids` as infected index cases: set
`:infected`, derive `:onset_time` from any incubation period, validate that
the interventions' and transitions' required fields are present (on the
first id, before any transition closure runs, so a missing field surfaces
as the engine's friendly message), and record the run's initial bookkeeping
(`cumulative_cases`, `active_ids`, `extinct`). Used by a model's
`initialise_state` after [`add_individuals!`](@ref).
"""
function seed!(state::SimulationState, ids, interventions, transitions)
    for id in ids
        ind = state.individuals[id]
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

Build the starting [`SimulationState`](@ref) for `model`: its population and
its index cases, before the engine steps any generations. The default
offspring-driven method mints `sim_opts.n_initial` index cases and seeds them.
A structure-driven model (a fixed, depleting population such as a network or
households) defines its own method, typically by building the population with
[`new_state`](@ref) and [`add_individuals!`](@ref) and infecting the index
cases with [`seed!`](@ref).
"""
function initialise_state(model::TransmissionModel, sim_opts::SimOpts,
        interventions, transitions, attributes, rng::AbstractRNG)
    state = new_state(model, transitions, attributes, rng)
    add_individuals!(state, sim_opts.n_initial, interventions;
        n_types = n_types(model))
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

Fraction of the population still susceptible at this point in the
simulation. Dispatched on the `population_size` type carried by
`SimulationState{<:Any, <:Any, P}`, so downstream packages can introduce
structured populations (households, contact networks, age-stratified
pools) by defining a new population-size type and adding a method
here.

`extra_infected` accounts for contacts already infected within the
current step but not yet registered in `state.cumulative_cases`.

Built-in methods:

- `NoPopulation` — unbounded, always `1.0`.
- `Int` — single global pool of that size; depletion is global.
"""
function susceptible_fraction(state::SimulationState{<:Any, <:Any, NoPopulation},
        extra_infected::Int = 0)
    1.0
end

function susceptible_fraction(state::SimulationState{<:Any, <:Any, Int},
        extra_infected::Int = 0)
    n_susceptible = state.population_size - state.cumulative_cases - extra_infected
    n_susceptible <= 0 && return 0.0
    return n_susceptible / state.population_size
end

"""Create a new Individual with attributes applied. `:infected` is left
at `false` so that every freshly created contact carries a
not-yet-decided flag — only the engine's competing-risks resolution may
set it to `true`. Intervention `initialise_individual!` is called by the
engine after exposures are collected, not here; this keeps contact
creation (`make_contact!`, and any model's [`contacts_of`](@ref))
intervention-free.
"""
function _create_individual(state::SimulationState, parent_id::Int,
        chain_id::Int, next_id::Int, inf_time::Real)
    T = _timetype(state)
    s = Dict{Symbol, Any}(:infected => false)

    # Build `Individual{T}` directly (not the keyword constructor) so the type
    # matches `state.individuals`, whatever `T` is: seed cases pass a plain
    # `Float64` infection time but must still land as `Individual{T}` under AD.
    ind = Individual{T}(
        next_id, parent_id,
        state.current_generation + (parent_id == 0 ? 0 : 1),
        chain_id, convert(T, inf_time), one(T), one(T), Int[], s)

    _apply_attributes!(state.attributes, state.rng, ind)

    return ind
end

_set_type!(contact, ::NoTypeLabels) = nothing
_set_type!(contact, idx::Int) = (contact.state[:type] = idx)

"""
    make_contact!(state, parent, infection_time; type_idx = NoTypeLabels())

Create a new contact of `parent` at `infection_time`, add it to
`state.individuals`, and register it in `parent.secondary_case_ids`.
Returns the new `Individual`. Attributes are applied at creation;
intervention state is initialised by the engine after the generation's
exposures are collected, not here.

This is an engine primitive. The default offspring-driven path calls it
for each contact a model's [`generate_offspring`](@ref) count asks for, so
those models never touch it. A structure-driven model that mints fresh
nodes inside its [`contacts_of`](@ref) may call it directly, returning
each contact with its infection time:

```julia
function contacts_of(m::MyModel, parent, state)
    map(1:rand(state.rng, m.offspring)) do _
        t = parent.infection_time + rand(state.rng, m.generation_time)
        (make_contact!(state, parent, t), t)
    end
end
```

The engine handles every other side effect: `resolve_individual!` on
each parent before collection, `initialise_individual!` and
`apply_post_transmission!` on the new contacts after, competing-risks
resolution that sets `:infected`, clinical transitions, and per-step
bookkeeping (`cumulative_cases`, `active_ids`, `max_infection_time`,
…).
"""
function make_contact!(state::SimulationState, parent::Individual,
        infection_time::Real;
        type_idx::Union{Int, NoTypeLabels} = NoTypeLabels())
    next_id = length(state.individuals) + 1
    contact = _create_individual(state, parent.id, parent.chain_id,
        next_id, infection_time)
    _set_type!(contact, type_idx)
    push!(parent.secondary_case_ids, next_id)
    push!(state.individuals, contact)
    return contact
end

"""
    resolve_transitions!(state, individual)

Resolve `individual`'s clinical natural history: run every transition on
`state.transitions` (the model's `progression`) against the individual — first
each transition's `initialise_individual!`, then each `resolve_individual!` —
and arbitrate the terminal outcome. This stamps the timeline keys downstream
code reads (`:infectious_time`, `:onset_time`, `:outcome`/`:outcome_time`, and
anything else a transition writes) onto `individual.state`.

Part of the public extension API. The built-in engine calls this for every new
case; a structure-driven model that runs its own simulation loop (rather than
the generation-based engine) calls it itself, once per case, after the case's
attributes and intervention state are set. The transitions come from the model's
`progression`, placed on the state when it is built with
[`new_state`](@ref EpiBranch.new_state).
"""
function resolve_transitions!(state::SimulationState, individual)
    transitions = state.transitions
    isempty(transitions) && return nothing
    for transition in transitions
        initialise_individual!(transition, individual, state)
    end
    for transition in transitions
        resolve_individual!(transition, individual, state)
    end
    _finalise_terminal!(individual, transitions)
    return nothing
end

"""Decide whether a single contact is infected along one edge by
composing competing risks. Built-in risks (per-individual
susceptibility, parent infectiousness, population-level susceptibility
for finite-population models) are applied first; any
[`competing_risk`](@ref) contributed by an intervention is then applied
in stack order. A risk that has fired by transmission time blocks
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

"""Default risk source: the host's per-individual susceptibility, as a
block probability `1 - susceptibility` on the [`competing_risk`](@ref)
surface."""
struct HostSusceptibility end
function competing_risk(::HostSusceptibility, parent, contact, state)
    contact.susceptibility < 1.0 ?
    Risk(block_probability = 1.0 - contact.susceptibility) : nothing
end

"""Default risk source: the infector's infectiousness, as a block
probability `1 - infectiousness` on the [`competing_risk`](@ref) surface."""
struct InfectorInfectiousness end
function competing_risk(::InfectorInfectiousness, parent, contact, state)
    parent.infectiousness < 1.0 ?
    Risk(block_probability = 1.0 - parent.infectiousness) : nothing
end

"""Default risk source: only an infected source transmits. An uninfected
source blocks transmission entirely, so the engine can keep uninfected
nodes active (e.g. for contact tracing depth, via [`keep_active`](@ref))
to grow their contacts without those contacts becoming infected. A no-op
in the usual case where every active node is infected."""
struct InfectiousSource end
function competing_risk(::InfectiousSource, parent, contact, state)
    is_infected(parent) ? nothing : Risk(block_probability = 1.0)
end

"""Default risk source: an infectiousness window's censoring. A contact
whose transmission time falls at or after the earliest of its window's
`until` states (the infector's death, recovery, burial, …) is blocked: the
infector was removed before the contact would have happened. The window's
`until` state names are tagged on the contact as `:censor_until`; each
resolves to the infector's `Symbol(state, :_time)`. A no-op for windows
with no `until` (the default), which write no tag."""
struct WindowCensor end
function competing_risk(::WindowCensor, parent, contact, state)
    until = get(contact.state, :censor_until, ())
    isempty(until) && return nothing
    t_end = Inf
    for s in until
        st = get(parent.state, Symbol(s, :_time), Inf)::Float64
        st < t_end && (t_end = st)
    end
    isfinite(t_end) || return nothing
    return Risk(event_time = t_end, block_probability = 1.0)
end

const _BUILTIN_RISK_SOURCES = (InfectiousSource(), WindowCensor(),
    HostSusceptibility(), InfectorInfectiousness())

"""Apply one risk source's [`competing_risk`](@ref)(s) to a transmission;
return `true` if any active risk blocks it. Built-in risk sources and
interventions share this single risk-evaluation path."""
function _risk_blocks(source, parent, contact, state, transmission_time)
    rng = state.rng
    for risk in _iter_risks(competing_risk(source, parent, contact, state))
        event_t = _sample_value(risk.event_time, rng, parent, contact, state)
        event_t > transmission_time && continue
        prob = _sample_value(risk.block_probability, rng, parent, contact, state)
        prob <= 0.0 && continue
        prob >= 1.0 && return true
        rand(rng) < prob && return true
    end
    return false
end

"""
    transmission_risks(model) -> iterable of risk sources

Risk sources the *model* contributes to competing-risks resolution, on the same
[`competing_risk`](@ref) surface as the built-ins and interventions. A
structure-driven model whose transmission probability is a property of the
*edge* — `NetworkProcess`'s per-edge probability, a metapopulation's coupling —
returns a source here so that probability is a competing risk on every potential
contact (the contact is still produced and seen by `apply_post_transmission!`),
rather than a filter that drops contacts before they reach the engine. Defaults
to none, so offspring-driven models are unaffected.
"""
transmission_risks(::TransmissionModel) = ()

"""
Decide whether `contact` is infected by its parent. The transmission risks act as
competing hazards, and the first to block wins. An index case (no parent) is
always infected. Otherwise finite-population depletion
([`susceptible_fraction`](@ref)) applies first, then each risk source in turn: the
built-in host susceptibility and infector infectiousness, then any from the
model's [`transmission_risks`](@ref), then the interventions. Returns `true` when
the contact is infected.
"""
function _decide_infected(state::SimulationState, contact::Individual,
        model_risks, interventions, infected_so_far::Int)
    contact.parent_id == 0 && return true
    rng = state.rng
    parent = state.individuals[contact.parent_id]
    transmission_time = contact.infection_time

    # Finite-population depletion (dispatched on the population type).
    pop_suscept = susceptible_fraction(state, infected_so_far)
    pop_suscept <= 0.0 && return false
    pop_suscept < 1.0 && rand(rng) > pop_suscept && return false

    # All transmission risks — susceptibility and infectiousness, then any the
    # model contributes, then interventions — on one surface, applied in order;
    # first to block wins.
    for source in _BUILTIN_RISK_SOURCES
        _risk_blocks(source, parent, contact, state, transmission_time) && return false
    end
    for source in model_risks
        _risk_blocks(source, parent, contact, state, transmission_time) && return false
    end
    for intervention in interventions
        _risk_blocks(intervention, parent, contact, state, transmission_time) &&
            return false
    end
    return true
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
_apply_attributes!(f::Function, rng, ind) = f(rng, ind)
function _apply_attributes!(builders::Union{Tuple, AbstractVector}, rng, ind)
    for build! in builders
        build!(rng, ind)
    end
    return nothing
end

# ── Attributes function constructors ─────────────────────────────────

"""
    clinical_presentation(; incubation_period, prob_asymptomatic = 0.0)

Return an attributes function that sets `:onset_time` and
`:asymptomatic` on each individual.

For symptomatic cases, `:onset_time = infection_time + rand(incubation_period)`.
For asymptomatic cases (drawn with probability `prob_asymptomatic`),
`:onset_time = NaN` and `:asymptomatic = true`. Required by
[`Isolation`](@ref) and used by [`linelist`](@ref) to populate
`date_onset`.

`prob_asymptomatic` accepts a `Real`, a `Distribution`, or a function
`(rng, ind) -> Real`. Use the function form for age- or
state-conditional asymptomatic fractions.

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
function clinical_presentation(; incubation_period::Distribution,
        prob_asymptomatic::Union{Real, Distribution, Function} = 0.0)
    return function (rng, ind)
        pa = _sample_value(prob_asymptomatic, rng, ind)
        is_asymp = rand(rng) < pa
        ind.state[:asymptomatic] = is_asymp
        # Incubation period is a host property; :onset_time follows from
        # it and the infection time via `_set_onset_from_incubation!`.
        ind.state[:incubation_period] = is_asymp ? NaN : rand(rng, incubation_period)
        _set_onset_from_incubation!(ind)
    end
end

"""
    _set_onset_from_incubation!(ind)

Set `:onset_time` to `infection_time + :incubation_period` from the
stored host incubation period. Asymptomatic individuals (`NaN`
incubation) get a `NaN` onset. Applies when `:incubation_period` is
present on the individual.
"""
function _set_onset_from_incubation!(ind::Individual)
    haskey(ind.state, :incubation_period) || return nothing
    inc = ind.state[:incubation_period]::Float64
    ind.state[:onset_time] = isnan(inc) ? NaN : ind.infection_time + inc
    return nothing
end

"""
    demographics(; age_distribution=nothing, age_range=(0, 90), prob_female=0.5)

Return an attributes function. `:age` and `:sex` are set on each individual.
"""
function demographics(;
        age_distribution::Union{Distribution, NoAgeDistribution} = NoAgeDistribution(),
        age_range::Tuple{Int, Int} = (0, 90),
        prob_female::Real = 0.5)
    pf = float(prob_female)
    return function (rng, ind)
        ind.state[:age] = _sample_age(rng, age_distribution, age_range)
        ind.state[:sex] = rand(rng) < pf ? :female : :male
    end
end

_sample_age(rng, ::NoAgeDistribution, age_range) = rand(rng, age_range[1]:age_range[2])
function _sample_age(rng, dist::Distribution, age_range)
    clamp(floor(Int, rand(rng, dist)), age_range...)
end

"""
    transmission_traits(; susceptibility = 1.0, infectiousness = 1.0)

Return an attributes function that sets `susceptibility` (per-contact
probability of infection given exposure) and `infectiousness` (parent-side
modifier on transmission) on each individual.

Each argument accepts:

- a `Real`: assigned directly to every individual.
- a `Distribution`: sampled per individual via `rand(rng, dist)`.
- a `Function` `(rng, ind) -> value`: called per individual; the
  returned value is assigned. Use this for attribute-dependent rules
  (e.g. age-conditional susceptibility) — place the builder after
  `demographics` in the attributes list so `ind.state[:age]` is set first.

Both default to `1.0` (no Bernoulli filtering in the transmission model).

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

The closure form `(rng, ind) -> (ind.susceptibility = ...)` as a list
entry remains available as an escape hatch for cases this builder does
not cover.

See also [`clinical_presentation`](@ref), [`demographics`](@ref).
"""
function transmission_traits(;
        susceptibility::Union{Real, Distribution, Function} = 1.0,
        infectiousness::Union{Real, Distribution, Function} = 1.0)
    sus = _trait_sampler(susceptibility)
    inf = _trait_sampler(infectiousness)
    return function (rng, ind)
        ind.susceptibility = sus(rng, ind)
        ind.infectiousness = inf(rng, ind)
    end
end

_trait_sampler(x::Real) =
    let v = float(x)
        (rng, ind) -> v
    end
_trait_sampler(d::Distribution) = (rng, ind) -> float(rand(rng, d))
_trait_sampler(f::Function) = (rng, ind) -> float(f(rng, ind))

# ── Intervention field validation ────────────────────────────────────

"""Fields that an intervention requires on individuals. Default: none."""
required_fields(::AbstractIntervention) = Symbol[]

"""Check that all required fields are present on an individual. Works for
any iterable of items that define `required_fields` — used for both
interventions and clinical transitions."""
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
end

function _field_hint(field::Symbol)
    hints = Dict(
        :onset_time => "Provide attributes = clinical_presentation(incubation_period = ...).",
        :asymptomatic => "Provide attributes = clinical_presentation(incubation_period = ...).",
        :age => "Provide attributes = demographics(age_distribution = ...).",
        :sex => "Provide attributes = demographics(...)."
    )
    return get(hints, field, "Set this field via an attributes function.")
end
