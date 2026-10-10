# ── ModelSpec ────────────────────────────────────────────────────────
#
# A model specification: a transmission process together with the modelling
# layers that force and observe it. The process is the reusable kernel — the
# pathogen you fit and hold fixed while varying policy — and the spec adds the
# within-host progression, the interventions, the per-individual attributes,
# and the observation model. The observed data is not held here; it is a
# `loglikelihood` argument. See the design notes in `docs/src/design.md`.
#
# A `ModelSpec` is not itself a process. `simulate`/`loglikelihood` unwrap it:
# the process is the dispatched model, and the spec's layers are passed in as
# the forcing inputs the engine already threads. So there are no per-method
# forwards — only the entry points know about the spec.

# Warn when every terminal transition in `progression` is independently gated
# below certainty: with none that always occurs, a case can clear every gate
# and reach no terminal state at all — `:outcome` stays unset and, on a
# structure-driven model, the infectious window this case opened never closes
# (a different symptom of the same gap that `_warn_uncovered_terminal_states`,
# in branching_process.jl, catches for states missing from `until`). Skipped
# when any terminal's `terminal_certainty` is `missing`: unknowable without an
# individual (or a transition type that hasn't declared it), so silence over a
# possible false warning.
function _warn_incomplete_terminal_coverage(progression)
    terminals = filter(is_terminal, progression)
    isempty(terminals) && return nothing
    certainties = Union{Bool, Missing}[terminal_certainty(t) for t in terminals]
    any(isequal(true), certainties) && return nothing
    any(ismissing, certainties) && return nothing
    @warn "Every terminal transition in `progression` is gated below " *
        "probability 1, and independent gates do not give an exclusive " *
        "outcome: a case can clear every gate and reach no terminal state, " *
        "leaving `:outcome` unset and, on a structure-driven model, its " *
        "infectious window never closed. If the outcomes are meant to " *
        "partition the population exactly (an exact case-fatality ratio, " *
        "say), build their probabilities with `exclusive_probabilities`; " *
        "otherwise add an unconditional terminal transition to guarantee " *
        "every case ends somewhere."
    return nothing
end

# The hooks whose default on `AbstractIntervention` is a no-op, together with
# the signature the engine calls each with (`trace_contacts!` has two, the
# four-argument one being what the five-argument fallback calls in turn). An
# intervention with no method of its own, at the right arity, for any of
# these can have no effect on a simulation; see `_warn_inert_interventions`.
const _EFFECT_HOOKS = (
    (initialise_individual!, 3, "initialise_individual!(intervention, individual, state)"),
    (resolve_individual!, 3, "resolve_individual!(intervention, individual, state)"),
    (
        apply_post_transmission!, 3,
        "apply_post_transmission!(intervention, state, new_contacts)",
    ),
    (competing_risk, 4, "competing_risk(intervention, parent, contact, state)"),
    (trace_contacts!, 4, "trace_contacts!(intervention, state, infector, contacts)"),
    (
        trace_contacts!, 5,
        "trace_contacts!(intervention, state, infector, contacts, not_before)",
    ),
    (keep_active, 4, "keep_active(intervention, state, targets, is_new)"),
    (
        on_infection_settled!, 4,
        "on_infection_settled!(intervention, individual, state, rng)",
    ),
    (infectious_removal_time, 2, "infectious_removal_time(intervention, individual)"),
    (intervention_actions, 3, "intervention_actions(intervention, state, candidates)"),
)

# Warn when a composed intervention has a method of its own for none of
# `_EFFECT_HOOKS`, at the right arity: a hook written with the wrong number
# of arguments, or defined without the `EpiBranch.` qualification needed to
# add a method rather than shadow it, leaves only the `AbstractIntervention`
# fallback reachable, and the intervention does nothing with no sign that
# anything is missing. Checked on the type an intervention wraps, if any,
# since a wrapper such as `Scheduled` always has its own delegating methods
# whatever it wraps.
function _warn_inert_interventions(interventions)
    for iv in interventions
        T = typeof(_unwrap_scheduled(iv))
        any(_EFFECT_HOOKS) do (f, n, _)
            _has_own_method(f, T, AbstractIntervention, n)
        end && continue
        @warn "$(nameof(T)) has a method of its own for none of the hooks the " *
            "engine calls, so it can have no effect on the simulation: " *
            "$(join((sig for (_, _, sig) in _EFFECT_HOOKS), ", ")). This is usually " *
            "a hook defined with the wrong number of arguments, or without the " *
            "`EpiBranch.` prefix needed to add a method to the package's " *
            "function instead of defining a new one of the same name." intervention = T
    end
    return nothing
end

struct ModelSpec{P <: TransmissionModel, A, O, C}
    process::P
    progression::Vector{AbstractClinicalTransition}
    interventions::Vector{AbstractIntervention}
    attributes::A
    observation::O
    recorder::C
end

"""
    ModelSpec(process; progression, interventions, attributes, observation, recorder)

Compose a transmission `process` with the modelling layers
that force and observe it: the within-host `progression`, the `interventions`,
the per-individual `attributes`, the `observation` model, and the `recorder`
([`ContactRecorder`](@ref)) that tells a continuous-time race which
standing-blocked pairs to keep drawing. Each keyword defaults to the value
already on `process`, so `ModelSpec(process)` wraps it faithfully and the
keywords override layer by layer.

`simulate(spec)` runs it; `loglikelihood(data, spec)` evaluates observed `data`
against it. The observations themselves stay outside the spec, as the
likelihood argument.
"""
function ModelSpec(
        process::TransmissionModel;
        progression = _progression(process),
        interventions = interventions(process),
        attributes = attributes(process),
        observation = observation(process),
        recorder = recorder(process)
    )
    prog = _progvec(progression)
    _validate_process_windows(process, prog)
    _warn_incomplete_terminal_coverage(prog)
    ivs = _intervention_vector(interventions)
    _validate_dose_schedule(ivs)
    _warn_inert_interventions(ivs)
    return ModelSpec(process, prog, ivs, attributes, observation, recorder)
end

# Convenience accessors — the spec's own modelling layers.
interventions(s::ModelSpec) = s.interventions
attributes(s::ModelSpec) = s.attributes
observation(s::ModelSpec) = s.observation
recorder(s::ModelSpec) = s.recorder
_progression(s::ModelSpec) = s.progression
population_size(s::ModelSpec) = population_size(s.process)

# Structural accessors delegate to the wrapped process, so the analytical
# helpers and the Turing `~` distribution wrappers treat a spec like the
# process it wraps.
single_type_offspring(s::ModelSpec) = single_type_offspring(s.process)
_analytic_offspring(s::ModelSpec) = _analytic_offspring(s.process)
n_types(s::ModelSpec) = n_types(s.process)
_single_kernel(s::ModelSpec) = _single_kernel(s.process)

# `simulate` unwraps the spec: the process is the model, the spec's layers are
# the forcing inputs.
function simulate(
        spec::ModelSpec;
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
    _warn_ignored_termination(
        spec.process, max_cases, max_generations, max_time, stopping_rules
    )
    _warn_unhonoured_interventions(spec.process, spec.interventions)
    sim_opts = SimOpts(;
        n_initial, initial_cases, max_cases, max_generations, max_time,
        stopping_rules
    )
    _validate_initial_cases(spec.process, sim_opts)
    return _simulate(
        spec.process, sim_opts; interventions = spec.interventions,
        attributes = spec.attributes, progression = spec.progression,
        observation = spec.observation, recorder = spec.recorder, rng, condition,
        max_attempts
    )
end

function simulate(
        spec::ModelSpec, n::Int;
        n_initial::Union{Int, Nothing} = nothing,
        initial_cases::Union{AbstractVector{<:Integer}, Nothing} = nothing,
        max_cases::Union{Int, Nothing} = _DEFAULT_MAX_CASES,
        max_generations::Union{Int, Nothing} = _DEFAULT_MAX_GENERATIONS,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng(),
        parallel::Bool = false
    )
    _warn_ignored_termination(
        spec.process, max_cases, max_generations, max_time, stopping_rules
    )
    _warn_unhonoured_interventions(spec.process, spec.interventions)
    sim_opts = SimOpts(;
        n_initial, initial_cases, max_cases, max_generations, max_time,
        stopping_rules
    )
    _validate_initial_cases(spec.process, sim_opts)
    return _simulate_n(
        spec.process, n, sim_opts;
        interventions = spec.interventions, attributes = spec.attributes,
        progression = spec.progression, observation = spec.observation,
        recorder = spec.recorder, rng, parallel
    )
end

function Base.show(io::IO, s::ModelSpec)
    return print(
        io, "ModelSpec(", s.process, "; ", length(s.interventions),
        " interventions, ", length(s.progression), " transitions)"
    )
end
