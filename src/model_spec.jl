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
# these can have no effect on a simulation or on the likelihood; see
# `_warn_inert_interventions`. `susceptibility_components` is on the list
# because an intervention whose effect is declared for the likelihood alone,
# as the extending guide's `Prophylaxis` is, reaches the model through it. A
# function rather than a constant, since `susceptibility_components` is
# defined in a file loaded after this one.
_effect_hooks() = (
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
    (
        susceptibility_components, 2,
        "susceptibility_components(intervention, host)",
    ),
)

# The arities the engine calls each hook with, gathered per hook, in the
# order `_effect_hooks()` lists them (`trace_contacts!` is the one hook with
# two).
function _hook_arities()
    out = Tuple{Any, Vector{Int}, Vector{String}}[]
    for (f, n, sig) in _effect_hooks()
        i = findfirst(entry -> entry[1] === f, out)
        if i === nothing
            push!(out, (f, [n], [sig]))
        else
            push!(out[i][2], n)
            push!(out[i][3], sig)
        end
    end
    return out
end

# Whether the author's own module holds a function of the hook's name that is
# not the package's, carrying a method that was written to be this hook: what
# a hook defined without the `EpiBranch.` prefix leaves behind. Looked for in
# `parentmodule(T)`, since the hook and the type are written together.
#
# The method has to look like the hook and not merely share its name: an
# intervention type first, and one of the arities the engine calls. Without
# both, an author's own helper that happens to be called `trace_contacts!`
# is reported as a mis-written hook, and the advice to add the `EpiBranch.`
# prefix would break it.
function _shadowing_hook(f, T::Type, arities)
    name = nameof(f)
    mod = parentmodule(T)
    isdefined(mod, name) || return false
    g = getglobal(mod, name)
    (g isa Function && g !== f) || return false
    return any(methods(g)) do mm
        params = Base.unwrap_unionall(mm.sig).parameters
        length(params) < 2 && return false
        if Base.isvarargtype(params[end])
            any(n -> n + 1 >= length(params) - 1, arities) || return false
        else
            (length(params) - 1) in arities || return false
        end
        p = params[2]
        p isa TypeVar && (p = p.ub)
        p isa Type || return false
        p !== AbstractIntervention && p <: AbstractIntervention && T <: p
    end
end

# The type a method of `f` was written for, where that method is one `T` can
# never reach: written at an arity the engine does not call, with no method at
# a called arity written for a type at least as specific to stand in for it.
# `nothing` when every method `T` has is either reachable or covered that way.
#
# Asked per method rather than per hook, because a correct method on an
# abstract intervention parent would otherwise excuse a child's broken
# override: the parent's method answers the call, so the child's own block
# never applies, which is the mistake this check exists to name.
function _unreachable_own_method(f, T::Type, base::Type, arities)
    own = Tuple{Type, Bool}[]
    for mm in methods(f, Tuple{T, Vararg{Any}})
        params = Base.unwrap_unionall(mm.sig).parameters
        length(params) < 2 && continue
        p = params[2]
        p isa TypeVar && (p = p.ub)
        p isa Type || continue
        (p !== base && p <: base) || continue
        called = if Base.isvarargtype(params[end])
            any(n -> n + 1 >= length(params) - 1, arities)
        else
            (length(params) - 1) in arities
        end
        push!(own, (p, called))
    end
    for (p, called) in own
        called && continue
        any(entry -> entry[2] && entry[1] <: p, own) && continue
        return p
    end
    return nothing
end

# Warn, per hook, about a method an intervention meant as that hook which the
# engine will never call: one written at an arity no call uses, or one added
# without the `EpiBranch.` prefix and so belonging to a new function of the
# same name. Either way the hook reads correctly and the no-op default runs.
# Returns how many it found, so the caller can tell a diagnosis from silence.
function _warn_hook_mistakes(T::Type)
    found = 0
    for (f, arities, sigs) in _hook_arities()
        unreachable = _unreachable_own_method(f, T, AbstractIntervention, arities)
        if unreachable !== nothing
            found += 1
            @warn "$(nameof(unreachable)) has a method of its own for " *
                "$(nameof(f)), but at no number of arguments the engine calls " *
                "it with, so that method is never reached: the engine uses the " *
                "next one that applies to $(nameof(T)), which is the no-op " *
                "default unless a wider intervention type supplies one. The " *
                "engine calls: $(join(sigs, ", "))." intervention = T
        elseif !_has_own_method(f, T, AbstractIntervention) &&
                _shadowing_hook(f, T, arities)
            found += 1
            @warn "$(parentmodule(T)) defines a method of $(nameof(f)) for " *
                "$(nameof(T)), but on its own function of that name rather " *
                "than the package's, so the engine never calls it. Write it " *
                "as `EpiBranch.$(nameof(f))(...)`." intervention = T
        end
    end
    return found
end

# Each intervention type is examined once per state of the method tables: the
# answer depends only on the type and the methods written for it, and
# `ModelSpec` is built inside an inference loop in the documented Turing
# recipe, where repeating the method scans costs far more than the checks are
# worth. Such a loop defines no methods, so the world counter holds still and
# each type is scanned once; a later definition, from an editor reload or a
# hook rewritten in the REPL, advances it and arms the check again.
const _CHECKED_INTERVENTIONS = Dict{Type, UInt}()
const _CHECKED_INTERVENTIONS_LOCK = ReentrantLock()

function _first_sight(T::Type)
    world = Base.get_world_counter()
    return lock(_CHECKED_INTERVENTIONS_LOCK) do
        get(_CHECKED_INTERVENTIONS, T, nothing) == world && return false
        _CHECKED_INTERVENTIONS[T] = world
        return true
    end
end

# Warn when a composed intervention has a method of its own for none of
# `_effect_hooks()`, at the right arity: a hook written with the wrong number
# of arguments, or defined without the `EpiBranch.` qualification needed to
# add a method rather than shadow it, leaves only the `AbstractIntervention`
# fallback reachable, and the intervention does nothing with no sign that
# anything is missing. Checked on the type an intervention wraps, if any,
# since a wrapper such as `Scheduled` always has its own delegating methods
# whatever it wraps.
function _warn_inert_interventions(interventions)
    for iv in interventions
        T = typeof(_unwrap_scheduled(iv))
        _first_sight(T) || continue
        reachable = any(_effect_hooks()) do (f, n, _)
            _has_own_method(f, T, AbstractIntervention, n)
        end
        # A diagnosis of the hook at fault stands in for the general message,
        # which lists every hook and has to guess at the cause.
        named = _warn_hook_mistakes(T)
        (reachable || named > 0) && continue
        @warn "$(nameof(T)) has a method of its own for none of the hooks the " *
            "package calls, so it can have no effect on the simulation or on " *
            "the likelihood: " *
            "$(join((sig for (_, _, sig) in _effect_hooks()), ", ")). This is usually " *
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
