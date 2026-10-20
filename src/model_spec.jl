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

# The hooks the engine dispatches on an intervention, together with the
# signature it calls each with (`trace_contacts!` has two, the
# four-argument one being what the five-argument fallback calls in turn). An
# intervention with no method of its own, at the right arity, for any of
# these can have no effect on a simulation or on the likelihood, which is what
# the default `has_effect` answers from. `susceptibility_components` is on the list
# because an intervention whose effect is declared for the likelihood alone,
# as the extending guide's `Prophylaxis` is, reaches the model through it, and
# `analytic_offspring_effect` because an intervention that acts on the
# offspring law in closed form reaches the analytical results through that one
# and nothing else. A function rather than a constant, since
# `susceptibility_components` is defined in a file loaded after this one.
_effect_hooks() = (
    (initialise_individual!, 3, "initialise_individual!(intervention, individual, state)"),
    (resolve_individual!, 3, "resolve_individual!(intervention, individual, state)"),
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
    (
        analytic_offspring_effect, 2,
        "analytic_offspring_effect(intervention, offspring)",
    ),
    (
        analytic_offspring_effect, 3,
        "analytic_offspring_effect(intervention, offspring, spec)",
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

# A hook written for a `Union` of intervention types answers a call for each
# of its members, so the specificity questions below treat it as written
# separately for each: a union that names the type covers it the way a method
# for the type itself would. `<:` alone says the opposite, since a union is a
# subtype of neither member.
function _member_types(p::Type)
    body = Base.unwrap_unionall(p)
    body isa Union || return (p,)
    return map(u -> Base.rewrap_unionall(u, p), Base.uniontypes(body))
end
function _covers(p::Type, bound::Type)
    return any(_member_types(p)) do u
        any(b -> u <: b, _member_types(bound))
    end
end

# A name for a method's first parameter in a warning. Only a plain type gets
# the short `nameof`: `nameof` has no method for a `Union`, which a hook shared
# by two intervention types leaves here, and it strips the parameters from a
# parametric type, which would print `Shield{Float64}` and `Shield` alike and
# leave the warning naming the same type twice. A whole parametric family,
# `Shield` for every `T`, is named the same way as a plain type.
function _hook_type_name(p::Type)
    body = Base.unwrap_unionall(p)
    return body isa DataType && p === body.name.wrapper ? string(nameof(body)) :
        string(p)
end

# The intervention type a hook method was written for: its first parameter,
# with the method's own type variables kept. Unwrapping the signature alone
# leaves `Shield{T}` with `T` free, which no `Shield{Float64}` is a subtype of,
# so a hook written `where {T}` would match nothing and two such methods would
# never cover each other. `nothing` where that parameter is not a type, as
# with a hook written with every argument variadic.
function _hook_param(mm::Method)
    p = Base.unwrap_unionall(mm.sig).parameters[2]
    p isa TypeVar && (p = p.ub)
    p isa Type || return nothing
    return Base.rewrap_unionall(p, mm.sig)
end

# The type a shadowing method of the hook's name was written for, and the
# module it was written in, where that name resolves in the author's own
# module to a function that is not the package's: what a hook defined without
# the `EpiBranch.` prefix leaves behind. `nothing` when there is none. Looked
# for in `parentmodule(T)`, since the hook and the type are written together,
# but reported against the method's own module, which is where the line to fix
# is: a family in a base module and its concrete types in a sibling put the
# two in different places.
#
# The method has to look like the hook and not merely share its name: an
# intervention type first, and exactly one of the arities the engine calls.
# Without both, an author's own helper that happens to be called
# `trace_contacts!` is reported as a mis-written hook, and the advice to add
# the `EpiBranch.` prefix would break it. A variadic method is left out for
# the same reason: it answers every arity, so it is a helper far more often
# than a hook, and prefixing one would have it answer calls its author never
# wrote it for.
function _shadowing_hook(f, T::Type, arities)
    name = nameof(f)
    mod = parentmodule(T)
    isdefined(mod, name) || return nothing
    g = getglobal(mod, name)
    (g isa Function && g !== f) || return nothing
    for mm in methods(g)
        params = Base.unwrap_unionall(mm.sig).parameters
        length(params) < 2 && continue
        Base.isvarargtype(params[end]) && continue
        (length(params) - 1) in arities || continue
        p = _hook_param(mm)
        p === nothing && continue
        p === AbstractIntervention && continue
        p <: AbstractIntervention && T <: p && return (p, mm.module)
    end
    return nothing
end

# Whether `T` reaches a method of `f` at an arity the engine calls, written
# for a type at least as specific as `bound`: the method that would stand in
# for one of `T`'s own. Asked with the shadowing method's own first parameter
# as `bound`, so a hook inherited from an abstract intervention parent does
# not excuse a missing prefix on the child's override, the same reasoning
# `_unreachable_own_method` is asked per method for.
function _reaches_own_method(f, T::Type, bound::Type, arities)
    return any(methods(f, Tuple{T, Vararg{Any}})) do mm
        params = Base.unwrap_unionall(mm.sig).parameters
        length(params) < 2 && return false
        called = if Base.isvarargtype(params[end])
            any(n -> n + 1 >= length(params) - 1, arities)
        else
            (length(params) - 1) in arities
        end
        called || return false
        p = _hook_param(mm)
        p === nothing && return false
        return _covers(p, bound)
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
        p = _hook_param(mm)
        p === nothing && continue
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
        any(entry -> entry[2] && _covers(entry[1], p), own) && continue
        return p
    end
    return nothing
end

# The type whose own method of `f` the engine reaches, where `T` has a method
# of its own that no call does: the method that stands in for it. `nothing`
# when nothing between `T` and `base` supplies one, so the package's own
# default is what runs. Collected the same way as above, so the warning can name what the
# engine uses rather than hedge about it.
#
# Ranked by `Base.morespecific`, the rule dispatch itself uses, rather than by
# `_covers`: a union and a member of it cover each other, so a predicate would
# leave the answer to the order `methods` happens to list them in, and that
# order is not the same on every Julia version.
function _standin_own_method(f, T::Type, base::Type, arities)
    standin = nothing
    standin_sig = nothing
    for mm in methods(f, Tuple{T, Vararg{Any}})
        params = Base.unwrap_unionall(mm.sig).parameters
        length(params) < 2 && continue
        p = _hook_param(mm)
        p === nothing && continue
        (p !== base && p <: base) || continue
        called = if Base.isvarargtype(params[end])
            any(n -> n + 1 >= length(params) - 1, arities)
        else
            (length(params) - 1) in arities
        end
        called || continue
        if standin === nothing || Base.morespecific(mm.sig, standin_sig)
            standin, standin_sig = p, mm.sig
        end
    end
    return standin
end

# Warn, per hook, about a method an intervention meant as that hook which the
# engine will never call: one written at an arity no call uses, or one added
# without the `EpiBranch.` prefix and so belonging to a new function of the
# same name. Either way the hook reads correctly and the package's own default
# runs: a no-op for most of them, and an error for `analytic_offspring_effect`.
# Returns how many it found, so the caller can tell a diagnosis from silence.
function _warn_hook_mistakes(T::Type)
    found = 0
    for (f, arities, sigs) in _hook_arities()
        unreachable = _unreachable_own_method(f, T, AbstractIntervention, arities)
        if unreachable !== nothing
            found += 1
            standin = _standin_own_method(f, T, AbstractIntervention, arities)
            instead = standin === nothing ?
                "the package's own default for $(nameof(f))" :
                "$(_hook_type_name(standin))'s method"
            @warn "$(_hook_type_name(unreachable)) has a method of its own for " *
                "$(nameof(f)), but at no number of arguments the engine calls " *
                "it with, so that method is never reached: the engine uses " *
                "$(instead) instead. The engine calls: " *
                "$(join(sigs, ", "))." intervention = T
            continue
        end
        stray = _shadowing_hook(f, T, arities)
        stray === nothing && continue
        shadowed, stray_module = stray
        _reaches_own_method(f, T, shadowed, arities) && continue
        found += 1
        @warn "$(stray_module) defines a method of $(nameof(f)) for " *
            "$(_hook_type_name(shadowed)), but on its own function of that " *
            "name rather than the package's, so the engine never calls it. " *
            "Write it as `EpiBranch.$(nameof(f))(...)`." intervention = T
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

# Warn when a composed intervention answers `has_effect` with `false`, which by
# default means it has a method of its own for none of `_effect_hooks()` at the
# right arity: a hook written with the wrong number of arguments, or defined
# without the `EpiBranch.` qualification needed to add a method rather than
# shadow it, leaves only the `AbstractIntervention` fallback reachable, and the
# intervention does nothing with no sign that anything is missing. Checked on
# the intervention a wrapper holds, if any, since a wrapper such as `Scheduled`
# always has its own delegating methods whatever it wraps.
function _warn_inert_interventions(interventions)
    for iv in interventions
        inner = _unwrap_scheduled(iv)
        T = typeof(inner)
        _first_sight(T) || continue
        # A diagnosis of the hook at fault stands in for the general message,
        # which lists every hook and has to guess at the cause.
        named = _warn_hook_mistakes(T)
        (named > 0 || has_effect(inner)) && continue
        @warn "$(_hook_type_name(T)) has a method of its own for none of the " *
            "hooks the package calls, so it can have no effect on the " *
            "simulation or on the likelihood: " *
            "$(join((sig for (_, _, sig) in _effect_hooks()), ", ")). This is usually " *
            "a hook defined with the wrong number of arguments, or without the " *
            "`EpiBranch.` prefix needed to add a method to the package's " *
            "function instead of defining a new one of the same name. An " *
            "intervention that acts through some other seam says so with " *
            "`EpiBranch.has_effect(::$(_hook_type_name(T))) = true`." intervention = T
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

# The `start_after` triggers a composed intervention holds, looked for through
# any wrapper around it.
_triggers_of(::AbstractIntervention) = ()
_triggers_of(w::InterventionWrapper) = _triggers_of(w.intervention)
_triggers_of(s::Scheduled) = (s.triggers..., _triggers_of(s.intervention)...)

# Warn when a trigger needs a transition the model's `progression` does not
# hold. The keys a trigger reads are written by the progression, so without
# the transition the gate stays shut for the whole run and the intervention
# never acts, with nothing to flag it.
function _warn_unmet_trigger_requirements(interventions, progression)
    for iv in interventions
        for trigger in _triggers_of(iv)
            for needed in required_transitions(trigger)
                any(t -> t isa needed, progression) && continue
                @warn "$(nameof(typeof(trigger))) needs a " *
                    "$(nameof(needed)) transition in `progression` to read " *
                    "the state it counts, and this model has none, so the " *
                    "schedule it starts will never open and the " *
                    "intervention it wraps will never act." trigger =
                    typeof(trigger) required = needed
            end
        end
    end
    return nothing
end

"""
    ModelSpec(process; progression, interventions, attributes, observation, recorder)

Combine a transmission model (who infects whom, and when) with the rest of
an outbreak scenario. Pass the result to [`simulate`](@ref), or to
`loglikelihood(data, model)` to compare it with observed data (the data are
not stored in the model).

- `progression`: the natural history of a case, as a vector of
  [`Transition`](@ref)s and other steps such as [`Recovery`](@ref) and
  [`Death`](@ref): latent period, symptom onset, hospitalisation, outcome.
- `interventions`: control measures, such as [`Isolation`](@ref),
  [`ContactTracing`](@ref) and [`RingVaccination`](@ref).
- `attributes`: population characteristics drawn for each person, such as
  [`clinical_presentation`](@ref) (incubation period, asymptomatic fraction)
  and [`demographics`](@ref) (age, sex). Give one, or several in a vector.
- `observation`: which outbreaks are recorded, such as
  [`MinimumSize`](@ref) (only chains of at least a given size). Report
  individual cases with a [`Reporting`](@ref) step in `progression`.
- `recorder`: rarely needed; see [`ContactRecorder`](@ref).

Every keyword left out keeps the value the transmission model already has,
so `ModelSpec(process)` behaves exactly like `process`.

# Examples
```julia
using EpiBranch, Distributions

model = ModelSpec(
    BranchingProcess(NegBin(2.5, 0.16), Gamma(2.5, 2.0));
    attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
    interventions = [Isolation(onset_to_isolation_delay = Exponential(2.0),
        duration = 14.0)],
)
state = simulate(model; max_cases = 500)
```
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
    _warn_unmet_trigger_requirements(ivs, prog)
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
has_single_type_law(s::ModelSpec) = has_single_type_law(s.process)
offspring_law_unchanged(s::ModelSpec) = offspring_law_unchanged(s.process)
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
    _warn_contactless(spec.process, spec.interventions)
    sim_opts = _build_sim_opts(
        spec.process, n_initial, initial_cases, max_cases, max_generations,
        max_time, stopping_rules
    )
    _validate_initial_cases(spec.process, sim_opts)
    return _run_once_or_retry(
        spec.process, sim_opts, condition, max_attempts;
        interventions = spec.interventions, attributes = spec.attributes,
        progression = spec.progression, observation = spec.observation,
        recorder = spec.recorder, rng
    )
end

simulate(spec::ModelSpec, n::Int; kwargs...) =
    _simulate_each(identity, SimulationState, spec, n; kwargs...)

function _simulate_each(
        each, ::Type{T}, spec::ModelSpec, n::Int;
        n_initial::Union{Int, Nothing} = nothing,
        initial_cases::Union{AbstractVector{<:Integer}, Nothing} = nothing,
        max_cases::Union{Int, Nothing} = _DEFAULT_MAX_CASES,
        max_generations::Union{Int, Nothing} = _DEFAULT_MAX_GENERATIONS,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng(),
        parallel::Bool = false
    ) where {T}
    _warn_contactless(spec.process, spec.interventions)
    sim_opts = _build_sim_opts(
        spec.process, n_initial, initial_cases, max_cases, max_generations,
        max_time, stopping_rules
    )
    _validate_initial_cases(spec.process, sim_opts)
    return _simulate_n(
        each, T, spec.process, n, sim_opts;
        interventions = spec.interventions, attributes = spec.attributes,
        progression = spec.progression, observation = spec.observation,
        recorder = spec.recorder, rng, parallel
    )
end

# ── Interventions in the analytical functions ────────────────────────
#
# Each analytical function reads the offspring distribution and passes it
# through every intervention the model holds, on a spec or on a process that
# stores its own, via `analytic_offspring_effect`. The three-argument form
# gives an intervention the model itself, for an effect that needs more than
# the offspring law (Isolation's generation time and incubation period, say);
# its default falls back to the two-argument one. An intervention with a
# method of neither reaches the generic default, which throws.
function _offspring_through_interventions(model, offspring)
    for iv in interventions(model)
        offspring = analytic_offspring_effect(iv, offspring, model)
    end
    return offspring
end

function Base.show(io::IO, s::ModelSpec)
    return print(
        io, "ModelSpec(", s.process, "; ", length(s.interventions),
        " interventions, ", length(s.progression), " transitions)"
    )
end
