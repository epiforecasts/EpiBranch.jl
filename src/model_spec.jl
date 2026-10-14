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
# never acts, with nothing to say so.
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
    return _run_once_or_retry(
        spec.process, sim_opts, condition, max_attempts;
        interventions = spec.interventions, attributes = spec.attributes,
        progression = spec.progression, observation = spec.observation,
        recorder = spec.recorder, rng
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

# ── Interventions in the analytical functions ────────────────────────
#
# Each analytical function reads the offspring distribution and passes it
# through every intervention the model holds, on a spec or on a process that
# stores its own, via `analytic_offspring_effect`. An intervention without a
# method reaches the generic one, which throws.
function _offspring_through_interventions(model, offspring)
    for iv in interventions(model)
        offspring = analytic_offspring_effect(iv, offspring)
    end
    return offspring
end

function Base.show(io::IO, s::ModelSpec)
    return print(
        io, "ModelSpec(", s.process, "; ", length(s.interventions),
        " interventions, ", length(s.progression), " transitions)"
    )
end
