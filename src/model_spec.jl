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

struct ModelSpec{P <: TransmissionModel, A, O}
    process::P
    progression::Vector{AbstractClinicalTransition}
    interventions::Vector{AbstractIntervention}
    attributes::A
    observation::O
end

"""
    ModelSpec(process; progression, interventions, attributes, observation)

Compose a transmission `process` (the pure kernel) with the modelling layers
that force and observe it: the within-host `progression`, the `interventions`,
the per-individual `attributes`, and the `observation` model. Each keyword
defaults to the value already on `process`, so `ModelSpec(process)` wraps it
faithfully and the keywords override layer by layer.

`simulate(spec)` runs it; `loglikelihood(data, spec)` scores observed `data`
against it. The observations themselves stay outside the spec, as the
likelihood argument.
"""
function ModelSpec(process::TransmissionModel;
        progression = _progression(process),
        interventions = interventions(process),
        attributes = attributes(process),
        observation = observation(process))
    prog = _progvec(progression)
    _validate_process_windows(process, prog)
    return ModelSpec(process, prog,
        _intervention_vector(interventions), attributes, observation)
end

# Convenience accessors — the spec's own modelling layers.
interventions(s::ModelSpec) = s.interventions
attributes(s::ModelSpec) = s.attributes
observation(s::ModelSpec) = s.observation
_progression(s::ModelSpec) = s.progression
population_size(s::ModelSpec) = population_size(s.process)

# Structural accessors delegate to the wrapped process, so the analytical
# helpers and the Turing `~` distribution wrappers treat a spec like the
# process it wraps.
single_type_offspring(s::ModelSpec) = single_type_offspring(s.process)
n_types(s::ModelSpec) = n_types(s.process)
_single_kernel(s::ModelSpec) = _single_kernel(s.process)

# `simulate` unwraps the spec: the process is the model, the spec's layers are
# the forcing inputs.
function simulate(spec::ModelSpec;
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
    return _simulate(spec.process, sim_opts; interventions = spec.interventions,
        attributes = spec.attributes, progression = spec.progression,
        observation = spec.observation, rng, condition, max_attempts)
end

function simulate(spec::ModelSpec, n::Int;
        n_initial::Int = 1,
        max_cases::Union{Int, Nothing} = 10_000,
        max_generations::Union{Int, Nothing} = 100,
        max_time::Union{Real, Nothing} = nothing,
        stopping_rules::Union{Vector{<:AbstractStoppingRule}, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng(),
        parallel::Bool = false)
    sim_opts = SimOpts(; n_initial, max_cases, max_generations, max_time,
        stopping_rules)
    return _simulate_n(spec.process, n, sim_opts;
        interventions = spec.interventions, attributes = spec.attributes,
        progression = spec.progression, observation = spec.observation, rng,
        parallel)
end

function Base.show(io::IO, s::ModelSpec)
    print(io, "ModelSpec(", s.process, "; ", length(s.interventions),
        " interventions, ", length(s.progression), " transitions)")
end
