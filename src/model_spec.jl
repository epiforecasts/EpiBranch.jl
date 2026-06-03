# ── ModelSpec ────────────────────────────────────────────────────────
#
# Bundles every knob that defines what a simulation/likelihood run
# means — the latent process, per-individual policies, clinical
# transitions, attribute builders, observation model, and run-control
# options — into one struct. The existing kwarg-form functions stay as
# thin shims that build a `ModelSpec` internally, so passing a spec is
# always equivalent to passing the unpacked pieces.

"""
    ModelSpec(; process, interventions=[], transitions=[],
               attributes=NoAttributes(), observation=NoObservation(),
               sim_opts=SimOpts())

Single-object configuration for a simulation or likelihood call. Pass
this to [`simulate`](@ref) or to a likelihood wrapper instead of
threading the same knobs through every call as kwargs.

# Fields

- `process::TransmissionModel` — the latent dynamics.
- `interventions::Vector{<:AbstractIntervention}` — per-individual
  policy overlay (isolation, contact tracing, vaccination).
- `transitions::Vector{<:AbstractClinicalTransition}` — clinical-state
  Markov chain (reporting, hospitalisation, recovery, death).
- `attributes::Union{Function, NoAttributes}` — per-individual builder
  closure (composed via [`compose`](@ref)) for setting things like
  incubation period, age, susceptibility.
- `observation::Union{ObservationModel, NoObservation}` — how the
  latent state generates observed data (e.g.
  [`PerCaseObservation`](@ref)).
- `sim_opts::SimOpts` — run-control options (max cases, max time,
  stopping rules, initial seeds).

The wrapper-style `Observed(process, observation)` is equivalent to
`ModelSpec(; process, observation)` — same dispatch surface, just a
single field on the spec instead of a nested type.
"""
struct ModelSpec{P <: TransmissionModel, I, T, A, O, S}
    process::P
    interventions::I
    transitions::T
    attributes::A
    observation::O
    sim_opts::S
end

"""Sentinel indicating no observation model is configured."""
struct NoObservation end

function ModelSpec(;
        process::TransmissionModel,
        interventions::Vector{<:AbstractIntervention} = AbstractIntervention[],
        transitions::Vector{<:AbstractClinicalTransition} = AbstractClinicalTransition[],
        attributes::Union{Function, NoAttributes} = NoAttributes(),
        observation::Union{ObservationModel, NoObservation} = NoObservation(),
        sim_opts::SimOpts = SimOpts())
    return ModelSpec(process, interventions, transitions,
        attributes, observation, sim_opts)
end

function Base.show(io::IO, spec::ModelSpec)
    print(io, "ModelSpec(process=$(spec.process)")
    isempty(spec.interventions) ||
        print(io, ", interventions=$(length(spec.interventions))")
    isempty(spec.transitions) ||
        print(io, ", transitions=$(length(spec.transitions))")
    spec.attributes isa NoAttributes || print(io, ", attributes=…")
    spec.observation isa NoObservation || print(io, ", observation=$(spec.observation)")
    print(io, ")")
end

# Resolve the model that the engine should actually simulate from a
# spec. With an observation model, wrap the process in `Observed` so
# the simulation path picks up post-simulation projection
# (e.g. `:reported`, `:report_time` for `PerCaseObservation`).
_simulation_model(spec::ModelSpec) = _simulation_model(spec.process, spec.observation)
_simulation_model(process, ::NoObservation) = process
_simulation_model(process, observation::ObservationModel) = Observed(process, observation)
