# Julia 1.11+ public declarations — extensible API that is not exported.
# These names are part of the public surface for writing extensions, but
# are not brought into scope by `using EpiBranch` (call them qualified,
# e.g. `EpiBranch.initialise_state`).

# Intervention interface: a package subtyping `AbstractIntervention`
# overrides these.
public initialise_individual!
public resolve_individual!
public apply_post_transmission!
public competing_risk
public reset!
public required_fields

# Transmission-model interface. A new process subtypes `TransmissionModel`
# and extends these seam methods:
#   - candidate generation: `generate_offspring` (offspring-driven) or
#     `contacts_of` + `collect_exposures` (structure-driven), all exported;
#   - `initialise_state` to build its starting population;
#   - `interventions`, `attributes`, `observation` to return the model
#     inputs it carries;
#   - `population_size`, `n_types`, `model_generation_time` metadata.
public initialise_state
public interventions
public attributes
public observation
public population_size
public n_types
public model_generation_time
#   - `transmission_risks` to contribute per-pair competing risks (e.g. a
#     network's per-edge probability), resolved alongside the built-ins.
public transmission_risks

# Helpers an `initialise_state` / `contacts_of` builds on, so a model never
# touches the engine's bookkeeping directly.
public new_state
public add_individuals!
public seed!
public get_generation_time
public transmission_time

# Resolve a case's natural history (the model's `progression`). The engine calls
# this for every new case; a model running its own simulation loop calls it.
public resolve_transitions!

# Apply the model's observation to a finished state (the simulation side of the
# observation protocol; `observe` is the exported analytical side). The engine
# calls this after a run; a model running its own simulation loop calls it.
public apply_observation!

# Types a model constructs or dispatches on.
public NoGenerationTime
