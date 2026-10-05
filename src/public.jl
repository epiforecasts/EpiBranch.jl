# Julia 1.11+ public declarations — extensible API that is not exported.
# These names are part of the public surface for writing extensions, but
# are not brought into scope by `using EpiBranch` (call them qualified,
# e.g. `EpiBranch.initialise_state`).

# Intervention interface: a package subtyping `AbstractIntervention`
# overrides these.
public initialise_individual!
public resolve_individual!
public apply_post_transmission!
public on_infection_settled!
public trace_contacts!
public traces_contacts
public supplies_contacts
public competing_risk
public persistent_competing_risks
public reset!
public required_fields
public infectious_removal_time
public risk_applies
public risk_depends_on_infector
public reads_population_state
public standing_block
# End an infection early from any intervention hook, and read when it ended.
public abort_infection!
public infection_aborted_time

# Isolation eligibility: whether an isolation time counts as a detection.
public records_isolation

# Pair kernels: the host records a kernel's hazards depend on.
public watched_records

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
#   - a model with more than one natural race partition (a household process,
#     over its households) defines `race_groups` to say how it splits into
#     independent `_sellke_race!` calls for a given kernel.
public race_groups

# Helpers an `initialise_state` / `contacts_of` builds on, so a model never
# touches the engine's bookkeeping directly.
public new_state
public add_individuals!
public seed!
public get_generation_time
public transmission_time

# Resolve a case's natural history (the model's `progression`). The engine calls
# this for every new case; a model running its own simulation loop calls it.
public transition_time
public resolve_transitions!
public transition_loglik, transition_term

# Apply the model's observation to a finished state (the simulation side of the
# observation protocol; `observe` is the exported analytical side). The engine
# calls this after a run; a model running its own simulation loop calls it.
public apply_observation!

# Types a model constructs or dispatches on.
public NoGenerationTime

# An `InfectionLayer` subtype names who could have infected whom, which lets the
# pairwise likelihood enumerate its (susceptible, possible infector) pairs.
public contact_structure
public followup_end
# A subtype holding per-host times under a name of its own overrides this
# rather than have the likelihood read a fixed field name off it.
public host_times

public pair_kernel
# A calendar schedule for a `PairKernel` implements `calendar_multiplier`, and
# either `next_calendar_break` (piecewise constant, the default shape) or a
# `calendar_shape` method returning `SmoothCalendar()`.
public calendar_multiplier, next_calendar_break, calendar_shape,
    PiecewiseConstantCalendar, SmoothCalendar
public infection_likelihood_compatible
public susceptibility_components, susceptibility_host_times, HazardScaling
public InterventionAction, intervention_actions, action_draw!, apply_actions!,
    continuous_actions, may_revise, is_settled

# Pairwise survival likelihood: how its two accumulation passes group rows
# into a result. `pairwise_surv_loglik` and `pairwise_surv_loglik_by_component`
# are the two built-in groupings; a new one is a `PairwiseReduction` subtype
# with `ngroups` and `group` methods, run with `pairwise_reduce`, no further
# change to the package.
public PairwiseReduction
public ngroups
public group
public pairwise_reduce

# Vaccine effect modes: a new mode defines `realised_efficacy`, and overrides
# `realise_prior_dose!` only to change what a dose recorded before the run gets.
public realised_efficacy
public realise_prior_dose!

# Stopping rules: a rule bounds a structure-driven run through `time_bound`,
# and says through `honoured_without_should_stop` whether such a run, which
# never consults `should_stop`, applies it in full.
public time_bound
public honoured_without_should_stop

# Vaccine effect modes: whether a mode's protection can be given a `waning`
# curve, which also tells a continuous-time race whether the block it composes
# is certain for good.
public supports_waning
