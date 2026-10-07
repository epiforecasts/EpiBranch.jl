# ── MixingProcess ────────────────────────────────────────────────────
#
# A closed population of fixed size N, split into mixing groups that contact
# each other at different rates: age bands, sex, income strata, spatial
# patches. It reuses the same Sellke pool as `HomogeneousProcess` — the
# homogeneous case is the one-group special case, `mixing_by = ()` — so the
# only thing a user supplies beyond the population size is how the force of
# infection depends on group membership. It describes the transmission alone:
# the natural history (progression), interventions, attributes and observation
# are composed onto it with a `ModelSpec`, and the infectious window is
# resolved from that progression when the model is simulated.

"""
    MixingProcess(; population_size, mixing_by = (), force,
                 from = nothing, until = (:recovered, :died, :isolated))

A closed population of `population_size` individuals, split into mixing
groups by `mixing_by`, a tuple of attribute keys already set on each
individual (`:age_band`, `:ses`, `:patch`; set by the composed `attributes`,
not by this process). A susceptible's group is the tuple of those attribute
values; `mixing_by = ()` puts everyone in the single group `()`, recovering
`HomogeneousProcess`.

`force(group, counts)` is the per-susceptible force of infection on a group,
given `counts`, a `Dict` mapping each group to the infectiousness-weighted
number currently infectious in it. It is simulated by the same Sellke
threshold construction as `HomogeneousProcess`; see
[`EpiBranch.sellke_pool!`](@ref) for the full contract `force` must meet.

The process describes the transmission alone. The natural history is a
`progression` of [`Transition`](@ref)s attached with a [`ModelSpec`](@ref),
exactly as for [`HomogeneousProcess`](@ref): `from` is the state the
infectious window opens at, derived from the progression when left `nothing`;
`until` names the removal states that close it.

The pool is simulated over its fixed population until extinction or
`max_time`, whichever comes first; the other `simulate` termination controls
do not apply, and `simulate` warns if you set one.

With more than one mixing group, an intervention whose `competing_risk`
depends on which individual is the infector (a leaky isolation, say) is
refused: see [`EpiBranch.sellke_pool!`](@ref EpiBranch._sellke_pool!) for why,
and [`EpiBranch.risk_depends_on_infector`](@ref) for how an intervention
whose risk reads only the contact opts back in.

# Example

```julia
using EpiBranch, Distributions

# Two age bands of equal size, tagged on creation; band 1 mixes more.
band = (rng, ind) -> (ind.state[:age_band] = rand(rng, 1:2))
M = [3.0 0.5; 0.5 0.5]
force = (group, counts) -> begin
    b = group[1]
    sum(M[b, h] * get(counts, (h,), 0) / 1500 for h in 1:2)
end

model = ModelSpec(
    MixingProcess(; population_size = 3000, mixing_by = (:age_band,), force);
    progression = [Transition(:recovered; from = :infection, rate = 1.0, terminal = true)],
    attributes = band)
state = simulate(model; n_initial = 5)
```
"""
struct MixingProcess{F} <: TransmissionModel
    population_size::Int
    mixing_by::Tuple
    force::F
    from::Union{Symbol, Nothing}
    until::Tuple
end

function MixingProcess(;
        population_size::Integer,
        mixing_by::Tuple = (),
        force,
        from = nothing,
        until = (:recovered, :died, :isolated)
    )
    population_size >= 1 || throw(ArgumentError("population_size must be ≥ 1"))
    return MixingProcess(Int(population_size), mixing_by, force, from, Tuple(until))
end

population_size(m::MixingProcess) = m.population_size

# The pool runs over its fixed population until extinction or `max_time`; the
# other termination controls do not apply, and `simulate` warns if any is set.
_honours_termination_controls(::MixingProcess) = false

# See `_warn_uncovered_terminal_states` in branching_process.jl.
function _validate_process_windows(m::MixingProcess, progression)
    return _warn_uncovered_terminal_states(m.until, progression; from = m.from)
end

function Base.show(io::IO, m::MixingProcess)
    groups = isempty(m.mixing_by) ? "()" : string(m.mixing_by)
    from = m.from === nothing ? "" : ", from=:$(m.from)"
    return print(
        io,
        "MixingProcess(population_size=$(m.population_size), mixing_by=$groups", from, ")"
    )
end

"""
    simulate_once(model::MixingProcess, sim_opts; interventions, attributes,
                  progression, observation, recorder, rng)

Simulate the structured pool by the Sellke threshold construction, with the
modelling layers supplied by the caller (a bare process, or a `ModelSpec`).
The infectious window's `from` state is resolved here from the composed
`progression`; the mixing groups are read off each individual's state, so the
`attributes` composing the model must set `model.mixing_by`'s keys before the
pool runs.
"""
function simulate_once(
        model::MixingProcess, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng
    )
    n_initial = sim_opts.n_initial
    n_initial >= 1 || throw(ArgumentError("n_initial must be ≥ 1"))
    n_initial <= model.population_size ||
        throw(ArgumentError("n_initial cannot exceed population_size"))

    from = something(model.from, infectious_from(progression))

    state = new_state(model, progression, attributes, rng)
    add_individuals!(
        state, model.population_size, interventions;
        setup = (ind, i) -> nothing
    )

    sellke_pool!(
        state, collect(1:model.population_size), rng, sim_opts;
        mixing_by = model.mixing_by, force = model.force,
        n_initial = n_initial, from = from, until = model.until, interventions,
        risks = transmission_risks(model)
    )

    apply_observation!(observation, state, rng)
    return state
end
