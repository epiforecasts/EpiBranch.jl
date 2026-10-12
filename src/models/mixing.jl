# ── MixingProcess ────────────────────────────────────────────────────
#
# A closed population of fixed size N, split into mixing groups that contact
# each other at different rates: age bands, sex, income strata, spatial
# patches. It reuses the same mass-action route as `HomogeneousProcess` — the
# homogeneous case is the one-group special case, `mixing_by = ()` — so the
# only thing a user supplies beyond the population size is the pair contact
# rate between groups. It describes the transmission alone: the natural
# history (progression), interventions, attributes and observation are
# composed onto it with a `ModelSpec`, and the infectious window is resolved
# from that progression when the model is simulated.

"""
    MixingProcess(; population_size, mixing_by = (), rate,
                 from = nothing, until = (:recovered, :died, :isolated))

A closed population of `population_size` individuals, split into mixing
groups by `mixing_by`, a tuple of attribute keys already set on each
individual (`:age_band`, `:ses`, `:patch`; set by the composed `attributes`,
not by this process). A member's group is the tuple of those attribute
values; `mixing_by = ()` puts everyone in the single group `()`, recovering
`HomogeneousProcess`.

`rate(infector_group, target_group)` is the rate at which one infective in
the first group contacts one member of the second. Homogeneous mixing is
`β/N` for every pair; a contact matrix `M[b, a]` giving the rate at which one
infective in group `a` contacts the `n[b]` members of group `b` between them
gives `M[b, a]/n[b]`. It is simulated as mass action on the same
continuous-time race as `HomogeneousProcess`.

The process describes the transmission alone. The natural history is a
`progression` of [`Transition`](@ref)s attached with a [`ModelSpec`](@ref),
exactly as for [`HomogeneousProcess`](@ref): `from` is the state the
infectious window opens at, derived from the progression when left `nothing`;
`until` names the removal states that close it.

The population is simulated until extinction or `max_time`, whichever comes
first; the other `simulate` termination controls do not apply, and `simulate`
warns if you set one.

Every contact is a contact between two named people, so competing risks,
per-individual susceptibility and infectiousness, and risks that read the
infector all apply as they do on any other route of the race.

# Example

```julia
using EpiBranch, Distributions

# Two age bands of equal size, tagged on creation; band 1 mixes more.
band = (rng, ind) -> (ind.state[:age_band] = rand(rng, 1:2))
M = [3.0 0.5; 0.5 0.5]
rate = (from, to) -> M[to[1], from[1]] / 1500

model = ModelSpec(
    MixingProcess(; population_size = 3000, mixing_by = (:age_band,), rate);
    progression = [Transition(:recovered; from = :infection, rate = 1.0, terminal = true)],
    attributes = band)
state = simulate(model; n_initial = 5)
```
"""
struct MixingProcess{F} <: TransmissionModel
    population_size::Int
    mixing_by::Tuple
    rate::F
    from::Union{Symbol, Nothing}
    until::Tuple
end

function MixingProcess(;
        population_size::Integer,
        mixing_by::Tuple = (),
        rate,
        from = nothing,
        until = (:recovered, :died, :isolated)
    )
    population_size >= 1 || throw(ArgumentError("population_size must be ≥ 1"))
    return MixingProcess(Int(population_size), mixing_by, rate, from, Tuple(until))
end

population_size(m::MixingProcess) = m.population_size

# The race runs over the fixed population until extinction or `max_time`; the
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

Simulate the structured population as mass action between its mixing groups,
with the modelling layers supplied by the caller (a bare process, or a
`ModelSpec`). The infectious window's `from` state is resolved here from the
composed `progression`; the mixing groups are read off each individual's
state, so the `attributes` composing the model must set `model.mixing_by`'s
keys before the race runs.
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
    N = model.population_size

    state = new_state(model, progression, attributes, rng)
    add_individuals!(
        state, N, interventions;
        setup = (ind, i) -> nothing
    )

    members = collect(1:N)
    sellke_race!(
        state, members, rng, sim_opts;
        seed! = (best, _, r) -> (best[randperm(r, N)[1:n_initial]] .= 0),
        targets = _MassAction(state, members; mixing_by = model.mixing_by, rate = model.rate),
        from = from, until = model.until, interventions,
        risks = transmission_risks(model), recorder
    )

    apply_observation!(observation, state, rng)
    return state
end
