# ── MixingProcess ────────────────────────────────────────────────────
#
# A closed population of fixed size N, split into mixing groups that contact
# each other at different rates: age bands, sex, income strata, spatial
# patches. It reuses the same mass-action route as `HomogeneousProcess` (the
# homogeneous case is the one-group special case, `mixing_by = ()`), so the
# only thing a user supplies beyond the population size is the pair contact
# rate between groups. It describes the transmission alone: the natural
# history (progression), interventions, attributes and observation are
# composed onto it with a `ModelSpec`, and the infectious window is resolved
# from that progression when the model is simulated.

"""
    MixingProcess(; population, mixing_by = (), rate, calendar = nothing,
                 from = nothing, until = (:recovered, :died, :isolated))

A closed population, split into mixing groups by `mixing_by`, a tuple of
attribute keys on each individual (`:age_band`, `:ses`, `:patch`). A member's
group is the tuple of those attribute values; `mixing_by = ()` puts everyone in
the single group `()`, recovering `HomogeneousProcess`.

The population is a [`Population`](@ref), or `population_size = n` for
`Population(n)`. With sizes by group, `Population(sizes; groups)`, the process
writes the `mixing_by` attributes itself, so that the groups have exactly those
sizes (`groups` gives each group's attribute values as a tuple, `(i,)` for the
`i`-th group by default); the composed `attributes` must not write them. With
`Population(n)` the groups come from the composed `attributes`, which set
`mixing_by`'s keys, and their sizes may differ from run to run.

`rate(infector_group, target_group)` is the rate at which one infective in
the first group contacts one member of the second. Homogeneous mixing is
`β/N` for every pair; a contact matrix `M[b, a]` giving the rate at which one
infective in group `a` contacts the `n[b]` members of group `b` between them
gives `M[b, a]/n[b]`. It is simulated as mass action on the same
continuous-time race as `HomogeneousProcess`.

`calendar`, a [`Steps`](@ref) or [`Seasonal`](@ref) schedule or a callable
`t -> multiplier`, scales every group's rate by the schedule's value on the
calendar day, as the `calendar` of a [`PairKernel`](@ref) does, for a
lockdown or a seasonal pattern applied on top of a fixed contact matrix. With
a calendar, contacts are drawn exactly from the schedule's integrated
hazard. `Steps` and `Seasonal` have a known largest value, so the route keeps
sampling at a bound, at a cost per contact. A callable has none; every
member of each group is sampled directly instead, which costs O(N) per case.

The process describes the transmission alone. The natural history is a
`progression` of [`Transition`](@ref)s attached with a [`ModelSpec`](@ref),
exactly as for [`HomogeneousProcess`](@ref): `from` is the state the
infectious window opens at, derived from the progression when left `nothing`;
`until` names the removal states that close it.

The population is simulated until extinction or `max_time`, whichever comes
first. The `max_cases` and `max_generations` keyword shortcuts default to off
here. A run continues to extinction or `max_time` unless an explicit
value or a `stopping_rules` entry sets a cap.

Every contact is a contact between two named people, so competing risks,
per-individual susceptibility and infectiousness, and risks that read the
infector all apply as they do on any other route of the race.

# Example

```julia
using EpiBranch, Distributions

# Two age bands of 1500 people each, assigned by the process; band 1 mixes more.
M = [3.0 0.5; 0.5 0.5]
rate = (from, to) -> M[to[1], from[1]] / 1500

model = ModelSpec(
    MixingProcess(; population = Population([1500, 1500]), mixing_by = (:age_band,), rate);
    progression = [Transition(:recovered; from = :infection, rate = 1.0, terminal = true)])
state = simulate(model; n_initial = 5)
```
"""
struct MixingProcess{P <: Union{SinglePopulation, GroupedPopulation}, F, C} <: TransmissionModel
    population::P
    mixing_by::Tuple
    rate::F
    calendar::C
    from::Union{Symbol, Nothing}
    until::Tuple
end

function MixingProcess(;
        population = NoPopulation(),
        population_size = NoPopulation(),
        mixing_by::Tuple = (),
        rate,
        calendar = nothing,
        from = nothing,
        until = (:recovered, :died, :isolated)
    )
    pop = _closed_population(population, population_size)
    return MixingProcess(pop, mixing_by, rate, calendar, from, Tuple(until))
end

population(m::MixingProcess) = m.population
population_size(m::MixingProcess) = total_size(m.population)

# The race honours a `stopping_rules` entry, or a non-default `max_cases` or
# `max_generations`, but leaves those two keywords' defaults off, so an
# existing run over the fixed population keeps going to extinction or
# `max_time` unless a cap is explicitly asked for (`_drop_default_shortcuts`).
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
        "MixingProcess(population_size=$(population_size(m)), mixing_by=$groups", from, ")"
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
function simulate_once(model::MixingProcess, sim_opts::SimOpts; layers...)
    return _simulate_mass_action(
        model, sim_opts, model.mixing_by, model.rate, model.calendar; layers...
    )
end
