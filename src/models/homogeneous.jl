# ── HomogeneousProcess ───────────────────────────────────────────────
#
# A homogeneously-mixing closed population of fixed size N: every infectious
# individual exerts the same force of infection on every susceptible, so the
# outbreak depletes a finite population with no structure beyond its size. It
# runs on the continuous-time race as mass action (`_MassAction`), every pair
# meeting at β/N, which reproduces the exact stochastic SIR final-size law and
# yields infection times, not just the final size. It describes the transmission
# alone: the natural history (progression), interventions, attributes and
# observation are composed onto it with a `ModelSpec`, and the infectious window
# is resolved from that progression when the model is simulated.

"""
    HomogeneousProcess(; transmission_rate, population_size,
                       from = nothing, until = (:recovered, :died, :isolated))

A closed population of `population_size` individuals that mix homogeneously:
every infectious person exerts the same force of infection on every susceptible.
It is simulated in continuous time as mass action, every pair meeting at the
same rate, which reproduces the exact stochastic SIR final-size law.
Transmission is the per-infective rate `transmission_rate` (β, so β/N to each
susceptible).

`transmission_rate` is a fixed β by default, but it may also be a calendar
schedule — a [`Steps`](@ref), a [`Seasonal`](@ref), or a callable
`t -> β(t)` — giving β as a function of calendar day, for a lockdown or a
seasonal rate. With a schedule, contacts are still drawn exactly, from the
schedule's integrated hazard, but every member of the population is sampled
directly rather than at the bound a fixed β allows, which costs O(N) per case.

The natural history comes from the `progression` of a [`ModelSpec`](@ref),
as for a [`BranchingProcess`](@ref). With only a recovery step,
`Transition(:recovered; rate = γ, terminal = true)`, this is an SIR model;
adding a latent period (a step to `:infectious`) makes it SEIR. Further steps
(onset, hospitalisation, death) appear in the line list.

- `from`: the state at which a case becomes infectious. Left as `nothing`, it
  is `:infectious` when the progression has a latent period, otherwise
  `:infection`.
- `until`: the states that end infectiousness, by default
  `(:recovered, :died, :isolated)`.

Interventions attach through two seams. An intervention that removes a case from
transmission, such as `Isolation`, shortens its infectious window and curtails
spread; one whose effect is a per-contact block, such as a leaky isolation or a
vaccine's efficacy, is resolved against each contact a pair makes, and a
blocked contact leaves the pair to meet again, so blocking a fraction of the
contacts thins the force of infection by the same fraction. Per-individual
susceptibility and infectiousness reach that thinning directly: they scale the
rate at which each pair meets. Four interventions are reported with a warning
rather than applied, because mass action settles one case at a time and never
produces the batches of new contacts their dosing and tracing act on:
`MassVaccination`'s rollout, `GroupVaccination`, `RingVaccination` and
`ContactTracing`, which has no contacts to trace under random mixing anyway.
Doses a person already holds when the outbreak starts still protect them;
only new dosing during the run is dropped. Control expressed as a removal
`Transition` in the progression always applies.

The population is simulated until extinction or `max_time`, whichever comes
first; with `max_time`, individuals whose infection would fall later stay
uninfected. The other `simulate` termination controls (`max_cases`,
`max_generations`, `stopping_rules` other than `MaxTime`) do not apply, and
`simulate` warns if you set one. This holds for the structure-driven models
generally.

A run costs about one draw per contact made inside an infectious window. A
window that never closes, as in an SI model with no removal transition, makes
each case contact every member, so such a run costs O(N) per case unless
`max_time` bounds it.

# Example

```julia
using EpiBranch, Distributions
model = ModelSpec(
    HomogeneousProcess(; transmission_rate = 2.0, population_size = 3000);
    progression = [Transition(:recovered; from = :infection, rate = 1.0, terminal = true)])
state = simulate(model; n_initial = 5)
```
"""
struct HomogeneousProcess{T <: Real, R} <: TransmissionModel
    population_size::Int
    transmission_rate::R            # β, fixed or a calendar schedule
    from::Union{Symbol, Nothing}   # infectious-window start; nothing → derive
    until::Tuple                   # removal states that close the infectious window
end

function HomogeneousProcess(;
        transmission_rate,
        population_size::Integer,
        from = nothing,
        until = (:recovered, :died, :isolated)
    )
    population_size >= 1 || throw(ArgumentError("population_size must be ≥ 1"))
    rate = _prepared_transmission_rate(transmission_rate)
    T = _rate_time_type(rate)
    return HomogeneousProcess{T, typeof(rate)}(
        Int(population_size), rate, from, Tuple(until)
    )
end

# A fixed β is validated and floated as before; a calendar schedule validates
# itself when it is built (`Steps`'s own values, say), so nothing further is
# checked here.
function _prepared_transmission_rate(β::Real)
    (isfinite(β) && β >= 0) || throw(
        ArgumentError(
            "transmission_rate must be a finite, non-negative number (β ≥ 0), " *
                "or a calendar schedule (a `Steps`, a `Seasonal`, or a function of time)"
        )
    )
    # Keep β at whatever real type it comes in as — a dual under automatic
    # differentiation — so a gradient with respect to β flows into the infection
    # times.
    return float(β)
end
_prepared_transmission_rate(schedule) = schedule

# The timing type a rate carries: a fixed β's own type, or a schedule's, where
# one is declared (`Steps`, `Seasonal`); a plain callable declares none, so its
# contacts are timed in `Float64`.
_rate_time_type(β::Real) = typeof(β)
_rate_time_type(s::Steps{T}) where {T} = T
_rate_time_type(s::Seasonal{T}) where {T} = T
_rate_time_type(schedule) = Float64

# The mass-action route's own rate and calendar for β: a fixed β is the route's
# rate, with no calendar; a schedule stands for β itself, so the route samples
# at a constant 1/N and reads β(t) off the schedule.
_homogeneous_rate(β::Real, N) = (_, _) -> β / N
_homogeneous_rate(schedule, N) = (_, _) -> 1 / N
_homogeneous_calendar(::Real) = nothing
_homogeneous_calendar(schedule) = schedule

population_size(m::HomogeneousProcess) = m.population_size

# The race honours a `stopping_rules` entry, or a non-default `max_cases` or
# `max_generations`, but leaves those two keywords' defaults off, so an
# existing run over the fixed population keeps going to extinction or
# `max_time` unless a cap is explicitly asked for (`_drop_default_shortcuts`).
_honours_termination_controls(::HomogeneousProcess) = false

# See `_warn_uncovered_terminal_states` in branching_process.jl.
function _validate_process_windows(m::HomogeneousProcess, progression)
    return _warn_uncovered_terminal_states(m.until, progression; from = m.from)
end

# The state's timing type follows β's type, so a dual β makes `Individual{Dual}`
# members and gradients flow through the contact times.
_time_type(::HomogeneousProcess{T}) where {T} = T

function Base.show(io::IO, m::HomogeneousProcess)
    β = m.transmission_rate isa AbstractFloat ?
        round(m.transmission_rate; digits = 4) : m.transmission_rate
    from = m.from === nothing ? "" : ", from=:$(m.from)"
    return print(io, "HomogeneousProcess(population_size=$(m.population_size), β=$β", from, ")")
end

"""
    simulate_once(model::HomogeneousProcess, sim_opts; interventions, attributes,
                  progression, observation, recorder, rng)

Simulate the homogeneous population on the continuous-time race, as mass action
among its members (`_MassAction`), with the modelling layers supplied by the
caller (a bare process, or a `ModelSpec`). The infectious window's `from` state
is resolved here from the composed `progression`.
"""
function simulate_once(
        model::HomogeneousProcess, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng
    )
    n_initial = sim_opts.n_initial
    n_initial >= 1 || throw(ArgumentError("n_initial must be ≥ 1"))
    n_initial <= model.population_size ||
        throw(ArgumentError("n_initial cannot exceed population_size"))

    from = something(model.from, infectious_from(progression))
    β = model.transmission_rate

    state = new_state(model, progression, attributes, rng)
    add_individuals!(
        state, model.population_size, interventions;
        setup = (ind, i) -> nothing
    )

    # Homogeneous mixing is mass action with a single group: every pair meets at
    # β/N, scaled by the two members' infectiousness and susceptibility. A
    # schedule stands for β itself, so the route's own rate is just 1/N and the
    # schedule supplies the calendar day's value.
    N = model.population_size
    members = collect(1:N)
    sellke_race!(
        state, members, rng, sim_opts;
        seed! = (best, _, r) -> (best[randperm(r, N)[1:n_initial]] .= 0),
        targets = _MassAction(
            state, members; rate = _homogeneous_rate(β, N),
            calendar = _homogeneous_calendar(β), from
        ),
        from = from, until = model.until, interventions,
        risks = transmission_risks(model), recorder
    )

    apply_observation!(observation, state, rng)
    return state
end

# ── Deriving the infectious window from a progression ────────────────
# Shared by the structure-driven models: the infectious window's `from` state is
# read from the composed progression when the model is simulated.

"""
    infectious_from(progression) -> Symbol

The state a case's infectious window opens at, derived from `progression`:
`:infectious` when the progression produces it (a latent period), otherwise
`:infection`. A model whose `from` field can override the derivation reads
that first and falls back to this only when it is `nothing`, as
`something(model.from, infectious_from(progression))`.
"""
function infectious_from(progression)
    return any(
            t -> hasproperty(t, :state) && getfield(t, :state) === :infectious,
            progression
        ) ? :infectious : :infection
end
