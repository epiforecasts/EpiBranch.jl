# ── HomogeneousProcess ───────────────────────────────────────────────
#
# A homogeneously-mixing closed population of fixed size N: every infectious
# individual exerts the same force of infection on every susceptible, so the
# outbreak depletes a finite population with no structure beyond its size. It
# runs on the continuous-time race as mass action (`_MassAction`), every pair
# meeting at β/N, which reproduces the exact stochastic SIR final-size law and
# yields infection times, not just the final size. It describes the transmission alone: the natural
# history (progression), interventions, attributes and observation are composed onto it
# with a `ModelSpec`, and the infectious window is resolved from that progression
# when the model is simulated.

"""
    HomogeneousProcess(; transmission_rate, population_size,
                       from = nothing, until = (:recovered, :died, :isolated))

A closed population of `population_size` individuals that mix homogeneously:
every infectious person exerts the same force of infection on every susceptible.
It is simulated in continuous time as mass action, every pair meeting at the
same rate, which reproduces the exact stochastic SIR final-size law. Transmission is the per-infective rate
`transmission_rate` (β, so β/N to each susceptible).

The process describes the transmission alone. The natural history is a
`progression` of [`Transition`](@ref)s attached with a [`ModelSpec`](@ref),
exactly as for [`BranchingProcess`](@ref): a latent period is a transition to
`:infectious`, an infectious period a terminal removal transition (to
`:recovered`, say). `from` is the state the infectious window opens at; left as
`nothing` it is derived from the progression (`:infectious` when a latent period
produces it, otherwise `:infection`). `until` names the removal states that
close the window.

How detailed the course of infection is is up to you: with just a recovery
transition this is an SIR model; adding a latent transition makes it SEIR, and
onset, hospitalisation and death come from further transitions in the
progression and appear in the line list. The model holds to two assumptions: a
closed population, and one infection per person.

Interventions attach through two seams. An intervention that removes a case from
transmission, such as `Isolation`, shortens its infectious window and curtails
spread; one whose effect is a per-contact block, such as a leaky isolation or a
vaccine's efficacy, is resolved against each contact a pair makes, and a
blocked contact leaves the pair to meet again, so blocking a fraction of the
contacts thins the force of infection by the same fraction. Per-individual
susceptibility and infectiousness reach that thinning directly: they scale the
rate at which each pair meets. An intervention that finds its targets only among
newly created contacts — `MassVaccination`'s rollout, `GroupVaccination`, or
`ContactTracing`, which has no contacts to act along under mass action — is
reported with a warning rather than applied. Control expressed as a removal
`Transition` in the progression always applies.

The population is simulated over its fixed population until extinction or `max_time`,
whichever comes first; with `max_time`, individuals whose infection would fall
later stay uninfected. The other `simulate` termination controls
(`max_cases`, `max_generations`, `stopping_rules` other than `MaxTime`) do not
apply, and `simulate` warns if you set one. This holds for the structure-driven
models generally.

# Example

```julia
using EpiBranch, Distributions
model = ModelSpec(
    HomogeneousProcess(; transmission_rate = 2.0, population_size = 3000);
    progression = [Transition(:recovered; from = :infection, rate = 1.0, terminal = true)])
state = simulate(model; n_initial = 5)
```
"""
struct HomogeneousProcess{T <: Real} <: TransmissionModel
    population_size::Int
    transmission_rate::T           # the per-infective rate β
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
    (isfinite(transmission_rate) && transmission_rate >= 0) || throw(
        ArgumentError(
            "transmission_rate must be a finite, non-negative number (β ≥ 0)"
        )
    )
    # Keep β at whatever real type it comes in as — a dual under automatic
    # differentiation — so a gradient with respect to β flows into the infection
    # times.
    return HomogeneousProcess(
        Int(population_size), float(transmission_rate), from, Tuple(until)
    )
end

population_size(m::HomogeneousProcess) = m.population_size

# The race runs over the fixed population until extinction or `max_time`; the
# other termination controls do not apply, and `simulate` warns if any is set.
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
    _simulate(model::HomogeneousProcess, sim_opts; interventions, attributes,
              progression, observation, recorder, rng, condition, max_attempts)

Simulate the homogeneous population on the continuous-time race, as mass action
among its members (`_MassAction`), with the modelling layers supplied by the
caller (a bare process, or a `ModelSpec`). The infectious window's `from` state
is resolved here from the composed `progression`.
"""
function _simulate(
        model::HomogeneousProcess, sim_opts::SimOpts;
        interventions, attributes, progression, observation, recorder, rng,
        condition, max_attempts
    )
    condition !== nothing && return _retry_for_condition(
        () -> _simulate(
            model, sim_opts; interventions, attributes, progression,
            observation, recorder, rng, condition = nothing, max_attempts
        ),
        condition, max_attempts
    )

    n_initial = sim_opts.n_initial
    n_initial >= 1 || throw(ArgumentError("n_initial must be ≥ 1"))
    n_initial <= model.population_size ||
        throw(ArgumentError("n_initial cannot exceed population_size"))

    from = _resolve_infectious_from(model.from, progression)
    β = model.transmission_rate

    state = new_state(model, progression, attributes, rng)
    add_individuals!(
        state, model.population_size, interventions;
        setup = (ind, i) -> nothing
    )

    # Homogeneous mixing is mass action with a single group: every pair meets at
    # β/N, scaled by the two members' infectiousness and susceptibility.
    N = model.population_size
    members = collect(1:N)
    extinct = _sellke_race!(
        state, members, rng;
        seed! = (best, _, r) -> (best[randperm(r, N)[1:n_initial]] .= 0),
        targets = _MassAction(state, members; rate = (_, _) -> β / N),
        from = from, until = model.until, interventions,
        risks = transmission_risks(model), max_time = _max_time(sim_opts), recorder
    )

    _reconcile_sellke_bookkeeping!(state, extinct)
    apply_observation!(observation, state, rng)
    return state
end

# ── Deriving the infectious window from a progression ────────────────
# Shared by the structure-driven models: the infectious window's `from` state is
# read from the composed progression when the model is simulated.

# The state the infectious window opens at: :infectious when the progression
# produces it (a latent period), otherwise :infection. An explicit `from`
# overrides the derivation.
_resolve_infectious_from(from::Symbol, progression) = from
_resolve_infectious_from(::Nothing, progression) = _infectious_from(progression)
function _infectious_from(progression)
    return any(
            t -> hasproperty(t, :state) && getfield(t, :state) === :infectious,
            progression
        ) ? :infectious : :infection
end
