# ── HomogeneousProcess ───────────────────────────────────────────────
#
# A homogeneously-mixing closed population of fixed size N: every infectious
# individual exerts the same force of infection on every susceptible, so the
# outbreak is a finite, depleting pool with no structure beyond its size. It is
# simulated by the Sellke threshold construction (`_sellke_pool!`), which
# reproduces the exact stochastic SIR final-size law and yields infection times,
# not just the final size. It describes the transmission alone: the natural
# history (progression), interventions, attributes and observation are composed onto it
# with a `ModelSpec`, and the infectious window is resolved from that progression
# when the model is simulated.

"""
    HomogeneousProcess(; transmission_rate, population_size,
                       from = nothing, until = (:recovered, :died, :isolated))

A stochastic SIR or SEIR epidemic in a closed population of
`population_size` people who mix homogeneously. Each infectious person
infects each susceptible at rate `transmission_rate / population_size`, so
`transmission_rate` is β, the rate at which one infectious person makes
infectious contacts (per day). Each person can be infected at most once. The
simulation is exact (it uses Sellke's construction) and gives infection
times as well as the final size.

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

Interventions:

- [`Isolation`](@ref) and other measures that remove a case end its
  infectious period.
- Measures that prevent each infection with some probability (leaky
  isolation, vaccine efficacy) reduce the force of infection by that
  proportion. Per-person susceptibility and infectiousness (see
  [`transmission_traits`](@ref)) scale it in the same way.
- [`MassVaccination`](@ref), [`GroupVaccination`](@ref) and
  [`ContactTracing`](@ref) are not applied, and `simulate` warns: there are
  no individual contacts to trace or vaccinate in a homogeneously mixing
  population.
- Control written as a removal step in the progression always applies.

!!! note
    The epidemic runs until no one is infectious or until `max_time` (days);
    with `max_time`, anyone whose infection would come later stays
    uninfected. `max_cases`, `max_generations` and stopping rules other than
    [`MaxTime`](@ref) do not apply, and `simulate` warns if one is set.

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
    # differentiation — so a gradient with respect to β flows into the pool.
    return HomogeneousProcess(
        Int(population_size), float(transmission_rate), from, Tuple(until)
    )
end

population_size(m::HomogeneousProcess) = m.population_size

# The pool runs over its fixed population until extinction or `max_time`; the
# other termination controls do not apply, and `simulate` warns if any is set.
_honours_termination_controls(::HomogeneousProcess) = false

# See `_warn_uncovered_terminal_states` in branching_process.jl.
function _validate_process_windows(m::HomogeneousProcess, progression)
    return _warn_uncovered_terminal_states(m.until, progression; from = m.from)
end

# The state's timing type follows β's type, so a dual β makes an
# `Individual{Dual}` pool and gradients flow through the crossing times.
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

Simulate the homogeneous pool by the Sellke threshold construction, with the
modelling layers supplied by the caller (a bare process, or a `ModelSpec`). The
infectious window's `from` state is resolved here from the composed
`progression`. The pool draws a fresh infector for every contact rather than
meeting the same one again (see `_proposal_blocked`'s call site in
`sellke_pool.jl`), so it has no standing pair for `recorder` to be asked
about; it is accepted for a uniform call signature and otherwise unused.
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

    # The homogeneous pool is the one-type case of the structured Sellke pool:
    # no attributes name the mixing, so every individual feels the same force
    # β/N per unit of infectiousness (`sum(values(counts))` = the
    # infectiousness-weighted number currently infectious).
    extinct = _sellke_pool!(
        state, collect(1:model.population_size), rng;
        force = (type, counts) -> β / model.population_size * sum(values(counts)),
        n_initial = n_initial, from = from, until = model.until, interventions,
        risks = transmission_risks(model), max_time = _max_time(sim_opts)
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
