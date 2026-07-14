# ── HomogeneousProcess ───────────────────────────────────────────────
#
# A homogeneously-mixing closed population of fixed size N: every infectious
# individual exerts the same force of infection on every susceptible, so the
# outbreak is a finite, depleting pool with no structure beyond its size. It is
# simulated by the Sellke threshold construction (`_sellke_pool!`), which
# reproduces the exact stochastic SIR final-size law and yields infection times,
# not just the final size. It is a pure transmission kernel: the natural history
# (progression), interventions, attributes and observation are composed onto it
# with a `ModelSpec`, and the infectious window and (for `R0`) β are resolved
# from that progression when the model is simulated.

"""
    HomogeneousProcess(; transmission_rate = nothing, R0 = nothing, population_size,
                       from = nothing, until = (:recovered, :died, :isolated))

A closed population of `population_size` individuals that mix homogeneously:
every infectious person exerts the same force of infection on every susceptible.
It is simulated by the Sellke threshold construction, which reproduces the exact
stochastic SIR final-size law. Set transmission either as `transmission_rate`
(the per-infective rate β, so β/N to each susceptible) or as `R0`; give exactly
one. With `R0`, β is resolved at simulate time as `R0 / mean infectious period`,
the mean read from the progression composed onto the model.

The process is a pure transmission kernel. The natural history is a
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

The pool is always simulated to extinction over its fixed population, so the
`simulate` termination controls (`max_cases`, `max_generations`, `max_time`,
`stopping_rules`) do not apply; only `n_initial` (and `condition`) are used.
This holds for the structure-driven models generally.

# Example

```julia
using EpiBranch, Distributions
model = ModelSpec(
    HomogeneousProcess(; R0 = 2.0, population_size = 3000);
    progression = [Transition(:recovered; from = :infection, rate = 1.0, terminal = true)])
state = simulate(model; n_initial = 5)
```
"""
struct HomogeneousProcess{T <: Real} <: TransmissionModel
    population_size::Int
    rate::T                        # the transmission parameter: β, or an R0 to resolve
    is_r0::Bool                    # whether `rate` is an R0 (else it is β directly)
    from::Union{Symbol, Nothing}   # infectious-window start; nothing → derive
    until::Tuple                   # removal states that close the infectious window
end

# A single `rate` field of type `T` keeps the type parameter bound; a bare
# `Union{T, Nothing}` pair for β and R0 would leave `T` unbound when both are
# `nothing`. `is_r0` records which of β or R0 `rate` holds.

function HomogeneousProcess(; transmission_rate = nothing, R0 = nothing,
        population_size::Integer,
        from = nothing,
        until = (:recovered, :died, :isolated))
    population_size >= 1 || throw(ArgumentError("population_size must be ≥ 1"))
    (transmission_rate === nothing) == (R0 === nothing) && throw(ArgumentError(
        "provide exactly one of `transmission_rate` (β directly) or `R0`"))
    # Keep the transmission parameter at whatever real type it comes in as — a
    # dual under automatic differentiation — so a gradient with respect to β or
    # R0 flows into the pool.
    is_r0 = transmission_rate === nothing
    val = float(is_r0 ? R0 : transmission_rate)
    return HomogeneousProcess(Int(population_size), val, is_r0, from, Tuple(until))
end

population_size(m::HomogeneousProcess) = m.population_size

# The state's timing type follows the transmission parameter's type, so a dual β
# or R0 makes an `Individual{Dual}` pool and gradients flow through the crossing
# times.
_time_type(::HomogeneousProcess{T}) where {T} = T

function Base.show(io::IO, m::HomogeneousProcess)
    r = m.rate isa AbstractFloat ? round(m.rate; digits = 4) : m.rate
    label = m.is_r0 ? "R0=$r" : "β=$r"
    from = m.from === nothing ? "" : ", from=:$(m.from)"
    print(io, "HomogeneousProcess(population_size=$(m.population_size), ", label, from, ")")
end

"""
    _simulate(model::HomogeneousProcess, sim_opts; interventions, attributes,
              progression, observation, rng, condition, max_attempts)

Simulate the homogeneous pool by the Sellke threshold construction, with the
modelling layers supplied by the caller (a bare process, or a `ModelSpec`). The
infectious window's `from` state and, when the model was built with `R0`, β are
resolved here from the composed `progression`.
"""
function _simulate(model::HomogeneousProcess, sim_opts::SimOpts;
        interventions, attributes, progression, observation, rng, condition,
        max_attempts)
    condition !== nothing && return _retry_for_condition(
        () -> _simulate(model, sim_opts; interventions, attributes, progression,
            observation, rng, condition = nothing, max_attempts),
        condition, max_attempts)

    n_initial = sim_opts.n_initial
    n_initial >= 1 || throw(ArgumentError("n_initial must be ≥ 1"))
    n_initial <= model.population_size ||
        throw(ArgumentError("n_initial cannot exceed population_size"))

    from = _resolve_infectious_from(model.from, progression)
    β = _resolve_transmission_rate(model, progression, from)

    state = new_state(model, progression, attributes, rng)
    add_individuals!(state, model.population_size, interventions;
        setup = (ind, i) -> nothing)

    # The homogeneous pool is the one-type case of the structured Sellke pool:
    # no attributes name the mixing, so every individual feels the same force
    # β/N per infective (`sum(values(counts))` = number currently infectious).
    _sellke_pool!(state, collect(1:model.population_size), rng;
        force = (type, counts) -> β / model.population_size * sum(values(counts)),
        n_initial = n_initial, from = from, until = model.until)

    _reconcile_sellke_bookkeeping!(state)
    apply_observation!(observation, state, rng)
    return state
end

# β from the model: the direct rate if given, otherwise R0 / mean infectious
# period, with the mean read from the composed progression.
function _resolve_transmission_rate(model::HomogeneousProcess, progression, from)
    model.is_r0 || return model.rate
    return model.rate / _infectious_period_mean(progression, from)
end

# ── Deriving the infectious window from a progression ────────────────
# Shared by the structure-driven models: the infectious window's `from` state,
# and (for R0) the mean infectious period, are read from the composed
# progression when the model is simulated.

# The state the infectious window opens at: :infectious when the progression
# produces it (a latent period), otherwise :infection. An explicit `from`
# overrides the derivation.
_resolve_infectious_from(from::Symbol, progression) = from
_resolve_infectious_from(::Nothing, progression) = _infectious_from(progression)
function _infectious_from(progression)
    any(t -> hasproperty(t, :state) && getfield(t, :state) === :infectious,
        progression) ? :infectious : :infection
end

# Mean infectious period implied by the progression: the mean delay of the
# terminal transition leaving the infectious-window `from` state. Turns an R0
# into β; errors when no such mean is available.
function _infectious_period_mean(progression, from)
    terminals = filter(progression) do t
        hasproperty(t, :from) && getfield(t, :from) === from &&
            hasproperty(t, :terminal) && getfield(t, :terminal)
    end
    if isempty(terminals)
        # A removal measured from a different state (e.g. :infection while the
        # window opens at :infectious) is a common mismatch; point at it.
        others = filter(t -> hasproperty(t, :terminal) && getfield(t, :terminal),
            progression)
        hint = isempty(others) ? "" :
               " A removal transition exists but is measured from " *
               ":$(getfield(others[1], :from)) rather than the infectious-window " *
               "start :$from; set `from` on the process to match, or measure the " *
               "removal from :$from."
        throw(ArgumentError(
            "cannot derive β from R0: the progression has no terminal transition " *
            "from :$from, so there is no mean infectious period.$hint " *
            "Otherwise set transmission_rate (β) directly."))
    end
    if length(terminals) > 1
        # R0 = β·E[infectious period] is only well-defined for a single removal
        # route. With competing terminals the period is the race to the first,
        # which this mean ignores, as it ignores any transition probabilities.
        @warn "Deriving β from R0 with $(length(terminals)) removal transitions " *
              "from :$from: the infectious period is taken from the first " *
              "(:$(getfield(terminals[1], :state))), ignoring the competing-risks " *
              "race and any transition probabilities. Set transmission_rate (β) " *
              "directly for full control."
    end
    return _delay_mean(getfield(terminals[1], :delay))
end
_delay_mean(d::Distribution) = mean(d)
_delay_mean(x::Real) = float(x)
function _delay_mean(::Any)
    throw(ArgumentError(
        "cannot take the mean of a function-valued delay to derive β from R0; " *
        "set transmission_rate (β) directly."))
end
