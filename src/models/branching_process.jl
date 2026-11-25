# ── Infectiousness window ────────────────────────────────────────────

"""
    Infectiousness(offspring; from = :infection, until = (), kernel = NoGenerationTime(),
                   calendar = nothing, name = :transmission)

When, and how much, a case transmits by one route. `offspring` is the
distribution of the number of people infected by this route; transmission
starts when the case reaches state `from` and stops at the earliest of the
`until` states. Give a [`BranchingProcess`](@ref) several of these for
different routes, such as community and funeral transmission of Ebola.

- `offspring`: a distribution of secondary cases, such as `NegBin(R, k)`
  (mean `R`, dispersion `k`), or a function `(rng, ind) -> n` returning the
  number infected by a given case (one count per type in a multi-type
  model). The function may also take the outbreak state as a third argument
  (see [`draw_offspring`](@ref EpiBranch.draw_offspring)). When the function
  already knows when each contact falls, for example because it builds
  transmission day by day, it can return an [`OffspringTimes`](@ref) instead
  of a count, giving each contact's time from `from` directly and skipping
  the kernel. This only suits a single-type model: `OffspringTimes` has no
  per-type label, so a multi-type model's function must keep returning a
  count.
- `from`: the state at which transmission starts, `:infection` by default.
  Any other name refers to a state set by a step of the natural history, such
  as `:infectious` or `:died`; transmission starts only once the case reaches
  it.
- `until`: the states at which transmission stops, for example
  `(:recovered, :died)` for community transmission or `(:buried,)` for a
  funeral. The earliest one reached ends transmission, and infections that
  would have happened later do not. Empty by default. Isolation is set with
  the [`Isolation`](@ref) intervention, not here. A factor that scales
  transmission for the whole infection, such as lower infectiousness of
  asymptomatic cases, is set with [`transmission_traits`](@ref) instead.
- `kernel`: the time in days from `from` to each infection: a distribution,
  a function of the individual returning a distribution, `(ind) -> ...`, or
  `NoGenerationTime()` to place every infection at the `from` time. With the
  default `from = :infection` and no `until`, this is the generation time.
  With `until` it is the contact interval (the time to a contact that would
  infect if nothing stopped it), and the generation time follows from which
  comes first, the contact or the end of transmission, so do not also
  shorten it by hand. A kernel measured from a later state, such as onset,
  can put a contact before the case's own infection. No exposure happens
  then, so that contact is dropped, and a case makes fewer contacts on
  average than its offspring distribution draws, by the share of the kernel
  that falls before infection. Unused for a contact whose time `offspring`
  already gave as an `OffspringTimes`.
- `calendar`: a [`Steps`](@ref) or [`Seasonal`](@ref) schedule, or a callable
  `t -> multiplier`, scaling `kernel`'s hazard by the schedule's value on the
  calendar day, as the `calendar` of a [`PairKernel`](@ref) does, for a
  lockdown or a seasonal pattern that changes how quickly this window's
  contacts arrive. Unused where `kernel` draws no contact
  (`NoGenerationTime()`) or where `offspring` already timed them
  (`OffspringTimes`). `end_of_outbreak_probability` and `Isolation`'s
  analytic effect both read a
  window's `kernel` directly and throw an `ArgumentError` on a window with a
  `calendar` rather than ignoring it; simulate a model that needs one
  instead.
- `name`: the name interventions see this window as, through
  [`risk_applies`](@ref EpiBranch.risk_applies)'s `route`. The default,
  `:transmission`, suits a process with a single window; give each of several
  windows its own name, for example `:community` and `:household`, so
  wrapping a measure in [`OnRoutes`](@ref) can act on one and leave the rest
  open.

See also [`HazardKernel`](@ref) for a `kernel` with a rate that follows
something computed per case, such as a within-host trajectory, rather than a
named distribution.

# Examples
```julia
using EpiBranch, Distributions

progression = [
    Transition(:infectious, from = :infection, delay = Gamma(4.0, 2.0)),
    Transition(:died, from = :infectious, delay = Gamma(4.0, 2.0),
        probability = 0.6, terminal = true),
    Transition(:recovered, from = :infectious, delay = Gamma(5.0, 2.0),
        terminal = true),
    Transition(:buried, from = :died, delay = 2.0),
]
community = Infectiousness(NegBin(1.2, 0.5); name = :community,
    from = :infectious, until = (:recovered, :died), kernel = Exponential(4.0))
funeral = Infectiousness(Poisson(0.5); name = :funeral,
    from = :died, until = (:buried,), kernel = Uniform(0.0, 2.0))
model = ModelSpec(BranchingProcess(community, funeral); progression)
```
"""
struct Infectiousness{O, F, U, K, C}
    offspring::O
    from::F
    from_key::Union{Symbol, Nothing}
    until::U
    kernel::K
    calendar::C
    name::Symbol
end
function Infectiousness(
        offspring; from = :infection, until = (),
        kernel = NoGenerationTime(), calendar = nothing, name::Symbol = :transmission
    )
    return Infectiousness(offspring, from, _from_key(from), until, kernel, calendar, name)
end

"""
    OffspringTimes(times)

The times of one case's contacts on a window, measured from the window's
`from` state, returned by an offspring function in place of a count. An
offspring function that already knows when each contact falls, such as one
built day by day, returns this to place every contact at its own time rather
than losing that to an independent draw from the window's `kernel`, which
this makes unnecessary and the simulation then never calls.

Has no per-type label, so it only works with a single-type model
(`n_types = 1`); a multi-type model raises an error rather than assigning
every contact to type 1. A multi-type offspring function still returns a
per-type `Vector{Int}` of counts.

# Examples
```julia
using EpiBranch

# a case infects 10 people on each of the days its state records an event
function offspring(rng, ind)
    days = findall(ind.state[:events]) .- 0.5
    return OffspringTimes(repeat(days, inner = 10))
end
Infectiousness(offspring; until = (:stopped,))
```
"""
struct OffspringTimes{T <: Real}
    times::Vector{T}
end

# ── Transmission timing from natural history ─────────────────────────

"""
    transmission_timing(contact_interval; offspring = nothing, contacts = nothing,
                        latent_period = nothing, infectious_period = nothing,
                        name::Symbol = :transmission)

Build the [`Transition`](@ref)s and the [`Infectiousness`](@ref) window that a
case's natural history needs, from whichever timing is known: a plain
generation time, or a latent and an infectious period with a contact
interval. Returns `(progression, window)`: `progression` for a
[`ModelSpec`](@ref), `window` for a [`BranchingProcess`](@ref).

With neither `latent_period` nor `infectious_period`, `contact_interval` is a
generation time: `progression` is empty and `window` opens at infection and
never closes, exactly as `BranchingProcess(offspring, contact_interval)`
builds it. Give either to place transmission on the case's own natural
history instead: `window` opens at `:infectious` once `latent_period` moves
it there, and closes at `:recovered` once `infectious_period` gives that
step something to wait for. `contact_interval` is then the contact interval,
drawn independently of `infectious_period`.

Exactly one of `contacts` and `offspring` sets how many people a case
reaches; both mean the same thing as the `offspring` distribution of a
[`BranchingProcess`](@ref) built from a plain generation time, but only one
keeps it so once there is a `:recovered` to cut transmission short:

- `contacts`: the number of contacts the case would make if it stayed
  infectious for as long as `contact_interval` ever needs, independent of how
  long its own infectious period turns out to be. One whose draw from
  `contact_interval` falls after `:recovered` is thinned away by the window
  closing, as `until` ordinarily works, so the realised reproduction number is
  `mean(contacts)` times the chance a draw from `contact_interval` falls
  before `infectious_period`, lower than `mean(contacts)` itself (see
  [`implied_R`](@ref)). A case's infectious period length and its number of
  secondary cases are linked through this thinning, as they are for a real
  infection.
- `offspring`: the number of secondary cases, with `mean` and dispersion
  exactly as given, unaffected by `infectious_period`. Each is still timed
  from `contact_interval`, but conditioned on falling before the case's own
  `:recovered` rather than thinned away if it would not, so every one of them
  is realised. This keeps R and its dispersion exact at the cost of the link
  `contacts` keeps: the number of secondary cases a case has no longer
  depends on how long it happens to stay infectious, only on when within
  that period each one falls.

# Examples

```julia
using EpiBranch, Distributions

# a generation time: the same thing `BranchingProcess(offspring, gt)` already builds
timing = transmission_timing(LogNormal(1.6, 0.5); offspring = NegBin(2.5, 0.16))
model = BranchingProcess(timing.window)

# latent and infectious periods with a contact interval; R realised after
# thinning by recovery
timing = transmission_timing(
    Exponential(4.0); contacts = NegBin(2.5, 0.16),
    latent_period = LogNormal(1.6, 0.5), infectious_period = Gamma(5.0, 2.0)
)
model = ModelSpec(BranchingProcess(timing.window); progression = timing.progression)

# the same natural history, keeping R and its dispersion exact instead
timing = transmission_timing(
    Exponential(4.0); offspring = NegBin(2.5, 0.16),
    latent_period = LogNormal(1.6, 0.5), infectious_period = Gamma(5.0, 2.0)
)
```
"""
function transmission_timing(
        contact_interval;
        offspring = nothing, contacts = nothing,
        latent_period = nothing, infectious_period = nothing,
        name::Symbol = :transmission
    )
    (offspring === nothing) == (contacts === nothing) && throw(
        ArgumentError("transmission_timing needs exactly one of `offspring` or `contacts`")
    )

    from = latent_period === nothing ? :infection : :infectious
    until = infectious_period === nothing ? () : (:recovered,)

    progression = AbstractClinicalTransition[]
    latent_period === nothing ||
        push!(progression, Transition(:infectious; from = :infection, delay = latent_period))
    infectious_period === nothing ||
        push!(progression, Transition(:recovered; from, delay = infectious_period, terminal = true))

    kernel = offspring === nothing ? contact_interval :
        _conditioned_on_recovery(contact_interval, from, infectious_period)
    window = Infectiousness(something(offspring, contacts); from, until, kernel, name)

    return (progression = progression, window = window)
end

# The kernel for the `offspring` form: `contact_interval`, conditioned on
# falling before the case's own `:recovered` rather than thinned away if it
# would not, so every draw from `offspring` is realised. A plain generation
# time (no `infectious_period`, hence no `:recovered` to fall before) has
# nothing to condition on.
_conditioned_on_recovery(contact_interval, from::Symbol, ::Nothing) = contact_interval
function _conditioned_on_recovery(contact_interval, from::Symbol, infectious_period)
    return function (ind)
        open_t = from === :infection ? ind.infection_time : ind.state[:infectious_time]
        duration = ind.state[:recovered_time] - open_t
        return truncated(get_generation_time(contact_interval, ind), 0.0, duration)
    end
end

"""
    implied_R(contacts, contact_interval, infectious_period)

The reproduction number realised by the `contacts` form of
[`transmission_timing`](@ref) once recovery thins its contacts:
`mean(contacts)` times the probability that a draw from `contact_interval`
falls before `infectious_period`, the two assumed independent, as
`transmission_timing` draws them.

`contact_interval` is a `Distribution`, a [`HazardKernel`](@ref) among
them; `infectious_period` is a `Distribution` or a fixed number of days. The
distribution case integrates over `infectious_period` numerically (with
QuadGK.jl).

# Examples
```julia
implied_R(NegBin(2.5, 0.16), Exponential(4.0), Gamma(5.0, 2.0))
```
"""
function implied_R(contacts, contact_interval::Distribution, infectious_period::Real)
    return mean(contacts) * cdf(contact_interval, infectious_period)
end
function implied_R(contacts, contact_interval::Distribution, infectious_period::Distribution)
    p = first(quadgk(x -> cdf(contact_interval, x) * pdf(infectious_period, x), 0.0, Inf))
    return mean(contacts) * p
end

# ── BranchingProcess type and constructors ──────────────────────────

"""
    AbstractResolution

The level at which a [`BranchingProcess`](@ref) is simulated: [`Individuals`](@ref)
(the default) or [`ByGeneration`](@ref). `simulate` dispatches on it.
"""
abstract type AbstractResolution end

"""
    Individuals()

The default `resolution` of a [`BranchingProcess`](@ref): `simulate` creates an
`Individual` for every case and returns a [`SimulationState`](@ref).
"""
struct Individuals <: AbstractResolution end

"""
    BranchingProcess(offspring, generation_time; population = NoPopulation(),
                     multiplier = nothing, importations = nothing,
                     resolution = Individuals())
    BranchingProcess(offspring)
    BranchingProcess(windows::Infectiousness...)

A stochastic branching process: each case independently infects a random
number of others, drawn from the offspring distribution `offspring`, and the
outbreak grows as a tree from the index cases. A negative binomial,
`NegBin(R, k)` with mean `R` and dispersion `k`, is the usual choice; smaller
`k` means more superspreading.

`generation_time` is the distribution of the time in days from a case's
infection to the infection of each of its secondary cases. Leave it out to
study only chain sizes and lengths, without timing. `population` sets a
finite [`Population`](@ref): each potential infection succeeds with
probability equal to the share of people of its type not yet infected.
Transmission slows as cases accumulate (unlimited by default).
`population_size = n` is shorthand for `population = Population(n)`. A
multi-type model (an offspring function or a contact matrix) takes the size of
each type's population, `Population(sizes)`, and depletes each type against
its own size. The `immune` of the population are never infected.

`multiplier` is a `Distribution` of mean one, or one per type, drawn once for
each generation (index cases are generation 0): it scales the offspring mean of
every case in that generation, across every infectiousness window. A vector
gives each type its own draw. The offspring must be a `Distribution` with an
[`EpiBranch.scale_mean`](@ref) method, or an offspring matrix whose laws have
one; any other model raises an `ArgumentError` when built. A model with a
`multiplier` runs as individuals, not on the daily count-level path of
[`simulate`](@ref). A model with `resolution = ByGeneration()` draws the same
multiplier in each generation of its counts.

`resolution` is [`Individuals`](@ref) (the default) or [`ByGeneration`](@ref).
With [`ByGeneration`](@ref), [`simulate`](@ref) returns a
[`GenerationCounts`](@ref) of case counts instead of a [`SimulationState`](@ref).

`importations` is a vector of times at which further index cases arrive,
or a rate of arrivals per day: a number, a [`Steps`](@ref) schedule giving
the rate on each day, or a function of the day returning it. A
[`Seasonal`](@ref) has no scale and gives about one arrival a day. To scale
it, use a function, `t -> 5 * EpiBranch.calendar_multiplier(Seasonal(...), t)`.
A rate needs `max_time` and draws the
arrivals as a Poisson process over it. A function of the day is bounded by
its largest value on a 1001-point grid over `max_time`. A peak narrower than
the grid spacing can be missed. An arrival on a day when the rate exceeds
that bound raises an `ArgumentError`. Imports come in addition to the
`n_initial` index cases at time 0, and their `:infection_route` is
`:external`. In a multi-type model each import's type is drawn uniformly from
the types; `index_types` applies only to the index cases at time 0. Only the
offspring route of `BranchingProcess` reads `importations`; a model with its
own `initialise_state` does not. Simulating with `resolution = ByGeneration()`
and starting from observed cases both raise an `ArgumentError` for a model with
`importations`.

The transmission model describes only who infects whom and when. Add the
natural history, interventions, population characteristics and reporting
with a [`ModelSpec`](@ref).

# Examples

```julia
# R = 2.5, k = 0.16; generation time log-normal with log-mean 1.6 and
# log-sd 0.5 (mean about 5.6 days)
BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))

# no timing: chain sizes and lengths only
BranchingProcess(NegBin(0.8, 0.5))

# two types (e.g. children and adults): M[i, j] is the mean number of
# type-i cases infected by one type-j case
M = [1.2 0.4; 0.3 0.9]
BranchingProcess(M, R -> NegBin(R, 0.16), LogNormal(1.6, 0.5))

# one index case at time 0 and imports on days 7 and 21
BranchingProcess(NegBin(0.9, 0.5), LogNormal(1.6, 0.5); importations = [7.0, 21.0])

# imports at 0.2 a day, falling to 0.05 a day from day 30 (needs `max_time`)
BranchingProcess(NegBin(0.9, 0.5), LogNormal(1.6, 0.5);
    importations = Steps([30.0], [0.2, 0.05]))

# add isolation with a ModelSpec
ModelSpec(BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5));
    attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
    interventions = [Isolation(delay = Exponential(2.0), duration = 7.0)])
```

For transmission by several routes, or starting and stopping at points in
the natural history, build the process from [`Infectiousness`](@ref)
windows. The forms above make a single one that starts at infection.
"""
struct BranchingProcess{W <: Tuple, P, L, M, I, R <: AbstractResolution} <: TransmissionModel
    infectiousness::W
    population::P
    n_types::Int
    type_labels::L
    multiplier::M
    importations::I
    resolution::R
end

function BranchingProcess(windows::Tuple, population, n_types::Int, type_labels)
    return BranchingProcess(
        windows, population, n_types, type_labels, nothing, nothing, Individuals()
    )
end

_check_multiplier(::Nothing, windows, n_types::Integer) = nothing
function _check_multiplier(multiplier, windows, n_types::Integer)
    all(d -> d isa UnivariateDistribution, multiplier isa AbstractVector ? multiplier : (multiplier,)) ||
        throw(
        ArgumentError(
            "`multiplier` must be a Distribution, or a vector with one per type, " *
                "not $(typeof(multiplier))"
        )
    )
    multiplier isa AbstractVector && length(multiplier) != n_types && throw(
        ArgumentError(
            "`multiplier` has $(length(multiplier)) distributions for $n_types types"
        )
    )
    foreach(w -> _check_scalable(w.offspring), windows)
    return multiplier
end
_check_scalable(o::Distribution) = scale_mean(o, 1.0)
function _check_scalable(o)
    throw(
        ArgumentError(
            "`multiplier` scales the mean of an offspring law, so the offspring must be " *
                "a Distribution or an offspring matrix, not $(typeof(o))"
        )
    )
end

_check_importations(::Nothing) = nothing
function _check_importations(times::AbstractVector{<:Real})
    all(t -> isfinite(t) && t >= 0, times) || throw(
        ArgumentError("`importations` must be finite, non-negative times")
    )
    return collect(times)
end
function _check_importations(rate::Real)
    isfinite(rate) && rate >= 0 || throw(
        ArgumentError("an `importations` rate must be finite and non-negative")
    )
    return Float64(rate)
end
function _check_importations(schedule)
    hasmethod(calendar_multiplier, Tuple{typeof(schedule), Float64}) || throw(
        ArgumentError(
            "`importations` must be a vector of times, a rate, a " *
                "`Steps` schedule or a function of time, got $(typeof(schedule))"
        )
    )
    return schedule
end

_import_times(::TransmissionModel, ::SimOpts, rng) = ()
function _import_times(m::BranchingProcess, opts::SimOpts, rng)
    imp = m.importations
    imp === nothing && return ()
    imp isa AbstractVector && return imp
    horizon = Float64(minimum(time_bound, opts.stopping_rules; init = Inf))
    isfinite(horizon) || throw(
        ArgumentError(
            "importations at a rate need a finite time limit: pass `max_time`"
        )
    )
    rate_at(t) = imp isa Real ? Float64(imp) : Float64(calendar_multiplier(imp, t))
    bound = imp isa Real ? Float64(imp) : calendar_bound(imp)
    bound === nothing && (bound = 1.1 * maximum(rate_at, range(0.0, horizon; length = 1001)))
    times = Float64[]
    bound > 0 || return times
    t = randexp(rng) / bound
    while t < horizon
        rate = rate_at(t)
        rate <= bound || throw(
            ArgumentError(
                "the `importations` rate $rate on day $t exceeds its bound $bound; " *
                    "a function of the day is bounded by its largest value on a grid " *
                    "over `max_time`, so use a `Steps` schedule for a " *
                    "narrow peak"
            )
        )
        rand(rng) * bound <= rate && push!(times, t)
        t += randexp(rng) / bound
    end
    return times
end

# The population a process is built with, from the `population` keyword or
# the `population_size` shorthand for one group, checked against the number of
# types: each type is a group with a size of its own.
function _branching_population(population, population_size, n_types::Integer)
    pop = _population_input(population, population_size)
    _check_type_sizes(pop, n_types)
    return pop
end
_check_type_sizes(::NoPopulation, ::Integer) = nothing
_check_type_sizes(::SinglePopulation, ::Integer) = nothing
function _check_type_sizes(p::GroupedPopulation, n_types::Integer)
    length(p.sizes) == n_types || throw(
        ArgumentError(
            "`population` has $(length(p.sizes)) groups for a model with $n_types types"
        )
    )
    return nothing
end

"""
    validate_process_windows(process::TransmissionModel, progression)

Cross-tier check, run when a process is composed with a progression in a
[`ModelSpec`](@ref): warn when an infectiousness window opens at a `from` state
the progression never produces, because the window would then never open. Also
warn when a window's `until` omits a terminal state the progression can reach,
because the case would go on transmitting after it. `from`
may legitimately be a state written by an attribute or intervention rather than
a progression transition, so this is a warning. The default does nothing, for
models without infectiousness windows; a [`FixedPopulationProcess`](@ref
EpiBranch.FixedPopulationProcess) checks its `until` states against the
progression's terminal states, and a structure with several windows extends this
and calls [`warn_uncovered_terminal_states`](@ref
EpiBranch.warn_uncovered_terminal_states) for each.
"""
validate_process_windows(::TransmissionModel, progression) = nothing
function validate_process_windows(m::BranchingProcess, progression)
    return _validate_windows(m.infectiousness, progression)
end
function _validate_windows(windows, progression)
    for w in windows
        warn_uncovered_terminal_states(
            w.until, progression;
            route = length(windows) > 1 ? w.name : nothing, from = w.from
        )
    end
    produced = Set{Symbol}((:infection,))
    for t in progression
        union!(produced, produced_states(t))
    end
    for w in windows
        w.from isa Symbol || continue
        (w.from === :infection || w.from in produced) && continue
        @warn "Infectiousness window has from = :$(w.from), which no progression " *
            "transition produces; the window only opens if something sets " *
            ":$(Symbol(w.from, :_time)) (an attribute, intervention, or transition)."
    end
    return nothing
end

# The terminal states a window's `until` (and `from`) does not cover, which
# the continuous-time structure-driven models warn about: a case reaching such
# a state keeps generating exposure proposals as if still infectious. A
# terminal transition that does not implement `terminal_target` stays
# exempt, so a custom one must opt in to be covered.
function _uncovered_terminal_states(until::Tuple, progression; from = nothing)
    covered = Set{Symbol}(until)
    from isa Symbol && push!(covered, from)
    states = Symbol[]
    for t in progression
        is_terminal(t) || continue
        target = terminal_target(t)
        target === nothing && continue
        target in covered || push!(states, target)
    end
    return unique(states)
end

"""
    warn_uncovered_terminal_states(until::Tuple, progression; route = nothing, from = nothing)

Warn when `until` does not cover every terminal state the progression can
reach, so the silent runaway of a case that keeps generating exposure proposals
is discoverable rather than showing up as an implausibly large outbreak.
`route` labels the warning when `until` belongs to one window among several, and
`from` is that window's own opening state, which counts as covered. A terminal
transition that does not implement `terminal_target` is exempt.
"""
function warn_uncovered_terminal_states(
        until::Tuple, progression;
        route = nothing, from = nothing
    )
    states = _uncovered_terminal_states(until, progression; from)
    isempty(states) && return nothing
    on_route = route === nothing ? "" : " on route :$route"
    @warn "Progression has a terminal transition to " *
        "$(join((":" * String(s) for s in states), ", ")), which `until`" *
        "$on_route $until does not list. A case reaching it never has its " *
        "infectious/exposure window closed, and keeps generating exposure " *
        "proposals indefinitely. Add it to `until` if it should end " *
        "transmission."
    return nothing
end

# Natural history, interventions, attributes and observation are not carried by
# the process — they are composed onto it with a [`ModelSpec`](@ref). The
# shared accessors (in model_inputs.jl and model_spec.jl) resolve to empty
# defaults for a bare process, so `_progvec` stays here for the spec to use.
_progvec(p) = convert(Vector{AbstractClinicalTransition}, p)
_progvec(::Tuple{}) = AbstractClinicalTransition[]

population(m::BranchingProcess) = m.population

# A multi-type process given only a total has no size for each type to deplete
# against. `simulate_by_generation` can still take the sizes as the deprecated
# `population_by_type`; the individual-based simulation cannot.
function _check_population(m::BranchingProcess)
    (m.n_types > 1 && m.population isa SinglePopulation) || return nothing
    throw(
        ArgumentError(
            "a multi-type BranchingProcess needs the size of each type's population to " *
                "deplete each type against its own: give " *
                "`population = Population(sizes)` with $(m.n_types) sizes, not " *
                "`population_size`"
        )
    )
end
population_size(m::BranchingProcess) = total_size(m.population)
n_types(m::BranchingProcess) = m.n_types

# A finite population thins transmission as cases accumulate, a window with
# `until` stops it before every drawn contact happens, a window's
# `calendar` scales its kernel's hazard over time, and a `multiplier` makes
# cases of a generation share their variation; all four reshape the
# offspring law away from the bare distribution, so the analytical chain-size
# likelihood must not take the model's offspring law as read.
function offspring_law_unchanged(m::BranchingProcess)
    m.population isa NoPopulation || return false
    m.multiplier === nothing || return false
    return all(w -> isempty(w.until) && w.calendar === nothing, m.infectiousness)
end

function single_type_offspring(m::BranchingProcess)
    length(m.infectiousness) == 1 || throw(
        ArgumentError(
            "Analytical helpers need a single infectiousness window (this model has " *
                "$(length(m.infectiousness))). The offspring law across several windows is a " *
                "fate-mixture with no closed form, so use simulation for multi-window models."
        )
    )
    return _single_type(m.infectiousness[1].offspring)
end

# The single-type law of an offspring specification, or an error for the kinds
# that have none: an offspring function here, and `MultiTypeOffspring` in
# multi_type_offspring.jl.
_single_type(off) = off
function _single_type(::Function)
    throw(
        ArgumentError(
            "This function only works with single-type models (not multi-type function offspring)"
        )
    )
end

# The contact interval of a single-window model (used by analytical
# helpers that assume one generation-time distribution).
function _single_kernel(m::BranchingProcess)
    length(m.infectiousness) == 1 || throw(
        ArgumentError(
            "this analytical helper needs a single infectiousness window; this model has $(length(m.infectiousness))"
        )
    )
    return m.infectiousness[1].kernel
end

# The calendar of a single-window model (used by analytical helpers that
# read `_single_kernel` directly and have no way to apply it).
_single_calendar(m::BranchingProcess) = m.infectiousness[1].calendar

_offspring_label(off::Distribution) = string(typeof(off))
_offspring_label(off) = "Function"

function Base.show(io::IO, m::BranchingProcess)
    pop_str = m.population isa NoPopulation ? "unlimited" :
        m.population isa SinglePopulation ? string(m.population.size) :
        string(m.population.sizes)
    return if length(m.infectiousness) == 1
        w = m.infectiousness[1]
        off_str = _offspring_label(w.offspring)
        gt_str = w.kernel isa NoGenerationTime ? "none" :
            w.kernel isa Distribution ? string(typeof(w.kernel)) : "Function"
        print(
            io,
            "BranchingProcess(offspring=$(off_str), generation_time=$(gt_str), population_size=$(pop_str))"
        )
    else
        print(
            io,
            "BranchingProcess($(length(m.infectiousness)) infectiousness windows, population_size=$(pop_str))"
        )
    end
end

# Single-type with a contact interval (one default window).
function BranchingProcess(
        offspring::Distribution, gt;
        population = NoPopulation(), population_size = NoPopulation(), multiplier = nothing,
        importations = nothing,
        resolution = Individuals()
    )
    return BranchingProcess(
        (Infectiousness(offspring; kernel = gt),); population, population_size, multiplier,
        importations, resolution
    )
end

# Single-type without a contact interval (pure chain statistics).
function BranchingProcess(
        offspring::Distribution;
        population = NoPopulation(), population_size = NoPopulation(), multiplier = nothing,
        importations = nothing,
        resolution = Individuals()
    )
    return BranchingProcess(
        (Infectiousness(offspring),); population, population_size, multiplier, importations,
        resolution
    )
end

# Multi-type with an explicit offspring function.
function BranchingProcess(
        offspring, gt;
        n_types::Int = 1, population = NoPopulation(), population_size = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels(), multiplier = nothing,
        importations = nothing,
        resolution = Individuals()
    )
    return BranchingProcess(
        (Infectiousness(offspring; kernel = gt),); n_types, population, population_size,
        type_labels, multiplier, importations, resolution
    )
end

# Explicit windows: pass `Infectiousness` windows directly.
function BranchingProcess(
        windows::Tuple{Infectiousness, Vararg{Infectiousness}};
        n_types::Int = 1, population = NoPopulation(), population_size = NoPopulation(),
        type_labels::Union{Vector{String}, NoTypeLabels} = NoTypeLabels(), multiplier = nothing,
        importations = nothing,
        resolution = Individuals()
    )
    pop = _branching_population(population, population_size, n_types)
    return BranchingProcess(
        windows, pop, n_types, _type_labels(pop, type_labels),
        _check_multiplier(multiplier, windows, n_types), _check_importations(importations),
        resolution
    )
end
function BranchingProcess(window::Infectiousness, windows::Infectiousness...; kwargs...)
    return BranchingProcess((window, windows...); kwargs...)
end

# ── Offspring drawing ────────────────────────────────────────────────

"""Draw the number of secondary cases of one case from an offspring
distribution."""
function draw_offspring(
        rng::AbstractRNG, offspring::Distribution,
        individual, state::SimulationState
    )
    return rand(rng, offspring)
end

"""Draw the number of secondary cases of one case from an offspring
function, written either as `(rng, individual) -> n` or as
`(rng, individual, state) -> n`. The second form can read the state of the
whole outbreak, for example to reduce transmission once the case count
passes a threshold. Either form may return an [`OffspringTimes`](@ref)
instead of `n`, giving each contact's time directly."""
function draw_offspring(rng, offspring, individual, state)
    if applicable(offspring, rng, individual, state)
        return offspring(rng, individual, state)
    end
    return offspring(rng, individual)
end
