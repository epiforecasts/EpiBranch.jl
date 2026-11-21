# ── Simulating a branching process by day, as counts ─────────────────
#
# A model in which nothing about an individual feeds back on transmission has
# the same joint distribution of daily counts whether each person is simulated
# or only the cohort of new cases on each day. The cohort infected on day `s`
# has `n` members, each with an infectiousness `ν` of mean one. The members of
# a cohort share nothing, so the total infectiousness `Λ` of the cohort is the
# sum of its `ν`: `Gamma(n k, 1/k)` when `ν ~ Gamma(k, 1/k)` (a negative
# binomial offspring law, or `RenewalOffspring` with a finite `k`), `n` when
# every `ν` is one, and otherwise the sum of `n` draws of `ν` (a Poisson
# lognormal or Poisson Weibull law, whose `ν` is the rate divided by its
# mean). The cohort infects `Poisson(Λ R(t) g(t - s))` people on day `t`, independently for each `t`, with `R(t)` the mean offspring number
# at `t` and `g` the generation time on the daily lags.
#
# The result keeps, for each day, the count of the cohort, its `Λ` and the
# number of infections it received from each earlier cohort. That is enough to
# continue the run and to build the people when they are asked for. The
# infections caused by a cohort are shared among its members in proportion to
# their `ν`, which are `Dirichlet(k, …, k)` given `Λ` for a gamma `ν` (equal
# shares when every `ν` is one). For another `ν` the values of the members are
# drawn at that point, from their law given the sum.

"""
    EpiBranch.count_rule(model)

The rule by which [`simulate`](@ref) runs `model` as daily counts rather than
as individuals, or `nothing` if `model` has to be simulated as individuals.
A model qualifies when nothing about an individual feeds back on
transmission, apart from the depletion of a finite population by type, and its
generation time is a distribution on whole days. Adding a method for a new
model type gives it the count-level path. The rule is a
[`CountRule`](@ref), or a [`MultiTypeCountRule`](@ref) for several types.
"""
count_rule(::TransmissionModel) = nothing

"""
    AbstractCountRule

A rule by which a cohort infects people on later days:
[`CountRule`](@ref) or [`OffspringCountRule`](@ref).
"""
abstract type AbstractCountRule end

"""
    CountRule(R, ν, g)

How a cohort of `n` people infected on one day infects people on later days
([`EpiBranch.count_rule`](@ref)): the number infected `lag` days later, on day
`t`, is `Poisson(Λ R(t) g[lag])`. `Λ` is the total infectiousness of the cohort, the sum
of the `n` infectiousnesses drawn from `ν`, a distribution of mean one:
`Dirac(1.0)` when every member is alike and `Gamma(k, 1/k)` for a negative
binomial offspring law, where `Λ` takes one draw as `Gamma(n k, 1/k)`. Any
other `ν` takes `n` draws, and the cost and memory grow with the number of
cases. A shape `k` in place of `ν` stands for `Gamma(k, 1/k)`. `R` is a function of the day.
"""
struct CountRule{F, D <: UnivariateDistribution} <: AbstractCountRule
    R::F
    ν::D
    g::Vector{Float64}
end

# A gamma shape `k` in place of `ν`; infinite `k` is a point mass.
CountRule(R, k::Real, g) = CountRule(R, _gamma_unit(k), g)

"""
    OffspringCountRule(law, g)

A rule for an offspring `law` given by a discrete distribution on the non-negative integers that is not a Poisson mixture, such as `Binomial` or `Geometric`. Each member of a
cohort draws its number of offspring from `law` when infected, and each
offspring falls on lag `l` with probability `g[l]`.
"""
struct OffspringCountRule{D <: DiscreteUnivariateDistribution} <: AbstractCountRule
    law::D
    g::Vector{Float64}
end

"""
    MultiTypeCountRule(types, alloc)

The [`CountRule`](@ref) of a model with several types: `types[j]` is the rule of
a type-`j` case (its `R` and `ν`, with the generation time `g` shared by every
type), and `alloc[i, j]` is the probability that a person a type-`j` case
infects is of type `i`. The infections of type `i` caused by a cohort of type
`j` on a day are `Poisson(Λ R(t) g[lag] alloc[i, j])`.
"""
struct MultiTypeCountRule{T <: AbstractCountRule}
    types::Vector{T}
    alloc::Matrix{Float64}
end

_type_rules(rule::AbstractCountRule) = (rule,)
_type_rules(rule::MultiTypeCountRule) = rule.types
_alloc(::AbstractCountRule) = _SINGLE_TYPE_ALLOC
_alloc(rule::MultiTypeCountRule) = rule.alloc
_lags(rule::AbstractCountRule) = rule.g
_lags(rule::MultiTypeCountRule) = first(rule.types).g
const _SINGLE_TYPE_ALLOC = ones(1, 1)

# A `ModelSpec` qualifies when the layers it adds do not act on transmission.
function count_rule(spec::ModelSpec)
    isempty(spec.interventions) || return nothing
    spec.case_definition === nothing || return nothing
    spec.recorder isa NoContactRecorder || return nothing
    return count_rule(spec.process)
end

function count_rule(model::BranchingProcess)
    length(model.infectiousness) == 1 || return nothing
    isempty(transmission_risks(model)) || return nothing
    _counts_population(population(model), n_types(model)) || return nothing
    w = only(model.infectiousness)
    w.from === :infection && isempty(w.until) && w.calendar === nothing || return nothing
    rule = _count_rule(w.offspring, w.kernel)
    rule isa MultiTypeCountRule || n_types(model) == 1 || return nothing
    return rule
end

_counts_population(::NoPopulation, nt) = true
_counts_population(::SinglePopulation, nt) = nt == 1
_counts_population(p::GroupedPopulation, nt) = length(p.sizes) == nt

_count_sizes(::NoPopulation) = nothing
_count_sizes(p::SinglePopulation) = [p.size]
_count_sizes(p::GroupedPopulation) = copy(p.sizes)

# A type that infects no one has no offspring law to read.
function _count_rule(o::MultiTypeOffspring, kernel)
    rules = map(1:_n_types(o)) do j
        R = o.R_by_type[j]
        R <= 0 && return missing
        return _count_rule(_type_law(o.dist_fn, R, j), kernel)
    end
    live = collect(skipmissing(rules))
    any(isnothing, live) && return nothing
    kind = isempty(live) ? CountRule : typeof(first(live)).name.wrapper
    all(r -> r isa kind, live) || return nothing
    sink = kind === CountRule ? _daily_rule(_ConstantR(0.0), Dirac(1.0), kernel) :
        OffspringCountRule(Binomial(1, 0.0), _lags(first(live)))
    sink === nothing && return nothing
    types = kind[ismissing(r) ? sink : r for r in rules]
    return MultiTypeCountRule(types, Matrix{Float64}(o.alloc_probs))
end

_count_rule(o::Poisson, kernel) = _daily_rule(_ConstantR(mean(o)), Dirac(1.0), kernel)
function _count_rule(o::NegativeBinomial, kernel)
    0 < o.p < 1 || return nothing
    return _daily_rule(_ConstantR(mean(o)), _gamma_unit(o.r), kernel)
end
_count_rule(o::_PoissonMixture, kernel) =
    _daily_rule(_ConstantR(mean(o)), _unit_rate_law(o), kernel)
function _count_rule(o::RenewalOffspring{<:Any, <:AbstractVector}, kernel)
    return CountRule(o.R, _gamma_unit(o.k), o.generation_time)
end

# The infectiousness of one member: mean one, `Gamma(k, 1/k)`, or a point mass
# at one when `k` is infinite.
_gamma_unit(k) = isfinite(k) ? Gamma(Float64(k), 1 / Float64(k)) : Dirac(1.0)
# The rate law of a Poisson mixture divided by its mean.
_unit_rate_law(o::PoissonLognormal) = LogNormal(-o.σ^2 / 2, Float64(o.σ))
_unit_rate_law(o::PoissonWeibull) = Weibull(Float64(o.k), Float64(o.λ) / mean(_rate_law(o)))
function _count_rule(o::DiscreteUnivariateDistribution, kernel)
    minimum(o) >= 0 || return nothing
    g = _daily_lags(kernel)
    return g === nothing ? nothing : OffspringCountRule(o, g)
end
_count_rule(o, kernel) = nothing

# The generation time as a pmf on lags 1, 2, …, when it is a distribution on
# whole days from 1 with a largest value.
function _daily_lags(kernel)
    kernel isa DiscreteUnivariateDistribution || return nothing
    lo, hi = minimum(kernel), maximum(kernel)
    lo >= 1 && isfinite(hi) && hi <= 10_000 || return nothing
    g = Float64[pdf(kernel, lag) for lag in 1:Int(hi)]
    return isapprox(sum(g), 1; atol = 1.0e-9) ? g : nothing
end

function _daily_rule(R, ν, kernel)
    g = _daily_lags(kernel)
    return g === nothing ? nothing : CountRule(R, ν, g)
end

"""
    CountSimulationState

The result of [`simulate`](@ref) for a model that runs on the count-level
path ([`EpiBranch.count_rule`](@ref)): the number of people infected on each
day, rather than the people. It is an [`AbstractSimulationState`](@ref), so
[`individuals`](@ref), [`linelist`](@ref), [`contacts`](@ref),
[`chain_statistics`](@ref), [`weekly_incidence`](@ref) and the other tables
read from it as from a [`SimulationState`](@ref).

The people are built the first time they are asked for, from a random number
stream the state holds, and are then kept. A seed gives the same line list
whenever it is asked for. The infector of each case, the clinical
presentation and onset of each person, and the reporting of an `observation`
are all drawn at that point. `simulate(model, state; max_time)` continues the
run, discards the people built so far and builds them again from the same
stream.

Fields:

- `counts`: the number of people infected on each day, from day 0 (the index
  cases) to `day`.
- `max_generation`: the generation at which cases make no contacts, or
  `nothing` when generations are not tracked.
- `generation_counts`: for each day, the number infected in each generation
  from 0 up to `max_generation`, or up to the day itself where that is
  earlier, because a generation cannot exceed the day of infection (a single
  entry, the count for the day, when generations are not tracked). With several
  types the entries of type 2 follow those of type 1, and so on.
- `susceptibles`, `sizes`: the number of susceptibles left in each type and
  the size of each type, or `nothing` when the population is unlimited.
- `day`: the last day simulated.
- `cumulative_cases`: the total number of cases.
- `extinct`: whether transmission had died out.
"""
mutable struct CountSimulationState{M, L, R} <: AbstractSimulationState
    model::M
    layers::L
    rule::R
    counts::Vector{Int}
    max_generation::Union{Nothing, Int}
    generation_counts::Vector{Vector{Int}}
    cohort_infectiousness::Vector{Vector{Float64}}
    cohort_members::Vector{Vector{Vector{Float64}}}
    inflow::Vector{Array{Int, 3}}
    susceptibles::Union{Nothing, Vector{Int}}
    sizes::Union{Nothing, Vector{Int}}
    day::Int
    cumulative_cases::Int
    extinct::Bool
    rng::AbstractRNG
    individual_stream::Xoshiro
    built::Union{Nothing, SimulationState}
end

function Base.show(io::IO, s::CountSimulationState)
    status = s.extinct ? "extinct" : "active"
    return print(io, "CountSimulationState(cases=$(s.cumulative_cases), day=$(s.day), $(status))")
end

"""
    copy(state::CountSimulationState) -> CountSimulationState

Copy the counts, random streams and realised people. Either state can then be
continued independently. The model, layers and count rule are shared.
Realised people are copied on access, as in [`copy(::SimulationState)`](@ref).
A reference obtained before copying can still change a person in both states;
access the person through the state again after copying.
"""
function Base.copy(s::CountSimulationState)
    built = s.built
    return CountSimulationState(
        s.model, s.layers, s.rule, copy(s.counts), s.max_generation,
        deepcopy(s.generation_counts), deepcopy(s.cohort_infectiousness),
        deepcopy(s.cohort_members), deepcopy(s.inflow), _copy_susceptibles(s.susceptibles),
        s.sizes, s.day, s.cumulative_cases,
        s.extinct, _copy_count_rng(s.rng), copy(s.individual_stream),
        built === nothing ? nothing : copy(built)
    )
end

_copy_susceptibles(::Nothing) = nothing
_copy_susceptibles(s::Vector{Int}) = copy(s)

_copy_count_rng(rng::Random.TaskLocalRNG) = copy(rng)
_copy_count_rng(rng::AbstractRNG) = deepcopy(rng)

# The total infectiousness of a cohort of `n`, and the infectiousness of each
# member where it has to be kept. A point mass or a gamma `ν` needs no members:
# the split among them is independent of the total, so it is drawn when the
# people are built. Any other `ν` gives a split that depends on the total, so
# the members are drawn first and the total is their sum.
_draw_cohort(rule::CountRule, n::Int, rng) =
    n == 0 ? (0.0, Float64[]) : _draw_cohort(rule.ν, n, rng)
_draw_cohort(::Dirac, n, rng) = (Float64(n), Float64[])
_draw_cohort(ν::Gamma, n, rng) = (rand(rng, Gamma(n * shape(ν), scale(ν))), Float64[])
function _draw_cohort(ν, n, rng)
    members = [rand(rng, ν) for _ in 1:n]
    return (sum(members), members)
end
function _draw_cohort(rule::OffspringCountRule, n::Int, rng)
    members = Float64[rand(rng, rule.law) for _ in 1:n]
    return (sum(members), members)
end

# Generation classes are `1:(max_generation + 1)`, or one class without a cap.
_n_classes(max_generation) = max_generation === nothing ? 1 : max_generation + 1
_infects(max_generation, k) = max_generation === nothing || k <= max_generation
_next_class(max_generation, k) = max_generation === nothing ? k : k + 1
# A generation is at least one day after the last, so day `d` holds at most
# `d + 1` classes however large the cap.
_classes_on(max_generation, d) = min(_n_classes(max_generation), d + 1)
# The cohorts of a day are indexed by type and generation class together: the
# `m` classes of type `j` follow those of type `j - 1`.
_cohort_index(m, j, k) = (j - 1) * m + k
_class_of(c, m) = (c - 1) % m + 1
_type_of(c, m) = (c - 1) ÷ m + 1

# Whether the last cohorts that can still infect anyone are all empty.
function _count_exhausted(s::CountSimulationState)
    first_day = max(0, s.day - length(_lags(s.rule)) + 1)
    g = s.max_generation
    for d in (first_day + 1):(s.day + 1), c in eachindex(s.generation_counts[d])
        m = _classes_on(g, d - 1)
        _infects(g, _class_of(c, m)) && s.generation_counts[d][c] > 0 && return false
    end
    return true
end

function _count_extinct(s::CountSimulationState)
    g = s.max_generation
    if g !== nothing
        for d in eachindex(s.generation_counts), j in eachindex(_type_rules(s.rule))
            m = _classes_on(g, d - 1)
            m == g + 1 && s.generation_counts[d][_cohort_index(m, j, g + 1)] > 0 && return false
        end
    end
    return _count_exhausted(s)
end

function _count_step!(s::CountSimulationState)
    rules = _type_rules(s.rule)
    alloc = _alloc(s.rule)
    lags = _lags(s.rule)
    nt = length(rules)
    g = s.max_generation
    t = s.day + 1
    r = [_rate_at(rule, t) for rule in rules]
    m = _classes_on(g, t)
    inflow = zeros(Int, length(lags), nt * _classes_on(g, t - 1), nt)
    by_class = zeros(Int, nt * m)
    sources = Tuple{Int, Int, Int, Int, Float64}[]
    for lag in 1:min(length(lags), t), c in eachindex(s.generation_counts[t - lag + 1])
        ms = _classes_on(g, t - lag)
        k, j = _class_of(c, ms), _type_of(c, ms)
        _infects(g, k) || continue
        s.generation_counts[t - lag + 1][c] == 0 && continue
        push!(sources, (lag, c, j, k, _lag_infections!(s, rules[j], t - lag, c, lag, r[j])))
    end
    if s.susceptibles === nothing
        _free_inflow!(s, first(rules), sources, alloc, inflow, by_class, m)
    else
        _deplete_sources!(s, first(rules), sources, alloc, inflow, by_class, m)
    end
    push!(s.inflow, inflow)
    push!(s.counts, sum(by_class))
    push!(s.generation_counts, by_class)
    _push_cohorts!(s, by_class)
    s.cumulative_cases += sum(by_class)
    s.day = t
    s.extinct = _count_extinct(s)
    return s
end

_rate_at(rule::CountRule, t) = _renewal_R_at(rule.R, t)
_rate_at(::OffspringCountRule, t) = 1.0

# The mean number of people a cohort infects on this lag; for an
# `OffspringCountRule`, the number of its remaining offspring that fall on it.
_lag_infections!(s, rule::CountRule, day, c, lag, r) =
    s.cohort_infectiousness[day + 1][c] * r * rule.g[lag]
function _lag_infections!(s, rule::OffspringCountRule, day, c, lag, r)
    remaining = Int(s.cohort_infectiousness[day + 1][c])
    tail = sum(@view rule.g[lag:end])
    x = remaining == 0 ? 0 : rand(s.rng, Binomial(remaining, min(1.0, rule.g[lag] / tail)))
    s.cohort_infectiousness[day + 1][c] = remaining - x
    return Float64(x)
end

function _free_inflow!(s, ::CountRule, sources, alloc, inflow, by_class, m)
    g = s.max_generation
    for (lag, c, j, k, mean_infections) in sources, i in axes(alloc, 1)
        mu = mean_infections * alloc[i, j]
        x = mu > 0 ? rand(s.rng, Poisson(mu)) : 0
        inflow[lag, c, i] = x
        by_class[_cohort_index(m, i, _next_class(g, k))] += x
    end
    return s
end

function _free_inflow!(s, ::OffspringCountRule, sources, alloc, inflow, by_class, m)
    g = s.max_generation
    for (lag, c, j, k, x) in sources
        x == 0 && continue
        for (i, n) in enumerate(rand(s.rng, Multinomial(Int(x), alloc[:, j])))
            inflow[lag, c, i] = n
            by_class[_cohort_index(m, i, _next_class(g, k))] += n
        end
    end
    return s
end

# Thin the day's contacts against the susceptibles of each type, then share the
# infections among the cohorts in proportion to the contacts each made.
function _deplete_sources!(s, ::CountRule, sources, alloc, inflow, by_class, m)
    g = s.max_generation
    nt = size(alloc, 1)
    rates = zeros(nt)
    for (_, _, j, _, mean_infections) in sources, i in 1:nt
        rates[i] += mean_infections * alloc[i, j]
    end
    kept = _deplete!(s.rng, PerContactDepletion(), rates, s.susceptibles, s.sizes)
    for i in 1:nt
        kept[i] == 0 && continue
        weights = [mean_infections * alloc[i, j] for (_, _, j, _, mean_infections) in sources]
        shares = rand(s.rng, Multinomial(kept[i], weights ./ sum(weights)))
        for (n, (lag, c, _, k, _)) in zip(shares, sources)
            inflow[lag, c, i] = n
            by_class[_cohort_index(m, i, _next_class(g, k))] += n
        end
    end
    return s
end

# For counted contacts the kept ones are shared among sources by hypergeometric draws.
function _deplete_sources!(s, ::OffspringCountRule, sources, alloc, inflow, by_class, m)
    g = s.max_generation
    nt = size(alloc, 1)
    contacts = zeros(Int, length(sources), nt)
    for (n, (_, _, j, _, x)) in enumerate(sources)
        x == 0 && continue
        contacts[n, :] = rand(s.rng, Multinomial(Int(x), alloc[:, j]))
    end
    kept = _deplete!(s.rng, PerContactDepletion(), vec(sum(contacts; dims = 1)), s.susceptibles, s.sizes)
    for i in 1:nt
        remaining, pool = kept[i], sum(@view contacts[:, i])
        for (n, (lag, c, _, k, _)) in enumerate(sources)
            remaining == 0 && break
            x = contacts[n, i]
            take = x == 0 ? 0 : rand(s.rng, Hypergeometric(x, pool - x, remaining))
            pool -= x
            remaining -= take
            inflow[lag, c, i] = take
            by_class[_cohort_index(m, i, _next_class(g, k))] += take
        end
    end
    return s
end

# Draw the infectiousness of each generation's cohort of a new day.
function _push_cohorts!(s::CountSimulationState, by_class)
    rules = _type_rules(s.rule)
    m = length(by_class) ÷ length(rules)
    total = zeros(length(by_class))
    members = [Float64[] for _ in by_class]
    for c in eachindex(by_class)
        _infects(s.max_generation, _class_of(c, m)) || continue
        total[c], members[c] = _draw_cohort(rules[_type_of(c, m)], by_class[c], s.rng)
    end
    push!(s.cohort_infectiousness, total)
    push!(s.cohort_members, members)
    return s
end

function _renewal_R_at(R, t)
    r = R(t)
    r >= 0 || throw(ArgumentError("`R(t)` is negative at t = $t"))
    return r
end

function _count_advance!(s::CountSimulationState, max_time, max_cases = nothing)
    last_day = max_time === nothing ? typemax(Int) : floor(Int, max_time)
    while s.day < last_day && !_count_exhausted(s) &&
            (max_cases === nothing || s.cumulative_cases < max_cases)
        _count_step!(s)
    end
    s.built = nothing
    return s
end

function _simulate_counts(
        model, rule, n_initial::Int, max_time, max_cases, rng,
        max_generations = nothing, index_types = nothing
    )
    stream = Xoshiro(rand(rng, UInt64))
    layers = (
        attributes = attributes(model), progression = progression(model),
        observation = observation(model),
    )
    nt = length(_type_rules(rule))
    by_class = _index_counts(index_types, n_initial, nt, rng)
    sizes = _count_sizes(population(process_of(model)))
    sizes === nothing || all(by_class .<= sizes) || throw(
        ArgumentError("initial cases cannot exceed the population size of any type")
    )
    state = CountSimulationState(
        process_of(model), layers, rule, [n_initial], max_generations, [by_class],
        Vector{Float64}[], Vector{Vector{Float64}}[],
        [zeros(Int, length(_lags(rule)), nt, nt)],
        sizes === nothing ? nothing : sizes .- by_class, sizes,
        0, n_initial, n_initial == 0, rng, stream, nothing
    )
    _push_cohorts!(state, by_class)
    return _count_advance!(state, max_time, max_cases)
end

_index_counts(::Any, n, nt, rng) = nt == 1 ? [n] : rand(rng, Multinomial(n, fill(1 / nt, nt)))
_index_counts(types::AbstractVector, n, nt, rng) = [count(==(j), types) for j in 1:nt]

process_of(model::TransmissionModel) = model
process_of(spec::ModelSpec) = spec.process

# Hand-off from `simulate`: the rule when counts can answer the request.
function _count_route(
        model, initial_cases, index_types, max_cases, max_generations, max_time,
        stopping_rules, condition, rng; default_generations = false
    )
    bounded = if max_time === nothing
        max_cases !== nothing || max_generations !== nothing
    else
        max_time isa Real && isfinite(max_time) && max_time >= 0
    end
    bounded || return nothing
    (
        (default_generations || max_generations != _DEFAULT_MAX_GENERATIONS) &&
            stopping_rules === nothing && condition === nothing &&
            initial_cases === nothing &&
            (index_types === nothing || index_types isa AbstractVector)
    ) || return nothing
    rng isa CommonRandomNumbers && return nothing
    return count_rule(model)
end

"""
    simulate(model, state::CountSimulationState; max_time::Real,
             rng::AbstractRNG = state.rng)

Continue `state`, a [`CountSimulationState`](@ref), up to a later `max_time`.
Every cohort already there keeps the infectiousness it was given, so the
continuation extends the same run. `state` is mutated in place and returned.
The people built from it so far are dropped and built again when next asked
for. `model` must still run on the count-level path.
"""
function simulate(
        model::TransmissionModel, state::CountSimulationState; max_time::Real,
        rng::AbstractRNG = state.rng
    )
    count_rule(model) === nothing && throw(
        ArgumentError(
            "$(typeof(model)) does not run on the count-level path, so it cannot " *
                "continue a CountSimulationState"
        )
    )
    max_time >= state.day || throw(
        ArgumentError(
            "max_time ($max_time) must not be before the state's own day ($(state.day))"
        )
    )
    state.rng = rng
    return _count_advance!(state, max_time)
end

# ── The people ───────────────────────────────────────────────────────

# The people of a count-level run: index cases first, then the cases of each
# day in turn, so the infector of a case always has the smaller id. The
# infections caused by a cohort on a day are shared among its members by their
# own infectiousness (see the top of this file). Every case is then given its
# natural history as in the individual simulation.
function _realise!(s::CountSimulationState)
    built = s.built
    built isa SimulationState && return built
    rng = copy(s.individual_stream)
    layers = s.layers
    sim = new_state(s.model, layers.progression, layers.attributes, rng)
    rules = _type_rules(s.rule)
    nt = length(rules)
    n0 = s.counts[1]
    add_individuals!(
        sim, n0, (); n_types = nt, infection_time = 0,
        types = vcat([fill(j, s.generation_counts[1][j]) for j in 1:nt]...)
    )
    seed!(sim, 1:n0, (), sim.transitions)
    for (i, ind) in enumerate(individuals(sim))
        ind.state[:index] = true
        _settle_case!(sim, ind, (), i)
    end
    g = s.max_generation
    ids = [[Int[] for _ in 1:(nt * _classes_on(g, d))] for d in 0:s.day]
    for j in 1:nt
        append!(ids[1][j], findall(i -> individual_type(i) == j, individuals(sim)))
    end
    shares = [Any[nothing for _ in 1:(nt * _classes_on(g, d))] for d in 0:s.day]
    for d in 1:s.day
        m = _classes_on(g, d)
        for lag in axes(s.inflow[d + 1], 1), c in axes(s.inflow[d + 1], 2), i in 1:nt
            x = s.inflow[d + 1][lag, c, i]
            x == 0 && continue
            src = d - lag
            k, j = _class_of(c, _classes_on(g, src)), _type_of(c, _classes_on(g, src))
            shares[src + 1][c] === nothing &&
                (shares[src + 1][c] = _member_shares(rules[j], s.generation_counts[src + 1][c], s.cohort_members[src + 1][c], rng))
            pool = shares[src + 1][c]
            for _ in 1:x
                member = _pick_member!(pool, rng)
                parent = individuals(sim)[ids[src + 1][c][member]]
                sim.current_generation = parent.generation
                child = make_contact!(
                    sim, parent, d; type_idx = nt > 1 ? i : NoTypeLabels()
                )
                child.state[:infected] = true
                child.state[:index] = false
                _set_onset_from_incubation!(child)
                _settle_case!(sim, child, (), length(individuals(sim)))
                push!(ids[d + 1][_cohort_index(m, i, _next_class(g, k))], length(individuals(sim)))
            end
        end
    end
    _reconcile_sellke_bookkeeping!(sim, s.extinct)
    apply_observation!(layers.observation, sim, rng)
    s.built = sim
    return sim
end

# The cumulative share of the infections of the cohort that each of its `n`
# members causes, in proportion to their infectiousness: equal when every
# member has the same, `Dirichlet(k, …, k)` for a gamma `ν` (the split of a
# `Gamma(n k)` total among `n` independent `Gamma(k)` parts), and given by the
# kept values of `members` otherwise.
function _member_shares(rule::CountRule, n::Int, members, rng)
    weights = if !isempty(members)
        members
    elseif rule.ν isa Dirac
        ones(n)
    else
        [rand(rng, rule.ν) for _ in 1:n]
    end
    cumulative = cumsum(weights)
    cumulative ./= cumulative[end]
    return cumulative
end
# One entry for each offspring of the cohort, naming its member, in random order.
function _member_shares(::OffspringCountRule, n::Int, members, rng)
    return shuffle!(rng, reduce(vcat, [fill(i, Int(c)) for (i, c) in enumerate(members)]; init = Int[]))
end

_pick_member!(cumulative::Vector{Float64}, rng) =
    min(searchsortedfirst(cumulative, rand(rng)), length(cumulative))
_pick_member!(slots::Vector{Int}, rng) = pop!(slots)

individuals(s::CountSimulationState) = individuals(_realise!(s))

# The tables are read from the people. Each takes the people built from the
# counts, so it agrees with `individuals(state)` and with a `SimulationState`.
for f in (
        :linelist, :contacts, :chain_statistics, :weekly_incidence, :incidence,
        :is_extinct, :generation_R, :realised_generation_intervals,
        :realised_serial_intervals, :case_classification_changes,
    )
    @eval $f(state::CountSimulationState, args...; kwargs...) =
        $f(_realise!(state), args...; kwargs...)
end
