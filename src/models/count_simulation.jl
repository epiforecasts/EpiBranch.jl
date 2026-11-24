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
transmission, apart from the depletion of a finite population by type and the
vaccination campaigns, which are applied to the counts as doses, and its
generation time is a distribution on whole days. A
[`BranchingProcess`](@ref) may have several infectiousness windows or several
types, but not a `multiplier` or an `immune` population, which run as
individuals. A method for a new model type gives it the count-level path. The rule is
a [`CountRule`](@ref), or a [`CountRules`](@ref) for several windows or types.
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

The count-level rule ([`EpiBranch.count_rule`](@ref)) for an offspring `law`
given by a discrete distribution on the non-negative integers that is not a
Poisson mixture, such as `Binomial` or `Geometric`. Each cohort member draws its
number of offspring from `law` when infected, and each offspring falls on lag
`l` with probability `g[l]`.
"""
struct OffspringCountRule{D <: DiscreteUnivariateDistribution} <: AbstractCountRule
    law::D
    g::Vector{Float64}
end

"""
    CountRules(channels, allocation)

The rule of a model with several infectiousness windows or several types,
returned by [`EpiBranch.count_rule`](@ref) in place of a single rule.
`channels[w, j]` is the [`CountRule`](@ref) or [`OffspringCountRule`](@ref) by
which a type-`j` case infects through window `w`, and `allocation[w][i, j]` the
share of those infections that are of type `i`. Each case draws its
infectiousness for each window independently. A cohort of one type therefore
holds one total for each window. The channels are all of one kind. Only a
model with one window has counted channels. `campaigns` are the vaccination
campaigns of a [`ModelSpec`](@ref) on a finite population: the doses of
`[t - 1, t)` reach the susceptibles before day `t` is drawn, as
[`simulate_by_generation`](@ref) gives them between generations.
"""
struct CountRules{C <: AbstractMatrix{<:AbstractCountRule}, V <: AbstractVector}
    channels::C
    allocation::Vector{Matrix{Float64}}
    campaigns::V
end

CountRules(channels, allocation) = CountRules(channels, allocation, [])

_as_rules(rule::AbstractCountRule) = CountRules(fill(rule, 1, 1), [ones(1, 1)])
_as_rules(rules::CountRules) = rules
_max_lag(rules::CountRules) = maximum(c -> length(c.g), rules.channels)

# A `ModelSpec` qualifies when the layers it adds do not act on transmission,
# apart from vaccination campaigns on a finite population.
function count_rule(spec::ModelSpec)
    spec.case_definition === nothing || return nothing
    spec.recorder isa NoContactRecorder || return nothing
    rule = count_rule(spec.process)
    rule === nothing && return nothing
    isempty(spec.interventions) && return rule
    _count_sizes(population(spec.process)) === nothing && return nothing
    # Eligibility depends only on how the interventions are read. A budget that
    # does not fit the population is a configuration error and raises, as in
    # `simulate_by_generation`.
    campaigns = try
        [_campaign(iv, (0.0, Inf), nothing) for iv in spec.interventions]
    catch err
        err isa ArgumentError && return nothing
        rethrow()
    end
    campaigns = [_by_type(c, population(spec.process)) for c in campaigns]
    rules = _as_rules(rule)
    return CountRules(rules.channels, rules.allocation, campaigns)
end

function count_rule(model::BranchingProcess)
    isempty(transmission_risks(model)) || return nothing
    model.multiplier === nothing || return nothing
    model.importations === nothing || return nothing
    _immune_total(population(model)) == 0 || return nothing
    _counts_population(population(model), n_types(model)) || return nothing
    windows = model.infectiousness
    all(w -> w.from === :infection && isempty(w.until) && w.calendar === nothing, windows) ||
        return nothing
    n = n_types(model)
    if n == 1 && length(windows) == 1
        return _count_rule(only(windows).offspring, only(windows).kernel)
    end
    channels = [_window_channels(w.offspring, w.kernel, n) for w in windows]
    any(isnothing, channels) && return nothing
    matrix = reduce(vcat, first.(channels))
    length(windows) == 1 || all(c -> c isa CountRule, matrix) || return nothing
    return CountRules(matrix, last.(channels))
end

_counts_population(::NoPopulation, nt) = true
_counts_population(::SinglePopulation, nt) = nt == 1
_counts_population(p::GroupedPopulation, nt) = length(p.sizes) == nt

_count_sizes(::NoPopulation) = nothing
_count_sizes(p::SinglePopulation) = [p.size]
_count_sizes(p::GroupedPopulation) = copy(p.sizes)

# The row of channels of one window and its allocation between types.
function _window_channels(o, kernel, n::Int)
    n == 1 || return nothing
    rule = _count_rule(o, kernel)
    rule isa AbstractCountRule || return nothing
    return (fill(rule, 1, 1), ones(1, 1))
end
function _window_channels(o::MultiTypeOffspring, kernel, n::Int)
    _n_types(o) == n || return nothing
    rules = map(1:n) do j
        R = o.R_by_type[j]
        R <= 0 && return missing
        return _count_rule(_type_law(o.dist_fn, R, j), kernel)
    end
    live = collect(skipmissing(rules))
    any(isnothing, live) && return nothing
    kind = isempty(live) ? CountRule : typeof(first(live)).name.wrapper
    all(r -> r isa kind, live) || return nothing
    # A type that infects no one has no offspring law to read.
    sink = kind === CountRule ? _daily_rule(_ConstantR(0.0), Dirac(1.0), kernel) :
        OffspringCountRule(Binomial(1, 0.0), first(live).g)
    sink === nothing && return nothing
    return (
        permutedims(kind[ismissing(r) ? sink : r for r in rules]),
        Matrix{Float64}(o.alloc_probs),
    )
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
- `generation_counts`: for each day, a matrix of the number infected of each
  type (rows) in each generation (columns) from 0 up to `max_generation`, or
  up to the day itself where that is earlier, because a generation cannot
  exceed the day of infection (a single column, the count for the day, when
  generations are not tracked).
- `cohort_infectiousness` and `cohort_members`: for each day, the total
  infectiousness of each cohort, and the infectiousness of each member where
  it is kept, indexed by window, type and generation.
- `inflow`: for each day, the number infected by each cohort, indexed by lag,
  window, type and generation of the infector, and type of the infected.
- `susceptibles`, `sizes`: the number of susceptibles left in each type and
  the size of each type, or `nothing` when the population is unlimited.
- `day`: the last day simulated.
- `cumulative_cases`: the total number of cases.
- `extinct`: whether transmission had died out.
"""
mutable struct CountSimulationState{M, L, R <: CountRules} <: AbstractSimulationState
    model::M
    layers::L
    rule::R
    counts::Vector{Int}
    max_generation::Union{Nothing, Int}
    generation_counts::Vector{Matrix{Int}}
    cohort_infectiousness::Vector{Array{Float64, 3}}
    cohort_members::Vector{Array{Vector{Float64}, 3}}
    inflow::Vector{Array{Int, 5}}
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
# Whether the last cohorts that can still infect anyone are all empty.
function _count_exhausted(s::CountSimulationState)
    first_day = max(0, s.day - _max_lag(s.rule) + 1)
    g = s.max_generation
    for d in (first_day + 1):(s.day + 1), k in axes(s.generation_counts[d], 2)
        _infects(g, k) && any(>(0), view(s.generation_counts[d], :, k)) && return false
    end
    return true
end

function _count_extinct(s::CountSimulationState)
    g = s.max_generation
    g !== nothing &&
        any(c -> size(c, 2) > g && any(>(0), view(c, :, g + 1)), s.generation_counts) &&
        return false
    return _count_exhausted(s)
end

_count_R_at(rule::CountRule, t) = _renewal_R_at(rule.R, t)
_count_R_at(::OffspringCountRule, t) = 1.0

function _count_step!(s::CountSimulationState)
    rules = s.rule
    channels = rules.channels
    n_windows, n_types = size(channels)
    g = s.max_generation
    t = s.day + 1
    _vaccinate_counts!(s, t)
    rates = [_count_R_at(c, t) for c in channels]
    ncls = _classes_on(g, t)
    max_lag = _max_lag(rules)
    inflow = zeros(Int, max_lag, n_windows, n_types, ncls, n_types)
    by_class = zeros(Int, n_types, ncls)
    sources = Tuple{Int, Int, Int, Int, Float64}[]
    for lag in 1:min(max_lag, t), k in axes(s.generation_counts[t - lag + 1], 2), j in 1:n_types
        _infects(g, k) || continue
        n = s.generation_counts[t - lag + 1][j, k]
        n == 0 && continue
        for w in 1:n_windows
            lag <= length(channels[w, j].g) || continue
            mean_infections = _lag_infections!(
                s, channels[w, j], t - lag, CartesianIndex(w, j, k), lag, rates[w, j]
            )
            mean_infections > 0 && push!(sources, (lag, w, j, k, mean_infections))
        end
    end
    if s.susceptibles === nothing
        _free_inflow!(s, first(channels), sources, inflow, by_class)
    else
        _deplete_sources!(s, first(channels), sources, inflow, by_class)
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

# The mean number of people a cohort infects on this lag; for an
# `OffspringCountRule`, the number of its remaining offspring that fall on it.
_lag_infections!(s, rule::CountRule, day, idx, lag, r) =
    s.cohort_infectiousness[day + 1][idx] * r * rule.g[lag]
function _lag_infections!(s, rule::OffspringCountRule, day, idx, lag, r)
    remaining = round(Int, s.cohort_infectiousness[day + 1][idx])
    tail = sum(@view rule.g[lag:end])
    x = remaining == 0 ? 0 : rand(s.rng, Binomial(remaining, min(1.0, rule.g[lag] / tail)))
    s.cohort_infectiousness[day + 1][idx] = remaining - x
    return Float64(x)
end

function _free_inflow!(s, ::CountRule, sources, inflow, by_class)
    g = s.max_generation
    allocation = s.rule.allocation
    for (lag, w, j, k, mean_infections) in sources, i in axes(allocation[w], 1)
        mu = mean_infections * allocation[w][i, j]
        x = mu > 0 ? rand(s.rng, Poisson(mu)) : 0
        inflow[lag, w, j, k, i] = x
        by_class[i, _next_class(g, k)] += x
    end
    return s
end

function _free_inflow!(s, ::OffspringCountRule, sources, inflow, by_class)
    g = s.max_generation
    allocation = s.rule.allocation
    for (lag, w, j, k, x) in sources
        for (i, n) in enumerate(rand(s.rng, Multinomial(Int(x), allocation[w][:, j])))
            inflow[lag, w, j, k, i] = n
            by_class[i, _next_class(g, k)] += n
        end
    end
    return s
end

# The doses of `[t0, t1)` given to each type. A shared budget is a pool of doses
# for everyone: each dose goes to a type in proportion to its population, as it
# would to a person picked at random. A budget per group gives each type its own.
function _doses_by_type(rng::AbstractRNG, c::_Campaign, budgets::AbstractVector, t0, t1, sizes)
    return [_interval_doses(c, t0, t1, b) for b in budgets]
end
function _doses_by_type(rng::AbstractRNG, c::_Campaign, ::CapacityConstrained, t0, t1, sizes)
    doses = _interval_doses(c, t0, t1)
    doses == 0 && return zeros(Int, length(sizes))
    length(sizes) == 1 && return [doses]
    return rand(rng, Multinomial(doses, Float64.(sizes) ./ sum(sizes)))
end

# Give a campaign's doses for `[t0, t1)` to the susceptibles. A dose reaches a
# susceptible of a type with probability equal to the share of susceptibles in
# the population of the type at the start of the interval, and protects with
# probability `efficacy`. With no doses nothing is drawn.
function _vaccinate!(rng::AbstractRNG, c::_Campaign, t0, t1, susceptibles, sizes)
    doses = _doses_by_type(rng, c, c.budget, t0, t1, sizes)
    for i in eachindex(susceptibles)
        doses[i] == 0 && continue
        (susceptibles[i] <= 0 || sizes[i] <= 0) && continue
        reached = min(rand(rng, Binomial(doses[i], susceptibles[i] / sizes[i])), susceptibles[i])
        susceptibles[i] -= rand(rng, Binomial(reached, c.efficacy))
    end
    return susceptibles
end

# The doses of `[t - 1, t)` reach the susceptibles before day `t` is drawn.
function _vaccinate_counts!(s::CountSimulationState, t)
    for c in s.rule.campaigns
        _vaccinate!(s.rng, c, t - 1, t, s.susceptibles, s.sizes)
    end
    return s
end

# Thin the day's contacts against the susceptibles of each type, then share the
# infections among the cohorts in proportion to the contacts each made.
function _deplete_sources!(s, ::CountRule, sources, inflow, by_class)
    g = s.max_generation
    allocation = s.rule.allocation
    nt = size(s.rule.channels, 2)
    rates = zeros(nt)
    for (_, w, j, _, mean_infections) in sources, i in 1:nt
        rates[i] += mean_infections * allocation[w][i, j]
    end
    kept = _deplete!(s.rng, PerContactDepletion(), rates, s.susceptibles, s.sizes)
    for i in 1:nt
        kept[i] == 0 && continue
        weights = [mean_infections * allocation[w][i, j] for (_, w, j, _, mean_infections) in sources]
        shares = rand(s.rng, Multinomial(kept[i], weights ./ sum(weights)))
        for (n, (lag, w, j, k, _)) in zip(shares, sources)
            inflow[lag, w, j, k, i] = n
            by_class[i, _next_class(g, k)] += n
        end
    end
    return s
end

# For counted contacts the kept ones are shared among sources by hypergeometric draws.
function _deplete_sources!(s, ::OffspringCountRule, sources, inflow, by_class)
    g = s.max_generation
    allocation = s.rule.allocation
    nt = size(s.rule.channels, 2)
    contacts = zeros(Int, length(sources), nt)
    for (n, (_, w, j, _, x)) in enumerate(sources)
        contacts[n, :] = rand(s.rng, Multinomial(Int(x), allocation[w][:, j]))
    end
    kept = _deplete!(s.rng, PerContactDepletion(), vec(sum(contacts; dims = 1)), s.susceptibles, s.sizes)
    for i in 1:nt
        taken = _rand_multivariate_hypergeometric(s.rng, @view(contacts[:, i]), kept[i])
        for (n, (lag, w, j, k, _)) in enumerate(sources)
            inflow[lag, w, j, k, i] = taken[n]
            by_class[i, _next_class(g, k)] += taken[n]
        end
    end
    return s
end

# Draw the infectiousness of each cohort of a new day, one for each window.
function _push_cohorts!(s::CountSimulationState, by_class)
    channels = s.rule.channels
    n_windows, n_types = size(channels)
    total = zeros(n_windows, n_types, size(by_class, 2))
    members = [Float64[] for _ in total]
    for k in axes(by_class, 2), j in 1:n_types
        _infects(s.max_generation, k) || continue
        for w in 1:n_windows
            total[w, j, k], members[w, j, k] =
                _draw_cohort(channels[w, j], by_class[j, k], s.rng)
        end
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
    rules = _as_rules(rule)
    n_types = size(rules.channels, 2)
    stream = Xoshiro(rand(rng, UInt64))
    layers = (
        attributes = attributes(model), progression = progression(model),
        observation = observation(model),
    )
    ncls = _classes_on(max_generations, 0)
    by_class = zeros(Int, n_types, ncls)
    for i in 1:n_initial
        by_class[n_types == 1 ? 1 : _draw_type(rng, index_types, i, n_types), 1] += 1
    end
    sizes = _count_sizes(population(process_of(model)))
    sizes === nothing || all(sum(by_class; dims = 2) .<= sizes) || throw(
        ArgumentError("initial cases cannot exceed the population size of any type")
    )
    state = CountSimulationState(
        process_of(model), layers, rules, [n_initial], max_generations, [by_class],
        Array{Float64, 3}[], Array{Vector{Float64}, 3}[],
        [zeros(Int, _max_lag(rules), size(rules.channels, 1), n_types, ncls, n_types)],
        sizes === nothing ? nothing : sizes .- vec(sum(by_class; dims = 2)), sizes,
        0, n_initial, n_initial == 0, rng, stream, nothing
    )
    _push_cohorts!(state, by_class)
    return _count_advance!(state, max_time, max_cases)
end

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
            initial_cases === nothing
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
    channels = s.rule.channels
    n_windows, n_types = size(channels)
    n0 = s.counts[1]
    index_types = reduce(vcat, (fill(j, s.generation_counts[1][j, 1]) for j in 1:n_types))
    add_individuals!(sim, n0, (); n_types, types = index_types, infection_time = 0)
    seed!(sim, 1:n0, (), sim.transitions)
    for (i, ind) in enumerate(individuals(sim))
        ind.state[:index] = true
        _settle_case!(sim, ind, (), i)
    end
    g = s.max_generation
    ids = [[Int[] for _ in 1:n_types, _ in 1:_classes_on(g, d)] for d in 0:s.day]
    for (id, ind) in enumerate(individuals(sim))
        push!(ids[1][individual_type(ind), 1], id)
    end
    pools = [
        Array{Any, 3}(nothing, n_windows, n_types, _classes_on(g, d))
            for d in 0:s.day
    ]
    for d in 1:s.day
        inflow = s.inflow[d + 1]
        for lag in axes(inflow, 1), k in axes(inflow, 4), j in 1:n_types, w in 1:n_windows, i in 1:n_types
            src = d - lag
            src < 0 && continue
            k <= size(s.generation_counts[src + 1], 2) || continue
            x = inflow[lag, w, j, k, i]
            x == 0 && continue
            if pools[src + 1][w, j, k] === nothing
                pools[src + 1][w, j, k] = _member_pool(
                    channels[w, j], s.generation_counts[src + 1][j, k],
                    s.cohort_members[src + 1][w, j, k], rng
                )
            end
            pool = pools[src + 1][w, j, k]
            for _ in 1:x
                member = _next_member!(pool, rng)
                parent = individuals(sim)[ids[src + 1][j, k][member]]
                sim.current_generation = parent.generation
                child = n_types == 1 ? make_contact!(sim, parent, d) :
                    make_contact!(sim, parent, d; type_idx = i)
                child.state[:infected] = true
                child.state[:index] = false
                _set_onset_from_incubation!(child)
                _settle_case!(sim, child, (), length(individuals(sim)))
                push!(ids[d + 1][i, _next_class(g, k)], length(individuals(sim)))
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
function _member_pool(rule::CountRule, n::Int, members, rng)
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
_next_member!(cumulative::Vector{Float64}, rng) =
    min(searchsortedfirst(cumulative, rand(rng)), length(cumulative))

# The offspring of a cohort, labelled by parent member, shuffled and taken in turn.
mutable struct _OffspringPool
    members::Vector{Int}
    next::Int
end
function _member_pool(::OffspringCountRule, n::Int, members, rng)
    each = Vector{Int}(undef, sum(c -> round(Int, c), members; init = 0))
    pos = 0
    for (i, c) in enumerate(members)
        k = round(Int, c)
        fill!(view(each, (pos + 1):(pos + k)), i)
        pos += k
    end
    return _OffspringPool(shuffle(rng, each), 0)
end
function _next_member!(pool::_OffspringPool, rng)
    pool.next += 1
    return pool.members[pool.next]
end

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
