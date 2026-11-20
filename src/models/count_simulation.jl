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
transmission and its generation time is a distribution on whole days. Adding a
method for a new model type gives it the count-level path. The rule is a
[`CountRule`](@ref).
"""
count_rule(::TransmissionModel) = nothing

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
struct CountRule{F, D <: UnivariateDistribution}
    R::F
    ν::D
    g::Vector{Float64}
end

# A gamma shape `k` in place of `ν`; infinite `k` is a point mass.
CountRule(R, k::Real, g) = CountRule(R, _gamma_unit(k), g)

# A `ModelSpec` qualifies when the layers it adds do not act on transmission.
function count_rule(spec::ModelSpec)
    isempty(spec.interventions) || return nothing
    spec.case_definition === nothing || return nothing
    spec.recorder isa NoContactRecorder || return nothing
    return count_rule(spec.process)
end

function count_rule(model::BranchingProcess)
    n_types(model) == 1 && length(model.infectiousness) == 1 || return nothing
    isempty(transmission_risks(model)) || return nothing
    population(model) isa NoPopulation || return nothing
    w = only(model.infectiousness)
    w.from === :infection && isempty(w.until) && w.calendar === nothing || return nothing
    return _count_rule(w.offspring, w.kernel)
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
_count_rule(o, kernel) = nothing

# The generation time as a pmf on lags 1, 2, …, when it is a distribution on
# whole days from 1 with a largest value.
function _daily_rule(R, ν, kernel)
    kernel isa DiscreteUnivariateDistribution || return nothing
    lo, hi = minimum(kernel), maximum(kernel)
    lo >= 1 && isfinite(hi) && hi <= 10_000 || return nothing
    g = Float64[pdf(kernel, lag) for lag in 1:Int(hi)]
    isapprox(sum(g), 1; atol = 1.0e-9) || return nothing
    return CountRule(R, ν, g)
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
  from 0 up to `max_generation` (a single entry, the day's count, when
  generations are not tracked).
- `day`: the last day simulated.
- `cumulative_cases`: the total number of cases.
- `extinct`: whether transmission had died out.
"""
mutable struct CountSimulationState{M, L, F} <: AbstractSimulationState
    model::M
    layers::L
    rule::CountRule{F}
    counts::Vector{Int}
    max_generation::Union{Nothing, Int}
    generation_counts::Vector{Vector{Int}}
    cohort_infectiousness::Vector{Vector{Float64}}
    cohort_members::Vector{Vector{Vector{Float64}}}
    inflow::Vector{Matrix{Int}}
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
        deepcopy(s.cohort_members), deepcopy(s.inflow), s.day, s.cumulative_cases,
        s.extinct, _copy_count_rng(s.rng), copy(s.individual_stream),
        built === nothing ? nothing : copy(built)
    )
end

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

# Generation classes are `1:(max_generation + 1)`, or one class without a cap.
_n_classes(max_generation) = max_generation === nothing ? 1 : max_generation + 1
_infects(max_generation, k) = max_generation === nothing || k <= max_generation
_next_class(max_generation, k) = max_generation === nothing ? k : k + 1

# Whether the last cohorts that can still infect anyone are all empty.
function _count_exhausted(s::CountSimulationState)
    first_day = max(0, s.day - length(s.rule.g) + 1)
    g = s.max_generation
    for d in (first_day + 1):(s.day + 1), k in 1:_n_classes(g)
        _infects(g, k) && s.generation_counts[d][k] > 0 && return false
    end
    return true
end

function _count_extinct(s::CountSimulationState)
    g = s.max_generation
    g !== nothing && any(c -> c[g + 1] > 0, s.generation_counts) && return false
    return _count_exhausted(s)
end

function _count_step!(s::CountSimulationState)
    rule = s.rule
    g = s.max_generation
    t = s.day + 1
    r = _renewal_R_at(rule.R, t)
    ncls = _n_classes(g)
    inflow = zeros(Int, length(rule.g), ncls)
    by_class = zeros(Int, ncls)
    for lag in 1:min(length(rule.g), t), k in 1:min(ncls, t)
        _infects(g, k) || continue
        n = s.generation_counts[t - lag + 1][k]
        n == 0 && continue
        mean_infections = s.cohort_infectiousness[t - lag + 1][k] * r * rule.g[lag]
        x = mean_infections > 0 ? rand(s.rng, Poisson(mean_infections)) : 0
        inflow[lag, k] = x
        by_class[_next_class(g, k)] += x
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

# Draw the infectiousness of each generation's cohort of a new day.
function _push_cohorts!(s::CountSimulationState, by_class)
    total = zeros(length(by_class))
    members = [Float64[] for _ in by_class]
    for k in eachindex(by_class)
        _infects(s.max_generation, k) || continue
        total[k], members[k] = _draw_cohort(s.rule, by_class[k], s.rng)
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
        model, rule::CountRule, n_initial::Int, max_time, max_cases, rng,
        max_generations = nothing
    )
    stream = Xoshiro(rand(rng, UInt64))
    layers = (
        attributes = attributes(model), progression = progression(model),
        observation = observation(model),
    )
    ncls = _n_classes(max_generations)
    by_class = zeros(Int, ncls)
    by_class[1] = n_initial
    state = CountSimulationState(
        process_of(model), layers, rule, [n_initial], max_generations, [by_class],
        Vector{Float64}[], Vector{Vector{Float64}}[], [zeros(Int, length(rule.g), ncls)],
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
        max_cases !== nothing
    else
        max_time isa Real && isfinite(max_time) && max_time >= 0
    end
    bounded || return nothing
    (
        (default_generations || max_generations != _DEFAULT_MAX_GENERATIONS) &&
            stopping_rules === nothing && condition === nothing &&
            initial_cases === nothing && index_types === nothing
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
    n0 = s.counts[1]
    add_individuals!(sim, n0, (); infection_time = 0)
    seed!(sim, 1:n0, (), sim.transitions)
    for (i, ind) in enumerate(individuals(sim))
        ind.state[:index] = true
        _settle_case!(sim, ind, (), i)
    end
    g = s.max_generation
    ncls = _n_classes(g)
    ids = [[Int[] for _ in 1:ncls] for _ in 0:s.day]
    append!(ids[1][1], 1:n0)
    shares = [[Float64[] for _ in 1:ncls] for _ in 0:s.day]
    for d in 1:s.day
        for lag in axes(s.inflow[d + 1], 1), k in 1:ncls
            x = s.inflow[d + 1][lag, k]
            x == 0 && continue
            src = d - lag
            isempty(shares[src + 1][k]) &&
                (shares[src + 1][k] = _member_shares(s.rule, s.generation_counts[src + 1][k], s.cohort_members[src + 1][k], rng))
            cumulative = shares[src + 1][k]
            for _ in 1:x
                member = min(searchsortedfirst(cumulative, rand(rng)), length(cumulative))
                parent = individuals(sim)[ids[src + 1][k][member]]
                sim.current_generation = parent.generation
                child = make_contact!(sim, parent, d)
                child.state[:infected] = true
                child.state[:index] = false
                _set_onset_from_incubation!(child)
                _settle_case!(sim, child, (), length(individuals(sim)))
                push!(ids[d + 1][_next_class(g, k)], length(individuals(sim)))
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
