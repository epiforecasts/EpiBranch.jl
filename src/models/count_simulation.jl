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
# lognormal or Poisson Weibull law, whose `ν` is the rate divided by its mean). The cohort infects `Poisson(Λ R(t) g(t - s))` people
# on day `t`, independently for each `t`, with `R(t)` the mean offspring number
# at `t` and `g` the generation time on the daily lags.
#
# The result keeps, for each day, the count of the cohort, its `Λ` and the
# number of infections it received from each earlier cohort. That is enough to
# continue the run and to build the people when they are asked for. The
# infections caused by a cohort are shared among its members in proportion to
# their `ν`, which are `Dirichlet(k, …, k)` given `Λ` for a gamma `ν` (equal
# shares when every `ν` is one). For another `ν` the members' own values are
# drawn at that point, which is the law of the `ν` given their sum.

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
`Dirac(1.0)` when every member is alike, and `Gamma(k, 1/k)` for a negative
binomial offspring law, which gives `Λ` in one draw as `Gamma(n k, 1/k)`. Any
other `ν` takes `n` draws. `R` is a function of the day.
"""
struct CountRule{F, D <: UnivariateDistribution}
    R::F
    ν::D
    g::Vector{Float64}
end

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
- `day`: the last day simulated.
- `cumulative_cases`: the total number of cases.
- `extinct`: whether transmission had died out.
"""
mutable struct CountSimulationState{M, L, F} <: AbstractSimulationState
    model::M
    layers::L
    rule::CountRule{F}
    counts::Vector{Int}
    cohort_infectiousness::Vector{Float64}
    cohort_members::Vector{Vector{Float64}}
    inflow::Vector{Vector{Int}}
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

# The total infectiousness of a cohort of `n`, and the infectiousness of each
# member where it has to be kept. A point mass or a gamma `ν` needs no members:
# the split among them does not depend on the total, so it is drawn when the
# people are built. Any other `ν` gives a split that does, so the members are
# drawn first and the total is their sum.
_draw_cohort(rule::CountRule, n::Int, rng) =
    n == 0 ? (0.0, Float64[]) : _draw_cohort(rule.ν, n, rng)
_draw_cohort(::Dirac, n, rng) = (Float64(n), Float64[])
_draw_cohort(ν::Gamma, n, rng) = (rand(rng, Gamma(n * shape(ν), scale(ν))), Float64[])
function _draw_cohort(ν, n, rng)
    members = [rand(rng, ν) for _ in 1:n]
    return (sum(members), members)
end

# Whether the last cohorts that can still infect anyone are all empty.
function _count_extinct(s::CountSimulationState)
    first_day = max(0, s.day - length(s.rule.g) + 1)
    return all(iszero, @view s.counts[(first_day + 1):(s.day + 1)])
end

function _count_step!(s::CountSimulationState)
    rule = s.rule
    t = s.day + 1
    r = _renewal_R_at(rule.R, t)
    inflow = zeros(Int, length(rule.g))
    for lag in 1:min(length(rule.g), t)
        n = s.counts[t - lag + 1]
        n == 0 && continue
        mean_infections = s.cohort_infectiousness[t - lag + 1] * r * rule.g[lag]
        inflow[lag] = mean_infections > 0 ? rand(s.rng, Poisson(mean_infections)) : 0
    end
    n = sum(inflow)
    push!(s.inflow, inflow)
    push!(s.counts, n)
    total, members = _draw_cohort(rule, n, s.rng)
    push!(s.cohort_infectiousness, total)
    push!(s.cohort_members, members)
    s.cumulative_cases += n
    s.day = t
    s.extinct = _count_extinct(s)
    return s
end

function _renewal_R_at(R, t)
    r = R(t)
    r >= 0 || throw(ArgumentError("`R(t)` is negative at t = $t"))
    return r
end

function _count_advance!(s::CountSimulationState, max_time::Real)
    last_day = floor(Int, max_time)
    while s.day < last_day && !s.extinct
        _count_step!(s)
    end
    s.built = nothing
    return s
end

function _simulate_counts(model, rule::CountRule, n_initial::Int, max_time::Real, rng)
    stream = Xoshiro(rand(rng, UInt64))
    layers = (
        attributes = attributes(model), progression = progression(model),
        observation = observation(model),
    )
    total, members = _draw_cohort(rule, n_initial, rng)
    state = CountSimulationState(
        process_of(model), layers, rule, [n_initial], [total], [members],
        [zeros(Int, length(rule.g))],
        0, n_initial, n_initial == 0, rng, stream, nothing
    )
    return _count_advance!(state, max_time)
end

process_of(model::TransmissionModel) = model
process_of(spec::ModelSpec) = spec.process

# Hand-off from `simulate`: the rule when counts can answer the request. A run
# that stops on anything but a time, or that asks for a particular index case,
# a repeated run or a shared random stream, is simulated as individuals,
# because those are decided per person.
function _count_route(
        model, initial_cases, index_types, max_cases, max_generations, max_time,
        stopping_rules, condition, rng
    )
    max_time isa Real && isfinite(max_time) && max_time >= 0 || return nothing
    (
        max_cases === nothing && max_generations === nothing &&
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
    first_id = Int[1]
    shares = Vector{Float64}[Float64[]]
    for d in 1:s.day
        push!(first_id, length(individuals(sim)) + 1)
        push!(shares, Float64[])
        for lag in eachindex(s.inflow[d + 1])
            x = s.inflow[d + 1][lag]
            x == 0 && continue
            src = d - lag
            isempty(shares[src + 1]) &&
                (shares[src + 1] = _member_shares(s.rule, s.counts[src + 1], s.cohort_members[src + 1], rng))
            cumulative = shares[src + 1]
            for _ in 1:x
                member = min(searchsortedfirst(cumulative, rand(rng)), length(cumulative))
                parent = individuals(sim)[first_id[src + 1] + member - 1]
                sim.current_generation = parent.generation
                child = make_contact!(sim, parent, d)
                child.state[:infected] = true
                child.state[:index] = false
                _set_onset_from_incubation!(child)
                _settle_case!(sim, child, (), length(individuals(sim)))
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
# `Gamma(n k)` total among `n` independent `Gamma(k)` parts), and the kept
# values of `members` otherwise.
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
