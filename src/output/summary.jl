"""
    containment_probability(states::Vector{<:SimulationState}; max_cases=nothing)

Fraction of simulations that went extinct (i.e. the outbreak was contained).

If `max_cases` is provided, simulations that hit the case cap are not
considered extinct (they are assumed to have continued growing).
"""
function containment_probability(states::Vector{<:SimulationState};
        max_cases::Union{Int, NoCases} = NoCases())
    n_extinct = count(s -> _is_contained(s, max_cases), states)
    return n_extinct / length(states)
end

_is_contained(s::SimulationState, ::NoCases) = s.extinct
_is_contained(s::SimulationState, cap::Int) = s.cumulative_cases < cap && s.extinct

"""Whether a simulation has exceeded the max_cases cap (false if no cap)."""
_check_max_cases(::SimulationState, ::NoCases) = false
_check_max_cases(state::SimulationState, cap::Int) = state.cumulative_cases >= cap

"""
    is_extinct(state::SimulationState; by_week=nothing, reference_date=Date(2020,1,1),
               max_cases=nothing)

Extinction classification for a single simulation with optional criteria.

- No keyword args: returns `state.extinct`
- `by_week::Int`: extinct if no cases with onset in that week
- `by_week::UnitRange{Int}`: extinct if no cases with onset in that week range
- `max_cases::Int`: outbreaks hitting this cap are not considered extinct
"""
function is_extinct(state::SimulationState;
        by_week::Union{Int, UnitRange{Int}, Nothing} = nothing,
        reference_date::Date = Date(2020, 1, 1),
        max_cases::Union{Int, NoCases} = NoCases())
    _check_max_cases(state, max_cases) && return false

    by_week === nothing && return state.extinct

    # Week-based extinction
    weeks = by_week isa Int ? (by_week:by_week) : by_week
    for ind in state.individuals
        !is_infected(ind) && continue
        isnan(ind.infection_time) && continue
        day = floor(Int, ind.infection_time)
        week_num = div(day, 7) + 1
        week_num in weeks && return false
    end
    return true
end

"""
    generation_R(state::SimulationState)

Realised per-generation offspring ratio: for each generation `g`,
the number of cases in generation `g+1` divided by the number of cases
in generation `g`. A DataFrame with columns `generation` and
`offspring_ratio` is returned.

This is not the time-varying effective reproduction number `Rt`
typically estimated from an incidence time series — it is a
generation-indexed average that only coincides with `Rt` under
strong assumptions. Use it as a within-simulation diagnostic of
how transmission is being reduced generation-by-generation
(e.g. by depletion or by interventions), not as an `Rt` proxy.
"""
function generation_R(state::SimulationState)
    # Single-pass: count infected individuals per generation
    gen_counts = Dict{Int, Int}()
    for ind in state.individuals
        is_infected(ind) || continue
        g = ind.generation
        gen_counts[g] = get(gen_counts, g, 0) + 1
    end

    isempty(gen_counts) && return DataFrame(
        generation = Int[], offspring_ratio = Float64[])
    max_gen = maximum(keys(gen_counts))

    generations = Int[]
    ratios = Float64[]
    for g in 0:(max_gen - 1)
        n_parents = get(gen_counts, g, 0)
        n_parents == 0 && continue
        n_children = get(gen_counts, g + 1, 0)
        push!(generations, g)
        push!(ratios, n_children / n_parents)
    end

    DataFrame(generation = generations, offspring_ratio = ratios)
end

"""
    weekly_incidence(state::SimulationState; by=:onset,
                     reference_date::Date=Date(2020, 1, 1))

Compute weekly case counts from a single simulation.
A DataFrame with columns `week` (Date) and `cases` (Int) is returned.

`by` selects the timing field to bin on:

- `:onset` (default) — uses `:onset_time` from individual state, which is
  what a surveillance epicurve plots. Falls back to `:infection_time`
  for any case whose `:onset_time` is missing or `NaN` (e.g.
  asymptomatic cases under `clinical_presentation`).
- `:infection` — uses `infection_time` directly. This is what was
  previously the only behaviour, but it is not directly observable in
  real surveillance and produces an epicurve that is shifted earlier
  by roughly one incubation period.
- `:reporting` — uses `:reporting_time` (set by the `Reporting`
  transition or `PerCaseObservation`); cases without a reporting time
  are excluded.

Pass a `Symbol` from `ind.state` to bin on any other field
(e.g. `:admission_time`).
"""
function weekly_incidence(state::SimulationState;
        by::Symbol = :onset,
        reference_date::Date = Date(2020, 1, 1))
    infected = filter(is_infected, state.individuals)
    isempty(infected) && return DataFrame(week = Date[], cases = Int[])

    times = Float64[]
    for ind in infected
        t = _weekly_time(by, ind)
        isnan(t) && continue
        push!(times, t)
    end
    isempty(times) && return DataFrame(week = Date[], cases = Int[])

    weeks = Dict{Date, Int}()
    for t in times
        date = reference_date + Day(floor(Int, t))
        week_start = date - Day(dayofweek(date) - 1)
        weeks[week_start] = get(weeks, week_start, 0) + 1
    end

    df = DataFrame(week = collect(keys(weeks)), cases = collect(values(weeks)))
    sort!(df, :week)
    return df
end

# Onset with infection-time fallback for cases without a recorded onset.
function _weekly_time(by::Symbol, ind)
    if by === :onset
        v = get(ind.state, :onset_time, NaN)
        return v isa Real && !isnan(v) ? float(v) : ind.infection_time
    elseif by === :infection
        return ind.infection_time
    elseif by === :reporting
        # No fallback: a case without a reporting time has not been observed.
        v = get(ind.state, :reporting_time, NaN)
        return v isa Real ? float(v) : NaN
    else
        v = get(ind.state, by, NaN)
        return v isa Real ? float(v) : NaN
    end
end

"""
    scenario_sweep(params::Dict{Symbol, Vector}; n_sim=500, rng=Random.default_rng(), sim_kwargs...)

Run a parameter sweep over all combinations of parameters and return a
DataFrame of results. Each row contains the parameter values and the
containment probability.

`params` must include `:offspring` (vector of offspring distributions) and
may include `:generation_time`, `:interventions`, `:attributes`, and any
other named parameters. `:interventions` values should be vectors of
intervention stacks (each element is a `Vector{<:AbstractIntervention}`).

```julia
results = scenario_sweep(Dict(
    :offspring => [NegBin(2.5, 0.16), NegBin(1.5, 0.5)],
    :interventions => [[Isolation(onset_to_isolation_delay=Exponential(d))] for d in [1.0, 2.0, 5.0]],
    :generation_time => [LogNormal(1.6, 0.5)],
))
```
"""
function scenario_sweep(params::Dict{Symbol, <:AbstractVector};
        n_sim::Int = 500,
        rng::AbstractRNG = Random.default_rng(),
        sim_kwargs...)
    haskey(params, :offspring) || throw(ArgumentError("params must include :offspring"))

    keys_ordered = collect(keys(params))
    value_lists = [params[k] for k in keys_ordered]
    combinations = Iterators.product(value_lists...)

    rows = Dict{Symbol, Vector{Any}}(k => Any[] for k in keys_ordered)
    rows[:containment_probability] = Any[]

    for combo in combinations
        vals = Dict(k => v for (k, v) in zip(keys_ordered, combo))

        offspring = vals[:offspring]
        gt = get(vals, :generation_time, NoGenerationTime())
        interventions = get(vals, :interventions, AbstractIntervention[])
        attributes = get(vals, :attributes, NoAttributes())
        pop_size = get(vals, :population_size, NoPopulation())

        model = ModelSpec(BranchingProcess(offspring, gt; population_size = pop_size);
            interventions, attributes)

        results = simulate(model, n_sim; rng = rng, sim_kwargs...)

        for k in keys_ordered
            push!(rows[k], vals[k])
        end
        push!(rows[:containment_probability], containment_probability(results))
    end

    DataFrame(rows)
end
