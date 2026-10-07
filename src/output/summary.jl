"""
    containment_probability(states::Vector{<:SimulationState}; max_cases=nothing)

Containment probability: the proportion of simulated outbreaks that died out,
from a vector of simulations such as `simulate(model, n)`. An outbreak still
going when the simulation stopped (at `max_cases`, `max_time` or
`max_generations`) counts as not contained.

With `max_cases`, an outbreak that reached that many cases also counts as not
contained, even if it then died out. Pass the cap used in `simulate`, so that
outbreaks large enough to reach it count as uncontrolled, as in ringbp.

This is the simulation estimate. [`probability_contain`](@ref) gives the
closed-form value for a model without interventions, and
[`extinction_probability`](@ref) the probability that transmission dies out
from a single case.

# Examples

```julia
model = ModelSpec(
    BranchingProcess(NegBin(2.5, 0.16), Gamma(2.0, 3.0));
    attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
    interventions = [Isolation(onset_to_isolation_delay = Exponential(2.0),
        isolation_duration = Inf)]
)
states = simulate(model, 1000; max_cases = 5000)
containment_probability(states; max_cases = 5000)
```
"""
function containment_probability(
        states::Vector{<:SimulationState};
        max_cases::Union{Int, NoCases} = NoCases()
    )
    n_extinct = count(s -> _is_contained(s, max_cases), states)
    return n_extinct / length(states)
end

_is_contained(s::SimulationState, ::NoCases) = s.extinct
_is_contained(s::SimulationState, cap::Int) = s.cumulative_cases < cap && s.extinct

"""Whether a simulation has exceeded the max_cases cap (false if no cap)."""
_check_max_cases(::SimulationState, ::NoCases) = false
_check_max_cases(state::SimulationState, cap::Int) = state.cumulative_cases >= cap

"""
    is_extinct(state::SimulationState; by_week=nothing, max_cases=nothing)

Whether one simulated outbreak died out.

- With no keywords: `true` if the outbreak ended with no cases left to infect
  anyone before the simulation stopped.
- `by_week = 12`: `true` if no case has symptom onset in week 12;
  `by_week = 12:16`: none in weeks 12 to 16.
- `max_cases`: an outbreak that reached this many cases counts as not extinct.

Weeks are 7-day blocks counted from day 0 of the simulation (week 1 is days 0
to 6). A case without an onset time, such as an asymptomatic case, is placed
by its infection time instead, as in [`weekly_incidence`](@ref).

# Examples

```julia
model = ModelSpec(
    BranchingProcess(NegBin(1.2, 0.5), Gamma(2.0, 3.0));
    attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5))
)
states = simulate(model, 100; max_time = 140.0)
mean(is_extinct.(states; by_week = 12:20))   # share with no onsets in weeks 12-20
```
"""
function is_extinct(
        state::SimulationState;
        by_week::Union{Int, UnitRange{Int}, Nothing} = nothing,
        max_cases::Union{Int, NoCases} = NoCases()
    )
    _check_max_cases(state, max_cases) && return false

    by_week === nothing && return state.extinct

    # Week-based extinction, binned on onset (with infection fallback) to match
    # the docstring and `weekly_incidence`.
    weeks = by_week isa Int ? (by_week:by_week) : by_week
    for ind in state.individuals
        !is_infected(ind) && continue
        t = _weekly_time(:onset, ind)
        isfinite(t) || continue
        week_num = div(floor(Int, t), 7) + 1
        week_num in weeks && return false
    end
    return true
end

"""
    generation_R(state::SimulationState)

Not Rt: the ratio of the number of cases in each generation to the number in
the generation before, in one simulated outbreak. Returns a DataFrame with
columns `generation` (`g`) and `offspring_ratio` (cases in generation `g + 1`
divided by cases in generation `g`).

This differs from the time-varying reproduction number `Rt` estimated from an
incidence time series; the two coincide only under strong assumptions. Use it
to see how transmission falls from one generation to the next in a
simulation, for example through depletion of susceptibles or interventions.
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
        generation = Int[], offspring_ratio = Float64[]
    )
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

    return DataFrame(generation = generations, offspring_ratio = ratios)
end

"""
    weekly_incidence(state::SimulationState; by=:onset,
                     reference_date::Date=Date(2020, 1, 1))

Weekly case counts from one simulated outbreak, the epidemic curve a
surveillance system would plot. Returns a DataFrame with columns `week` (the
date of the Monday starting each week, counted from `reference_date` as
simulation day 0) and `cases`. Weeks with no cases are left out.

`by` chooses which date places a case in a week:

- `:onset` (default): symptom onset, as on a surveillance epidemic curve. A
  case without an onset time (for example an asymptomatic case from
  `clinical_presentation`) is placed by its infection time.
- `:infection`: infection time. This is not observed in real surveillance,
  and the curve comes out earlier by roughly one incubation period.
- `:reporting`: reporting time, set by [`Reporting`](@ref). Cases never
  reported are left out.
- any other recorded time, such as `:admission_time`.
"""
function weekly_incidence(
        state::SimulationState;
        by::Symbol = :onset,
        reference_date::Date = Date(2020, 1, 1)
    )
    infected = filter(is_infected, state.individuals)
    isempty(infected) && return DataFrame(week = Date[], cases = Int[])

    times = Float64[]
    for ind in infected
        t = _weekly_time(by, ind)
        # Skip non-finite times: `NaN` (no such field) and `Inf` (an
        # unreached state, e.g. an unreported case whose `:reporting_time`
        # keeps its default). `Day(floor(Int, Inf))` would otherwise throw.
        isfinite(t) || continue
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

Containment probability for every combination of scenarios. Each row of the
returned DataFrame holds one combination of the listed values and its
[`containment_probability`](@ref) over `n_sim` simulated outbreaks of a
[`BranchingProcess`](@ref).

`params` maps each setting to the values to try. It must include
`:offspring` (offspring distributions) and may include `:generation_time`,
`:interventions` (each value is a vector of interventions applied together),
`:attributes` and `:population_size`. Other settings are rejected, since they
would not change the simulation. Settings that apply to every run, such as
`max_cases`, are passed once as keywords.

"""
function scenario_sweep(
        params::Dict{Symbol, <:AbstractVector};
        n_sim::Int = 500,
        rng::AbstractRNG = Random.default_rng(),
        sim_kwargs...
    )
    haskey(params, :offspring) || throw(ArgumentError("params must include :offspring"))
    recognised = (
        :offspring, :generation_time, :interventions, :attributes,
        :population_size,
    )
    unknown = setdiff(keys(params), recognised)
    isempty(unknown) || throw(
        ArgumentError(
            "scenario_sweep: unrecognised parameter key(s) $(collect(unknown)). " *
                "Recognised sweep axes are $(collect(recognised)); a simulation control " *
                "(e.g. max_cases) is not swept — pass it once through the keyword arguments."
        )
    )

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

        model = ModelSpec(
            BranchingProcess(offspring, gt; population_size = pop_size);
            interventions, attributes
        )

        results = simulate(model, n_sim; rng = rng, sim_kwargs...)

        for k in keys_ordered
            push!(rows[k], vals[k])
        end
        push!(rows[:containment_probability], containment_probability(results))
    end

    return DataFrame(rows)
end
