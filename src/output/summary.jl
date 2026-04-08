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

Compute effective reproduction number per generation.
A DataFrame with columns `generation` and `R_eff` is returned.
"""
function generation_R(state::SimulationState)
    # Single-pass: count infected individuals per generation
    gen_counts = Dict{Int, Int}()
    for ind in state.individuals
        is_infected(ind) || continue
        g = ind.generation
        gen_counts[g] = get(gen_counts, g, 0) + 1
    end

    isempty(gen_counts) && return DataFrame(generation = Int[], R_eff = Float64[])
    max_gen = maximum(keys(gen_counts))

    generations = Int[]
    r_effs = Float64[]
    for g in 0:(max_gen - 1)
        n_parents = get(gen_counts, g, 0)
        n_parents == 0 && continue
        n_children = get(gen_counts, g + 1, 0)
        push!(generations, g)
        push!(r_effs, n_children / n_parents)
    end

    DataFrame(generation = generations, R_eff = r_effs)
end

"""
    weekly_incidence(state::SimulationState; reference_date::Date=Date(2020, 1, 1))

Compute weekly case counts from a single simulation.
A DataFrame with columns `week` (Date) and `cases` (Int) is returned.
"""
function weekly_incidence(state::SimulationState;
        reference_date::Date = Date(2020, 1, 1))
    infected = filter(is_infected, state.individuals)
    isempty(infected) && return DataFrame(week = Date[], cases = Int[])

    inf_days = [floor(Int, ind.infection_time) for ind in infected]

    weeks = Dict{Date, Int}()
    for d in inf_days
        date = reference_date + Day(d)
        week_start = date - Day(dayofweek(date) - 1)
        weeks[week_start] = get(weeks, week_start, 0) + 1
    end

    df = DataFrame(week = collect(keys(weeks)), cases = collect(values(weeks)))
    sort!(df, :week)
    return df
end

"""
    scenario_sweep(params::Dict{Symbol, Vector}; n_sim=500, sim_opts=SimOpts(), rng=Random.default_rng())

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
    :interventions => [[Isolation(delay=Exponential(d))] for d in [1.0, 2.0, 5.0]],
    :generation_time => [LogNormal(1.6, 0.5)],
))
```
"""
function scenario_sweep(params::Dict{Symbol, <:AbstractVector};
        n_sim::Int = 500,
        sim_opts::SimOpts = SimOpts(),
        rng::AbstractRNG = Random.default_rng())
    haskey(params, :offspring) || throw(ArgumentError("params must include :offspring"))

    keys_ordered = collect(keys(params))
    value_lists = [params[k] for k in keys_ordered]
    combinations = Iterators.product(value_lists...)

    rows = Dict{Symbol, Vector{Any}}(k => Any[] for k in keys_ordered)
    rows[:containment_probability] = Any[]

    for combo in combinations
        vals = Dict(k => v for (k, v) in zip(keys_ordered, combo))

        offspring = vals[:offspring]
        gt = get(vals, :generation_time, nothing)
        interventions = get(vals, :interventions, AbstractIntervention[])
        attributes = get(vals, :attributes, NoAttributes())
        pop_size = get(vals, :population_size, NoPopulation())

        model = gt === nothing ?
                BranchingProcess(offspring; population_size = pop_size) :
                BranchingProcess(offspring, gt; population_size = pop_size)

        results = simulate_batch(model, n_sim;
            interventions = interventions isa Vector ? interventions : [interventions],
            attributes = attributes,
            sim_opts = sim_opts,
            rng = rng)

        for k in keys_ordered
            push!(rows[k], vals[k])
        end
        push!(rows[:containment_probability], containment_probability(results))
    end

    DataFrame(rows)
end
