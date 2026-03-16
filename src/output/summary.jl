"""
    containment_probability(states::Vector{SimulationState}; max_cases=nothing)

Fraction of simulations that went extinct (i.e. the outbreak was contained).

If `max_cases` is provided, simulations that hit the case cap are not
considered extinct (they are assumed to have continued growing).
"""
function containment_probability(states::Vector{SimulationState};
                                  max_cases::Union{Int, Nothing}=nothing)
    n_extinct = count(states) do s
        if max_cases !== nothing && s.cumulative_cases >= max_cases
            return false
        end
        s.extinct
    end
    return n_extinct / length(states)
end

"""
    is_extinct(state::SimulationState; by_week=nothing, reference_date=Date(2020,1,1),
               max_cases=nothing)

Flexible extinction classification for a single simulation.

- No keyword args: returns `state.extinct`
- `by_week::Int`: extinct if no cases with onset in that week
- `by_week::UnitRange{Int}`: extinct if no cases with onset in that week range
- `max_cases::Int`: outbreaks hitting this cap are not considered extinct
"""
function is_extinct(state::SimulationState;
                    by_week::Union{Int, UnitRange{Int}, Nothing}=nothing,
                    reference_date::Date=Date(2020, 1, 1),
                    max_cases::Union{Int, Nothing}=nothing)
    if max_cases !== nothing && state.cumulative_cases >= max_cases
        return false
    end

    by_week === nothing && return state.extinct

    # Week-based extinction
    weeks = by_week isa Int ? (by_week:by_week) : by_week
    for ind in state.individuals
        !is_infected(ind) && continue
        isnan(ind.infection_time) && continue
        day = floor(Int, ind.infection_time)
        date = reference_date + Day(day)
        week_num = div(Dates.value(date - reference_date), 7) + 1
        week_num in weeks && return false
    end
    return true
end

"""
    effective_R(state::SimulationState)

Compute effective reproduction number per generation.
Returns a DataFrame with columns `generation` and `R_eff`.
"""
function effective_R(state::SimulationState)
    infected = filter(is_infected, state.individuals)
    isempty(infected) && return DataFrame(generation=Int[], R_eff=Float64[])
    max_gen = maximum(ind.generation for ind in infected)

    generations = Int[]
    r_effs = Float64[]

    for g in 0:(max_gen - 1)
        parents = filter(i -> i.generation == g, infected)
        children = filter(i -> i.generation == g + 1, infected)

        n_parents = length(parents)
        n_parents == 0 && continue

        push!(generations, g)
        push!(r_effs, length(children) / n_parents)
    end

    DataFrame(generation=generations, R_eff=r_effs)
end

"""
    weekly_incidence(state::SimulationState; reference_date::Date=Date(2020, 1, 1))

Compute weekly case counts from a single simulation.
Returns a DataFrame with columns `week` (Date) and `cases` (Int).
"""
function weekly_incidence(state::SimulationState;
                          reference_date::Date=Date(2020, 1, 1))
    infected = filter(is_infected, state.individuals)
    isempty(infected) && return DataFrame(week=Date[], cases=Int[])

    inf_days = [floor(Int, ind.infection_time) for ind in infected]

    weeks = Dict{Date, Int}()
    for d in inf_days
        date = reference_date + Day(d)
        week_start = date - Day(dayofweek(date) - 1)
        weeks[week_start] = get(weeks, week_start, 0) + 1
    end

    df = DataFrame(week=collect(keys(weeks)), cases=collect(values(weeks)))
    sort!(df, :week)
    return df
end
