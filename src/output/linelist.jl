"""
    linelist(state::SimulationState; reference_date=Date(2020, 1, 1),
             delay_opts=DelayOpts(), outcome_opts=OutcomeOpts(),
             demographic_opts=DemographicOpts(), rng=Random.default_rng())

Convert simulation output to a line list DataFrame with one row per case.

Columns: id, case_type, sex, age, date_infection, date_onset, date_reporting,
date_admission, outcome, date_outcome, ct_value.
"""
function linelist(state::SimulationState;
                  reference_date::Date=Date(2020, 1, 1),
                  delay_opts::DelayOpts=DelayOpts(),
                  outcome_opts::OutcomeOpts=OutcomeOpts(),
                  demographic_opts::DemographicOpts=DemographicOpts(),
                  rng::AbstractRNG=Random.default_rng())
    cases = filter(is_infected, state.individuals)
    n = length(cases)
    n == 0 && return _empty_linelist()

    ids = Vector{Int}(undef, n)
    case_types = Vector{String}(undef, n)
    sexes = Vector{String}(undef, n)
    ages = Vector{Int}(undef, n)
    date_infections = Vector{Date}(undef, n)
    date_onsets = Vector{Union{Date, Missing}}(undef, n)
    date_reportings = Vector{Union{Date, Missing}}(undef, n)
    date_admissions = Vector{Union{Date, Missing}}(undef, n)
    outcomes = Vector{String}(undef, n)
    date_outcomes = Vector{Union{Date, Missing}}(undef, n)
    ct_values = Vector{Union{Float64, Missing}}(undef, n)

    for (j, ind) in enumerate(cases)
        ids[j] = ind.id
        case_types[j] = ind.parent_id == 0 ? "index" : "secondary"

        # Demographics
        sexes[j] = rand(rng) < demographic_opts.prob_female ? "female" : "male"
        ages[j] = _sample_age(rng, demographic_opts)

        # Dates
        date_infections[j] = reference_date + Day(floor(Int, ind.infection_time))

        ind_onset = onset_time(ind)
        if !isnan(ind_onset)
            onset_date = reference_date + Day(floor(Int, ind_onset))
            date_onsets[j] = onset_date

            # Reporting delay
            if delay_opts.onset_to_reporting !== nothing
                delay = rand(rng, delay_opts.onset_to_reporting)
                date_reportings[j] = onset_date + Day(floor(Int, delay))
            else
                date_reportings[j] = missing
            end

            # Hospitalisation
            prob_hosp = outcome_opts.prob_hospitalisation
            if rand(rng) < prob_hosp
                if delay_opts.onset_to_admission !== nothing
                    delay = rand(rng, delay_opts.onset_to_admission)
                    date_admissions[j] = onset_date + Day(floor(Int, delay))
                else
                    date_admissions[j] = onset_date
                end
            else
                date_admissions[j] = missing
            end

            # Outcome
            prob_death = _get_cfr(ages[j], outcome_opts)
            if rand(rng) < prob_death
                outcomes[j] = "died"
            else
                outcomes[j] = "recovered"
            end

            if delay_opts.onset_to_outcome !== nothing
                delay = rand(rng, delay_opts.onset_to_outcome)
                date_outcomes[j] = onset_date + Day(floor(Int, delay))
            else
                date_outcomes[j] = missing
            end
        else
            date_onsets[j] = missing
            date_reportings[j] = missing
            date_admissions[j] = missing
            outcomes[j] = "recovered"
            date_outcomes[j] = missing
        end

        ct_values[j] = missing
    end

    DataFrame(
        id=ids,
        case_type=case_types,
        sex=sexes,
        age=ages,
        date_infection=date_infections,
        date_onset=date_onsets,
        date_reporting=date_reportings,
        date_admission=date_admissions,
        outcome=outcomes,
        date_outcome=date_outcomes,
        ct_value=ct_values,
    )
end

"""
    contacts(state::SimulationState; reference_date=Date(2020, 1, 1))

Generate a contacts DataFrame with one row per contact (infected and non-infected).

Columns: from, to, was_case, generation, infection_time, date_infection.
"""
function contacts(state::SimulationState;
                  reference_date::Date=Date(2020, 1, 1))
    froms = Int[]
    tos = Int[]
    was_cases = Bool[]
    generations = Int[]
    infection_times = Float64[]
    date_infections = Date[]

    for ind in state.individuals
        for child_id in ind.secondary_case_ids
            child_idx = findfirst(i -> i.id == child_id, state.individuals)
            child_idx === nothing && continue
            child = state.individuals[child_idx]
            push!(froms, ind.id)
            push!(tos, child.id)
            push!(was_cases, is_infected(child))
            push!(generations, child.generation)
            push!(infection_times, child.infection_time)
            push!(date_infections, reference_date + Day(floor(Int, child.infection_time)))
        end
    end

    DataFrame(
        from=froms, to=tos, was_case=was_cases, generation=generations,
        infection_time=infection_times, date_infection=date_infections,
    )
end

# ── Internal helpers ─────────────────────────────────────────────────

function _empty_linelist()
    DataFrame(
        id=Int[], case_type=String[], sex=String[], age=Int[],
        date_infection=Date[], date_onset=Union{Date, Missing}[],
        date_reporting=Union{Date, Missing}[],
        date_admission=Union{Date, Missing}[],
        outcome=String[], date_outcome=Union{Date, Missing}[],
        ct_value=Union{Float64, Missing}[],
    )
end

function _sample_age(rng::AbstractRNG, opts::DemographicOpts)
    if opts.age_distribution !== nothing
        age = floor(Int, rand(rng, opts.age_distribution))
        return clamp(age, opts.age_range[1], opts.age_range[2])
    else
        return rand(rng, opts.age_range[1]:opts.age_range[2])
    end
end

function _get_cfr(age::Int, opts::OutcomeOpts)
    if opts.age_specific_cfr !== nothing
        for ((lo, hi), cfr) in opts.age_specific_cfr
            lo <= age <= hi && return cfr
        end
    end
    return opts.prob_death
end
