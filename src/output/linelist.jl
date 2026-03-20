"""
    linelist(state::SimulationState; reference_date=Date(2020, 1, 1),
             delays=DelayOpts(), outcomes=OutcomeOpts(),
             demographics=nothing, rng=Random.default_rng())

Convert simulation output to a line list DataFrame with one row per case
(infected individuals only).

Columns are built dynamically from what is available on each individual:

- `id`, `parent_id`, `generation`, `chain_id`, `case_type` — always present
- `date_infection` — always present (from `infection_time` + `reference_date`)
- `date_onset` — if `:onset_time` was set (via `clinical_presentation()`)
- `age`, `sex` — if set via `demographics()` attributes or multi-type.
  If not present and `demographics` kwarg is provided, generated post-hoc.
- `date_reporting`, `date_admission`, `date_outcome`, `outcome` — if
  `delays` and/or `outcomes` are provided and onset times are available
- Any other state dict fields are included as additional columns.

If a `DemographicOpts` is passed via the `demographics` kwarg, age/sex
are generated post-hoc for individuals that don't already have them.
"""
function linelist(state::SimulationState;
                  reference_date::Date=Date(2020, 1, 1),
                  delays::DelayOpts=DelayOpts(),
                  outcomes::Union{OutcomeOpts, Nothing}=nothing,
                  demographics::Union{DemographicOpts, Nothing}=nothing,
                  rng::AbstractRNG=Random.default_rng())
    cases = filter(is_infected, state.individuals)
    n = length(cases)
    n == 0 && return DataFrame()

    # Core columns — always present
    cols = Dict{Symbol, Vector}(
        :id => [ind.id for ind in cases],
        :parent_id => [ind.parent_id for ind in cases],
        :generation => [ind.generation for ind in cases],
        :chain_id => [ind.chain_id for ind in cases],
        :case_type => [ind.parent_id == 0 ? "index" : "secondary" for ind in cases],
        :date_infection => [reference_date + Day(floor(Int, ind.infection_time))
                           for ind in cases],
    )

    # Onset time → date_onset
    has_onset = any(haskey(ind.state, :onset_time) for ind in cases)
    if has_onset
        cols[:date_onset] = Union{Date, Missing}[
            let ot = onset_time(ind)
                isnan(ot) ? missing : reference_date + Day(floor(Int, ot))
            end
            for ind in cases
        ]
    end

    # Demographics — from state dict or generated post-hoc
    has_age = any(haskey(ind.state, :age) for ind in cases)
    has_sex = any(haskey(ind.state, :sex) for ind in cases)

    # Apply post-hoc demographics if not already set on individuals
    if demographics !== nothing && (!has_age || !has_sex)
        demo_fn = EpiBranch.demographics(;
            age_distribution=demographics.age_distribution,
            age_range=demographics.age_range,
            prob_female=demographics.prob_female)
        for ind in cases
            haskey(ind.state, :age) || demo_fn(rng, ind)
        end
        has_age = true
        has_sex = true
    end

    if has_age
        cols[:age] = [get(ind.state, :age, missing) for ind in cases]
    end
    if has_sex
        cols[:sex] = [string(get(ind.state, :sex, missing)) for ind in cases]
    end

    # Type (from multi-type branching process)
    has_type = any(haskey(ind.state, :type) for ind in cases)
    if has_type
        cols[:type] = [get(ind.state, :type, missing) for ind in cases]
    end

    # Reporting delay
    if has_onset && delays.onset_to_reporting !== nothing
        cols[:date_reporting] = Union{Date, Missing}[
            let ot = onset_time(ind)
                if isnan(ot)
                    missing
                else
                    d = rand(rng, delays.onset_to_reporting)
                    reference_date + Day(floor(Int, ot + d))
                end
            end
            for ind in cases
        ]
    end

    # Hospitalisation
    if has_onset && delays.onset_to_admission !== nothing
        prob_hosp = outcomes !== nothing ? outcomes.prob_hospitalisation : 0.2
        cols[:date_admission] = Union{Date, Missing}[
            let ot = onset_time(ind)
                if isnan(ot) || rand(rng) >= prob_hosp
                    missing
                else
                    d = rand(rng, delays.onset_to_admission)
                    reference_date + Day(floor(Int, ot + d))
                end
            end
            for ind in cases
        ]
    end

    # Outcomes
    if outcomes !== nothing && has_onset
        age_col = get(cols, :age, nothing)
        outcome_strs = Vector{String}(undef, n)
        date_outcomes = Vector{Union{Date, Missing}}(undef, n)

        for (j, ind) in enumerate(cases)
            ot = onset_time(ind)
            if isnan(ot)
                outcome_strs[j] = "recovered"
                date_outcomes[j] = missing
            else
                prob_death = if age_col !== nothing && !ismissing(age_col[j])
                    _get_cfr(Int(age_col[j]), outcomes)
                else
                    outcomes.prob_death
                end

                outcome_strs[j] = rand(rng) < prob_death ? "died" : "recovered"

                if delays.onset_to_outcome !== nothing
                    d = rand(rng, delays.onset_to_outcome)
                    date_outcomes[j] = reference_date + Day(floor(Int, ot + d))
                else
                    date_outcomes[j] = missing
                end
            end
        end

        cols[:outcome] = outcome_strs
        cols[:date_outcome] = date_outcomes
    end

    # Intervention state — include if present
    for field in [:asymptomatic, :isolated, :traced, :quarantined, :vaccinated]
        if any(haskey(ind.state, field) for ind in cases)
            cols[field] = [get(ind.state, field, missing) for ind in cases]
        end
    end

    # Build DataFrame with a sensible column order
    ordered_keys = _column_order(keys(cols))
    return DataFrame([k => cols[k] for k in ordered_keys])
end

"""
    contacts(state::SimulationState; reference_date=Date(2020, 1, 1))

Generate a contacts DataFrame with one row per contact (infected and non-infected).
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
            child_id > length(state.individuals) && continue
            child = state.individuals[child_id]
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


function _get_cfr(age::Int, opts::OutcomeOpts)
    if opts.age_specific_cfr !== nothing
        for ((lo, hi), cfr) in opts.age_specific_cfr
            lo <= age <= hi && return cfr
        end
    end
    return opts.prob_death
end

"""Sensible column ordering for the linelist DataFrame."""
function _column_order(ks)
    priority = [
        :id, :parent_id, :generation, :chain_id, :case_type, :type,
        :age, :sex, :date_infection, :date_onset, :date_reporting,
        :date_admission, :outcome, :date_outcome,
        :asymptomatic, :isolated, :traced, :quarantined, :vaccinated,
    ]
    ordered = Symbol[]
    for k in priority
        k in ks && push!(ordered, k)
    end
    for k in sort(collect(ks))
        k in ordered || push!(ordered, k)
    end
    return ordered
end
