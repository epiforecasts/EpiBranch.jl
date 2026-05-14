"""
    linelist(state::SimulationState; reference_date=Date(2020, 1, 1))

Project simulation output to a line list DataFrame with one row per
infected case. Pure projection of `state` — no RNG draws, no
probabilistic decisions. Clinical events (reporting, admission, death,
recovery) and demographics are read from the state keys that
attributes and clinical transitions have already populated during
[`simulate`](@ref).

Columns are built dynamically from what is available on each individual:

- `id`, `parent_id`, `generation`, `chain_id`, `case_type` — always present.
- `date_infection` — always present (from `infection_time` + `reference_date`).
- `date_onset` — if `:onset_time` was set, e.g. via [`clinical_presentation`](@ref).
- `age`, `sex` — if set via the [`demographics`](@ref) attributes builder.
- `type` — if the model is multi-type.
- `date_reporting` — if a [`Reporting`](@ref) transition wrote `:reporting_time`.
- `date_admission` — if a [`Hospitalisation`](@ref) transition wrote `:admission_time`.
- `outcome`, `date_outcome` — if [`Death`](@ref) / [`Recovery`](@ref) (or
  any user-defined terminal transition) wrote `:outcome` / `:outcome_time`.
- Any present `:asymptomatic` / `:isolated` / `:traced` / `:quarantined` /
  `:vaccinated` flags.

To get reporting, admission, or outcome events in the line list, build
the corresponding transitions and pass them via `simulate(...; transitions = ...)`.
"""
function linelist(state::SimulationState;
        reference_date::Date = Date(2020, 1, 1))
    cases = filter(is_infected, state.individuals)
    isempty(cases) && return DataFrame()

    cols = Dict{Symbol, Vector}(
        :id => [ind.id for ind in cases],
        :parent_id => [ind.parent_id for ind in cases],
        :generation => [ind.generation for ind in cases],
        :chain_id => [ind.chain_id for ind in cases],
        :case_type => [ind.parent_id == 0 ? "index" : "secondary" for ind in cases],
        :date_infection => [reference_date + Day(floor(Int, ind.infection_time))
                            for ind in cases]
    )

    _add_time_column!(cols, :date_onset, cases, :onset_time, reference_date)
    _add_time_column!(cols, :date_reporting, cases, :reporting_time, reference_date)
    _add_time_column!(cols, :date_admission, cases, :admission_time, reference_date)
    _add_time_column!(cols, :date_outcome, cases, :outcome_time, reference_date)

    if any(haskey(ind.state, :age) for ind in cases)
        cols[:age] = [get(ind.state, :age, missing) for ind in cases]
    end
    if any(haskey(ind.state, :sex) for ind in cases)
        cols[:sex] = Union{String, Missing}[let v = get(ind.state, :sex, missing)
                                                v === missing ? missing : string(v)
                                            end
                                            for ind in cases]
    end
    if any(haskey(ind.state, :type) for ind in cases)
        cols[:type] = [get(ind.state, :type, missing) for ind in cases]
    end

    if any(haskey(ind.state, :outcome) for ind in cases)
        cols[:outcome] = Union{String, Missing}[haskey(ind.state, :outcome) ?
                                                string(ind.state[:outcome]) : missing
                                                for ind in cases]
    end

    for field in (:asymptomatic, :isolated, :traced, :quarantined, :vaccinated)
        if any(haskey(ind.state, field) for ind in cases)
            cols[field] = [get(ind.state, field, missing) for ind in cases]
        end
    end

    ordered_keys = _column_order(keys(cols))
    return DataFrame([k => cols[k] for k in ordered_keys])
end

"""
    contacts(state::SimulationState; reference_date=Date(2020, 1, 1))

Generate a contacts DataFrame with one row per contact (infected and non-infected).
"""
function contacts(state::SimulationState;
        reference_date::Date = Date(2020, 1, 1))
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
        from = froms, to = tos, was_case = was_cases, generation = generations,
        infection_time = infection_times, date_infection = date_infections
    )
end

# ── Internal helpers ─────────────────────────────────────────────────

"""Add a date column projected from a Float64 time stored on `ind.state[time_key]`.
Includes the column iff at least one case has a finite time. Cases with no
key or a NaN/Inf value get `missing`."""
function _add_time_column!(cols, col_name, cases, time_key, reference_date)
    any(haskey(ind.state, time_key) && isfinite(ind.state[time_key]::Float64)
    for ind in cases) || return nothing
    cols[col_name] = Union{Date, Missing}[let t = get(ind.state, time_key, NaN)::Float64
                                              isfinite(t) ?
                                              reference_date + Day(floor(Int, t)) : missing
                                          end
                                          for ind in cases]
    return nothing
end

"""Sensible column ordering for the linelist DataFrame."""
function _column_order(ks)
    priority = [
        :id, :parent_id, :generation, :chain_id, :case_type, :type,
        :age, :sex, :date_infection, :date_onset, :date_reporting,
        :date_admission, :outcome, :date_outcome,
        :asymptomatic, :isolated, :traced, :quarantined, :vaccinated
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
