"""
    linelist(state::SimulationState; reference_date=Date(2020, 1, 1))

Return a DataFrame with one row per infected case. The core columns
(`id`, `parent_id`, `generation`, `chain_id`, `date_infection`) are
always present; any other typed field or `state` entry becomes a
column too. Keys ending in `_time` are converted to dates using
`reference_date`, so `:onset_time` ends up as `date_onset`.

To add a column, write the field during the simulation. `linelist`
reads whatever is on `state`.
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
        :date_infection => [reference_date + Day(floor(Int, ind.infection_time))
                            for ind in cases]
    )

    state_keys = Set{Symbol}()
    for ind in cases
        union!(state_keys, keys(ind.state))
    end
    delete!(state_keys, :infected)  # encoded by the row's existence

    for key in state_keys
        _add_state_column!(cols, cases, key, reference_date)
    end

    ordered_keys = _column_order(keys(cols))
    return DataFrame([k => cols[k] for k in ordered_keys])
end

"""
    contacts(state::SimulationState; reference_date=Date(2020, 1, 1))

Return a DataFrame with one row per contact event (infected and
non-infected), with columns `from`, `to`, `infected`, `generation`,
`infection_time`, `date_infection`.
"""
function contacts(state::SimulationState;
        reference_date::Date = Date(2020, 1, 1))
    froms = Int[]
    tos = Int[]
    infecteds = Bool[]
    generations = Int[]
    infection_times = Float64[]
    date_infections = Date[]

    for ind in state.individuals
        for child_id in ind.secondary_case_ids
            child_id > length(state.individuals) && continue
            child = state.individuals[child_id]
            push!(froms, ind.id)
            push!(tos, child.id)
            push!(infecteds, is_infected(child))
            push!(generations, child.generation)
            push!(infection_times, child.infection_time)
            push!(date_infections, reference_date + Day(floor(Int, child.infection_time)))
        end
    end

    DataFrame(
        from = froms, to = tos, infected = infecteds, generation = generations,
        infection_time = infection_times, date_infection = date_infections
    )
end

# ── Internal helpers ─────────────────────────────────────────────────

"""Add a column for a single state key, applying the `_time` → `date_`
convention for keys whose name ends in `_time` and which carry numeric
values. Other keys pass through. Columns are omitted only if every
case's value is `missing` (or, for `_time` keys, every value is
non-finite/non-numeric)."""
function _add_state_column!(cols, cases, key::Symbol, reference_date)
    key_name = String(key)
    if endswith(key_name, "_time")
        prefix = key_name[1:(end - length("_time"))]
        col_name = Symbol("date_" * prefix)
        col_name in keys(cols) && return nothing  # don't shadow core columns
        any_finite = false
        values = Vector{Union{Date, Missing}}(undef, length(cases))
        for (i, ind) in pairs(cases)
            t = get(ind.state, key, missing)
            if t isa Real && isfinite(float(t))
                values[i] = reference_date + Day(floor(Int, float(t)))
                any_finite = true
            else
                values[i] = missing
            end
        end
        any_finite || return nothing
        cols[col_name] = values
    else
        key in keys(cols) && return nothing
        raw = [get(ind.state, key, missing) for ind in cases]
        all(ismissing, raw) && return nothing
        cols[key] = _normalise_column(raw)
    end
    return nothing
end

"""Convert `Symbol` entries to `String` so DataFrames serialises cleanly;
otherwise leave the column untouched."""
function _normalise_column(values)
    if any(v -> v isa Symbol, values)
        return Union{String, Missing}[v isa Symbol ? String(v) : v for v in values]
    end
    return values
end

"""Sensible column ordering for the linelist DataFrame: core simulation
columns first, then date columns (alphabetical), then the rest
(alphabetical)."""
function _column_order(ks)
    core = [:id, :parent_id, :generation, :chain_id, :date_infection]
    keyset = Set(ks)
    ordered = Symbol[k for k in core if k in keyset]
    remaining = [k for k in ks if !(k in ordered)]
    dates = sort!([k for k in remaining if startswith(String(k), "date_")])
    others = sort!([k for k in remaining if !startswith(String(k), "date_")])
    append!(ordered, dates)
    append!(ordered, others)
    return ordered
end
