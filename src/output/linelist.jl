"""
    linelist(state::SimulationState; reference_date=Date(2020, 1, 1),
             infected_only=true)

Return a DataFrame with one row per case. The core columns (`id`,
`parent_id`, `generation`, `chain_id`, `date_infection`) are always
present; any other typed field or `state` entry becomes a column too.
Keys ending in `_time` are converted to dates using `reference_date`, so
`:onset_time` ends up as `date_onset`.

With `infected_only = false`, the table has a row for every individual in
`state` and an extra `infected` column. On a structure-driven model such as
`NetworkProcess` or `HouseholdProcess` this is the whole population; on an
offspring-driven model such as `BranchingProcess` it is the cases plus every
contact they exposed who was not infected. A row that is not infected has no
infection, so `date_infection` and every date derived from it (onset,
reporting, admission, outcome, a traced isolation held back to onset, and any
custom `_time` field) are `missing`. Dates of events that happen to a person
whether or not they are infected are kept: `date_trace`, `date_vaccination`,
`date_immunity`, and `date_isolation` when the isolation is a quarantine on
tracing. An isolation that [`Isolation`](@ref) derived from a
provisional onset is `missing`, or the date of the quarantine it replaced.
Columns that are not dates are reported as stored.

To add a column, write the field during the simulation. `linelist`
reads whatever is on `state`.
"""
function linelist(state::SimulationState;
        reference_date::Date = Date(2020, 1, 1), infected_only::Bool = true)
    cases = infected_only ? filter(is_infected, state.individuals) : state.individuals
    isempty(cases) && return DataFrame()

    cols = Dict{Symbol, Vector}(
        :id => [ind.id for ind in cases],
        :parent_id => [ind.parent_id for ind in cases],
        :generation => [ind.generation for ind in cases],
        :chain_id => [ind.chain_id for ind in cases],
        :date_infection => [_infection_date(reference_date, ind) for ind in cases]
    )
    infected_only || (cols[:infected] = [is_infected(ind) for ind in cases])

    state_keys = Set{Symbol}()
    for ind in cases
        union!(state_keys, keys(ind.state))
    end
    delete!(state_keys, :infected)  # encoded by the row's existence, or the column above

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
    df = DataFrame(from = Int[], to = Int[], infected = Bool[],
        generation = Int[], infection_time = Float64[],
        date_infection = Date[])
    for ind in state.individuals
        for child_id in ind.secondary_case_ids
            child_id > length(state.individuals) && continue
            child = state.individuals[child_id]
            push!(df,
                (
                    from = ind.id,
                    to = child.id,
                    infected = is_infected(child),
                    generation = child.generation,
                    infection_time = child.infection_time,
                    date_infection = _to_date(reference_date, child.infection_time)
                ))
        end
    end
    return df
end

# ── Internal helpers ─────────────────────────────────────────────────

"""Convert a simulation time (real number) to a `Date` offset from
`reference_date`. Non-finite or non-numeric inputs return `missing`."""
function _to_date(reference_date::Date, t::Real)
    isfinite(t) ? reference_date + Day(floor(Int, t)) : missing
end
_to_date(::Date, _) = missing

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
        values = Vector{Union{Date, Missing}}(undef, length(cases))
        any_finite = false
        for (i, ind) in pairs(cases)
            t = is_infected(ind) ? get(ind.state, key, missing) :
                _uninfected_event_time(ind, key)
            d = t isa Real ? _to_date(reference_date, t) : missing
            values[i] = d
            d === missing || (any_finite = true)
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

# An exposed contact who escaped infection keeps its exposure time in state,
# which is not an infection date.
function _infection_date(reference_date::Date, ind)
    is_infected(ind) ? _to_date(reference_date, ind.infection_time) : missing
end

"""The time stored under `key` on an individual who was never infected, or
`missing` when that time is not of an event that happened to them.

An exposed contact who escaped infection still carries its exposure time and
the times derived from it, such as an onset, because interventions like ring
vaccination read them during the run. Those describe an infection that did not
happen, so only events that act on a person regardless of infection are
reported: being traced, vaccinated, gaining vaccine immunity, or being
quarantined. An isolation that `Isolation` wrote was derived from the
provisional onset, so the quarantine it replaced, if any, is reported
instead."""
function _uninfected_event_time(ind, key::Symbol)
    if key in (:trace_time, :vaccination_time, :immunity_time)
        return get(ind.state, key, missing)
    elseif key === :isolation_time
        get(ind.state, :isolated_by_isolation, false) ||
            return get(ind.state, key, missing)
        return get(ind.state, :isolation_time_before_isolation, missing)
    end
    return missing
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
    core = [:id, :parent_id, :generation, :chain_id, :infected, :date_infection]
    keyset = Set(ks)
    ordered = Symbol[k for k in core if k in keyset]
    remaining = [k for k in ks if !(k in ordered)]
    dates = sort!([k for k in remaining if startswith(String(k), "date_")])
    others = sort!([k for k in remaining if !startswith(String(k), "date_")])
    append!(ordered, dates)
    append!(ordered, others)
    return ordered
end
