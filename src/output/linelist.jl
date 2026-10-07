"""
    event_time_metadata(::Val{key})

How a recorded event time appears as a date column in [`linelist`](@ref). Use
it when a custom intervention or progression step records its own event times
and you want them shown as dates.

Returns `(column = :date_name, requires_infection = true)`, or `nothing` for a
column that is not a date. By default, any time recorded under a name ending
in `_time` becomes a `date_` column (`:onset_time` becomes `date_onset`) and
is shown only for infected people. Times that are infinite or not numbers
become `missing`.

An event that can happen to people who were never infected (an appointment,
say) declares that its date applies to them too:

```julia
EpiBranch.event_time_metadata(::Val{:appointment_time}) =
    (column = :date_appointment, requires_infection = false)
```

Tracing, vaccination and immunity dates, including those of named doses, are
shown whether or not the person was infected. This setting changes only the
line list, not the simulation.
"""
function event_time_metadata(::Val{key}) where {key}
    name = String(key)
    # Dose labels follow the time marker rather than preceding it.
    for event in (:vaccination, :immunity)
        prefix = string(event, "_time_")
        if startswith(name, prefix)
            label = name[(length(prefix) + 1):end]
            return (
                column = Symbol("date_", event, "_", label),
                requires_infection = false,
            )
        end
    end
    endswith(name, "_time") || return nothing
    return (
        column = Symbol("date_", name[1:(end - length("_time"))]),
        requires_infection = true,
    )
end
for key in (:trace_time, :vaccination_time, :immunity_time)
    column = Symbol("date_", String(key)[1:(end - length("_time"))])
    @eval event_time_metadata(::Val{$(QuoteNode(key))}) = (
        column = $(QuoteNode(column)), requires_infection = false,
    )
end

"""
    linelist(state::SimulationState; reference_date=Date(2020, 1, 1),
             infected_only=true)

The line list of a simulated outbreak: a DataFrame with one row per case,
with event times converted to calendar dates counted from `reference_date`
(simulation day 0).

Columns always present:

| Column | Meaning |
|:--- | :--- |
| `id` | case identifier |
| `parent_id` | `id` of the infector (0 for an index case) |
| `generation` | generation number (index cases are generation 0) |
| `chain_id` | which index case the case descends from |
| `date_infection` | date of infection |

Further columns depend on the model and are added whenever any case has the
information: for example `date_onset` and `asymptomatic` from
[`clinical_presentation`](@ref), `age` and `sex` from [`demographics`](@ref),
`isolated` and `date_isolation` from [`Isolation`](@ref) (with
`date_isolation_release` when isolation or quarantine has a finite
`duration`), `date_reporting` from [`Reporting`](@ref), and `outcome` and
`date_outcome` from [`Death`](@ref) and [`Recovery`](@ref). Any time recorded under a name ending
in `_time` becomes a `date_` column (`:onset_time` becomes `date_onset`);
other information appears as recorded. Names starting with an underscore are
internal to an intervention and never shown. To add your own date columns see
[`event_time_metadata`](@ref).

`isolated`, `date_isolation` and `date_isolation_release` show an isolation
only when it counted as a detection (see [`is_isolated`](@ref)).

# Including people who were not infected

With `infected_only = false`, the table has a row for every person in the
simulation and an extra `infected` column. For a model with a fixed
population, such as `NetworkProcess` or `HouseholdProcess`, that is everyone;
for `BranchingProcess` it is the cases plus every contact they exposed who was
not infected.

An uninfected person has `missing` for `date_infection` and every date that
depends on infection (onset, reporting, admission, outcome, custom dates that
require infection). Dates of events that can happen without infection are
kept: `date_trace`, `date_vaccination`, `date_immunity`, and, for a traced
contact put in quarantine, `date_isolation` (with `date_isolation_release`
when the isolation or quarantine has a finite duration). If [`Isolation`](@ref)
later replaced a quarantine with an isolation based on an onset that never
happened, these two columns show the original quarantine, or `missing` if
there was none.

# Examples

```julia
using Dates
model = ModelSpec(
    BranchingProcess(NegBin(2.5, 0.16), Gamma(2.0, 3.0));
    attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5))
)
state = simulate(model; max_cases = 100)
ll = linelist(state; reference_date = Date(2024, 3, 1))
first(ll, 5)
```
"""
function linelist(
        state::SimulationState;
        reference_date::Date = Date(2020, 1, 1), infected_only::Bool = true
    )
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
    filter!(key -> !startswith(String(key), "_"), state_keys)  # an intervention's own bookkeeping
    delete!(state_keys, :infected)  # encoded by the row's existence, or the column above

    for key in state_keys
        _add_state_column!(cols, cases, key, reference_date)
    end

    ordered_keys = _column_order(keys(cols))
    return DataFrame([k => cols[k] for k in ordered_keys])
end

"""
    contacts(state::SimulationState; reference_date=Date(2020, 1, 1))

Who exposed whom in a simulated outbreak: a DataFrame with one row per
exposure of a person by an infectious case, whether or not it led to
infection. Filter on `infected` to keep only the transmission tree.

Columns: `from` (`id` of the case), `to` (`id` of the person exposed),
`infected` (whether that exposure infected them), `generation` (generation of
the person exposed), `infection_time` (day of exposure, which is the day of
infection when `infected` is true) and `date_infection` (the same as a date
counted from `reference_date`).

# Examples

```julia
state = simulate(ModelSpec(BranchingProcess(NegBin(2.5, 0.16), Gamma(2.0, 3.0)));
    max_cases = 100)
ct = contacts(state)
tree = ct[ct.infected, :]   # infector-infectee pairs
```
"""
function contacts(
        state::SimulationState;
        reference_date::Date = Date(2020, 1, 1)
    )
    df = DataFrame(
        from = Int[], to = Int[], infected = Bool[],
        generation = Int[], infection_time = Float64[],
        date_infection = Date[]
    )
    for ind in state.individuals
        for child_id in ind.secondary_case_ids
            child_id > length(state.individuals) && continue
            child = state.individuals[child_id]
            push!(
                df,
                (
                    from = ind.id,
                    to = child.id,
                    infected = is_infected(child),
                    generation = child.generation,
                    infection_time = child.infection_time,
                    date_infection = _to_date(reference_date, child.infection_time),
                )
            )
        end
    end
    return df
end

# ── Internal helpers ─────────────────────────────────────────────────

"""Convert a simulation time (real number) to a `Date` offset from
`reference_date`. Non-finite or non-numeric inputs return `missing`."""
function _to_date(reference_date::Date, t::Real)
    return isfinite(t) ? reference_date + Day(floor(Int, t)) : missing
end
_to_date(::Date, _) = missing

"""Add a column for a single state key, applying the `_time` → `date_`
convention for keys whose name ends in `_time` and which carry numeric
values. Other keys pass through. Columns are omitted only if every
case's value is `missing` (or, for `_time` keys, every value is
non-finite/non-numeric)."""
function _add_state_column!(cols, cases, key::Symbol, reference_date)
    metadata = event_time_metadata(Val(key))
    if metadata !== nothing
        col_name = metadata.column
        col_name in keys(cols) && return nothing  # don't shadow core columns
        values = Vector{Union{Date, Missing}}(undef, length(cases))
        any_finite = false
        for (i, ind) in pairs(cases)
            t = is_infected(ind) ? _reported_state(ind, key) :
                _uninfected_event_time(ind, key, metadata)
            d = t isa Real ? _to_date(reference_date, t) : missing
            values[i] = d
            d === missing || (any_finite = true)
        end
        any_finite || return nothing
        cols[col_name] = values
    else
        key in keys(cols) && return nothing
        raw = [_reported_state(ind, key) for ind in cases]
        all(ismissing, raw) && return nothing
        cols[key] = _normalise_column(raw)
    end
    return nothing
end

# An exposed contact who escaped infection keeps its exposure time in state,
# which is not an infection date.
function _infection_date(reference_date::Date, ind)
    return is_infected(ind) ? _to_date(reference_date, ind.infection_time) : missing
end

"""The time stored under `key` on an individual who was never infected, or
`missing` when that time is not of an event that happened to them.

An exposed contact who escaped infection still holds its exposure time and the
times derived from it, such as an onset, because interventions like ring
vaccination read them during the run. Those times describe an infection that
never happened, so the reported events are only the ones that act on a person
regardless of infection: being traced, vaccinated, gaining vaccine immunity, or
being quarantined. An isolation written by `Isolation` came from the
provisional onset; where one replaced a recorded quarantine, that quarantine's
time, and its release time, are reported in its place."""
function _uninfected_event_time(ind, key::Symbol, metadata)
    if key === :isolation_time || key === :isolation_release_time
        get(ind.state, :_isolated_by_isolation, false) ||
            return _reported_state(ind, key)
        get(ind.state, :_isolation_unrecorded_before_isolation, false) && return missing
        before_key = key === :isolation_time ? :_isolation_time_before_isolation :
            :_isolation_release_time_before_isolation
        return get(ind.state, before_key, missing)
    end
    return metadata.requires_infection ? missing : _reported_state(ind, key)
end

# The value a state key reports in the line list. The isolation columns report
# detections, so an isolation that removes the case from transmission without
# being recorded (see `records_isolation`) reads as no isolation.
function _reported_state(ind, key::Symbol)
    haskey(ind.state, key) || return missing
    key === :isolated && return is_isolated(ind)
    (key === :isolation_time || key === :isolation_release_time) &&
        _isolation_unrecorded(ind) && return missing
    return ind.state[key]
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
