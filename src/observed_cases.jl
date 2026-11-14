# ── Starting an outbreak from observed cases ─────────────────────────
# A line list is a set of cases already infected, some with a known infector.
# The state built from it is the race of a branching process, run from those
# cases up to `now` and cut there, so that `simulate(model, state; max_time)`
# continues it exactly as it continues any cut run.
#
# Each observed case is a member of the race at its observed infection time.
# What the table already says about a case's offspring (the rows naming it as
# their infector) is kept; the model draws the rest, conditioned on those
# being all the contacts that fell by `now`.

const _MAX_CONDITIONING_ATTEMPTS = 100_000

# The offspring of a window, in which an observed case's contacts are drawn
# given what is known of them. A case the table did not list draws from the
# window's own offspring as usual.
#
# For an observed case, the offspring count and each contact's time are drawn
# from the window together and kept only when exactly as many contacts fall
# up to `now` as the table has rows naming the case as their infector. The
# contacts the table already holds stand as they are; those that fall later
# are returned with their times. A case isolated at a known time infects no
# one after it, so contacts after that time are neither counted nor returned.
struct _ObservedOffspring{O, K, F, C, T <: Real}
    base::O
    kernel::K
    from::F
    from_key::Union{Symbol, Nothing}
    calendar::C
    known::Vector{Int}
    cutoff::Vector{T}
    now::T
end

function (w::_ObservedOffspring)(rng, ind, state)
    ind.id <= length(w.known) || return draw_offspring(rng, w.base, ind, state)
    from_t = _state_time(ind, w.from, w.from_key)
    isfinite(from_t) || return 0
    known = w.known[ind.id]
    cut = min(w.now, w.cutoff[ind.id])
    kernel = _calendar_scaled_kernel(get_generation_time(w.kernel, ind), w.calendar, from_t)
    for _ in 1:_MAX_CONDITIONING_ATTEMPTS
        n = draw_offspring(rng, w.base, ind, state)
        n isa Integer || throw(
            ArgumentError(
                "starting from observed cases needs an offspring law that draws a " *
                    "number of contacts per case, not $(typeof(n))"
            )
        )
        later = typeof(from_t)[]
        seen = 0
        for _ in 1:n
            t = _window_infection_time(kernel, from_t, state)
            t < ind.infection_time && continue
            if t <= cut
                seen += 1
            elseif t > w.now && t <= w.cutoff[ind.id]
                push!(later, t - from_t)
            end
        end
        seen == known && return OffspringTimes(later)
    end
    throw(
        ErrorException(
            "case $(ind.id) has $known known offspring by time $(w.now), which the " *
                "model's offspring law and generation time made no draw of in " *
                "$_MAX_CONDITIONING_ATTEMPTS attempts"
        )
    )
end

# The observed cases' columns, validated and converted: infection times,
# the position of each case's infector (0 for none), and the optional onset
# and isolation times (`NaN` and `Inf` where absent).
function _read_observed_cases(cases::AbstractDataFrame)
    names_ = Set(propertynames(cases))
    :infection_time in names_ || throw(
        ArgumentError("cases needs an `infection_time` column")
    )
    unknown = setdiff(names_, (:id, :parent_id, :infection_time, :onset_time, :isolation_time))
    isempty(unknown) || throw(
        ArgumentError(
            "cases has unrecognised columns $(sort!(collect(unknown))); the columns " *
                "read are id, parent_id, infection_time, onset_time and isolation_time"
        )
    )
    n = nrow(cases)
    n > 0 || throw(ArgumentError("cases has no rows"))
    labels = :id in names_ ? collect(cases.id) : collect(1:n)
    allunique(labels) || throw(ArgumentError("the `id` column must hold distinct values"))
    position = Dict(label => i for (i, label) in enumerate(labels))
    inf = _required_times(cases.infection_time, "infection_time")
    parent = zeros(Int, n)
    if :parent_id in names_
        for (i, p) in enumerate(cases.parent_id)
            ismissing(p) && continue
            haskey(position, p) || throw(
                ArgumentError("case $(labels[i]) names an infector, $p, that is not in cases")
            )
            parent[i] = position[p]
        end
    end
    for i in 1:n
        parent[i] == 0 && continue
        parent[i] == i && throw(ArgumentError("case $(labels[i]) is its own infector"))
        inf[parent[i]] <= inf[i] || throw(
            ArgumentError(
                "case $(labels[i]) is infected before its infector, $(labels[parent[i]])"
            )
        )
    end
    onset = _optional_times(cases, :onset_time, NaN)
    isolation = _optional_times(cases, :isolation_time, Inf)
    return (; inf, parent, onset, isolation)
end

function _required_times(column, name)
    any(ismissing, column) && throw(ArgumentError("`$name` must not have missing values"))
    times = Float64.(column)
    all(isfinite, times) || throw(ArgumentError("`$name` must be finite"))
    return times
end

function _optional_times(cases, name, absent)
    name in propertynames(cases) || return fill(absent, nrow(cases))
    return Float64[ismissing(t) ? absent : Float64(t) for t in cases[!, name]]
end

# Generation and chain of each case from its infector's, which comes earlier
# in time order. Cases are visited in order of infection time, so an infector
# is always done before the cases it infected, whatever the row order.
function _generations(parent, inf)
    n = length(parent)
    generation = zeros(Int, n)
    chain = collect(1:n)
    for i in sortperm(inf)
        parent[i] == 0 && continue
        generation[i] = generation[parent[i]] + 1
        chain[i] = chain[parent[i]]
    end
    return generation, chain
end

function _initialise_observed(
        process::BranchingProcess, cases::AbstractDataFrame;
        interventions, attributes, progression, recorder, now, rng
    )
    length(process.infectiousness) == 1 || throw(
        ArgumentError(
            "starting from observed cases needs a branching process with a single " *
                "`Infectiousness` window: which window made each observed contact " *
                "is not recorded"
        )
    )
    process.n_types == 1 || throw(
        ArgumentError(
            "starting from observed cases is not supported for a multi-type process"
        )
    )
    _validate_process_windows(process, progression)
    _check_population(process)
    obs = _read_observed_cases(cases)
    n = length(obs.inf)
    now_ = now === nothing ? maximum(obs.inf) : Float64(now)
    maximum(obs.inf) <= now_ || throw(
        ArgumentError("`now` ($now_) is before the latest infection in cases")
    )
    state = new_state(process, progression, attributes, rng)
    T = _timetype(state)
    generation, chain = _generations(obs.parent, obs.inf)
    for i in 1:n
        ind = _create_individual(state, obs.parent[i], chain[i], i, obs.inf[i])
        ind.generation = generation[i]
        isnan(obs.onset[i]) ||
            (ind.state[:incubation_period] = T(obs.onset[i] - obs.inf[i]))
        isfinite(obs.isolation[i]) &&
            set_isolated!(ind, obs.isolation[i]; release_time = Inf)
        _initialise!(interventions, ind, state)
        push!(state.individuals, ind)
    end
    known = zeros(Int, n)
    for i in 1:n
        obs.parent[i] == 0 && continue
        push!(state.individuals[obs.parent[i]].secondary_case_ids, i)
        known[obs.parent[i]] += 1
    end
    _validate_required_fields(first(state.individuals), interventions)
    if any(t -> !isempty(required_fields(t)), progression)
        _validate_required_fields(
            _transition_probe(first(state.individuals), progression, state), progression
        )
    end
    state.active_ids = collect(1:n)
    state.cumulative_cases = 0

    w = only(process.infectiousness)
    observed = _ObservedOffspring(
        w.offspring, w.kernel, w.from, w.from_key, w.calendar, known,
        T.(obs.isolation), T(now_)
    )
    window = Infectiousness(
        observed, w.from, w.from_key, w.until, w.kernel, w.calendar, w.name
    )
    tree = _take_tree(T, n, (window.until,))
    routes = ((_offspring_window(window), _Offspring(window, 1, tree, 1)),)
    ivs = isempty(interventions) ? () : interventions
    extinct = _sellke_race!(
        state, tree, rng;
        seed! = (best, _, _) -> copyto!(best, T.(obs.inf)), routes,
        interventions = ivs,
        contacts = (id, state) -> state.individuals[id].secondary_case_ids,
        risks = transmission_risks(process), recorder, max_time = T(now_)
    )
    state.continuation === nothing && _return_pooled!(tree)
    _reconcile_sellke_bookkeeping!(state, extinct)
    # The race settles every observed case as a seed; those with a known
    # infector are not index cases.
    for i in 1:n
        obs.parent[i] == 0 || (state.individuals[i].state[:index] = false)
    end
    return state
end

"""
    initialise_state(model, cases::AbstractDataFrame; now = nothing, rng = Random.default_rng())

Build the [`SimulationState`](@ref) of an outbreak that has already started,
from a table of the cases observed so far. Continue it with
`simulate(model, state; max_time)`, which draws the infections still to come
from the cases already there, for a short-term projection or for the
probability that an outbreak is over under interventions that have no closed
form.

`cases` has one row per case, with these columns:

- `infection_time` (required): when the case was infected, in the units of the
  model's delays.
- `id`: a label for each case, distinct across rows. Row numbers are used when
  there is none.
- `parent_id`: the `id` of the case that infected this one, `missing` if
  unknown. A case whose infector is not known is treated as an index case. The
  infector's offspring are then counted from these rows.
- `onset_time`: when symptoms began, `missing` if they have not. It is used in
  place of the model's draw of the incubation period, so a model that times
  its contacts or interventions from onset reads the observed one.
- `isolation_time`: when the case was isolated, `missing` if it has not been.
  The case infects no one from then on.

`now` is the time of the data, defaulting to the latest `infection_time`, and
no case may be infected after it. The state is the outbreak at `now`: every
contact the table lists stands, and the model draws any others that the cases
make after `now`.

For each observed case, the model draws its offspring and their times and
keeps the draw only if exactly as many contacts fall by `now` as the table has
rows naming the case as their infector, which is the offspring law conditioned
on the cases seen. Later contacts are drawn from that draw, so an old case with
no known offspring has few left, and one that is isolated has none after its
isolation time. Where the model's own interventions would have isolated or
traced a case before `now`, they apply to it as to any case. They are not
conditioned on what was observed.

Only a [`BranchingProcess`](@ref) with a single [`Infectiousness`](@ref)
window and a single type is supported, bare or inside a [`ModelSpec`](@ref),
and its offspring must draw a number of contacts rather than their times.
Natural-history draws that the table gives no value for, such as the onset of
a case with no `onset_time`, are made from the model as usual.

```julia
using EpiBranch, Distributions, DataFrames, Random

model = BranchingProcess(Poisson(1.5), Gamma(2.5, 2.0))
cases = DataFrame(
    id = 1:4, parent_id = [missing, 1, 1, 2],
    infection_time = [0.0, 3.1, 4.6, 7.2],
)
state = initialise_state(model, cases; rng = Xoshiro(1))
simulate(model, state; max_time = 21.0)
```
"""
function initialise_state(
        model::BranchingProcess, cases::AbstractDataFrame;
        now::Union{Real, Nothing} = nothing, rng::AbstractRNG = Random.default_rng()
    )
    return _initialise_observed(
        model, cases; interventions = interventions(model),
        attributes = attributes(model), progression = progression(model),
        recorder = recorder(model), now, rng
    )
end

function initialise_state(
        spec::ModelSpec, cases::AbstractDataFrame;
        now::Union{Real, Nothing} = nothing, rng::AbstractRNG = Random.default_rng()
    )
    return _initialise_observed(
        spec.process, cases; interventions = spec.interventions,
        attributes = spec.attributes, progression = spec.progression,
        recorder = spec.recorder, now, rng
    )
end
