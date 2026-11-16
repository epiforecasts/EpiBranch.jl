# ── Case definitions ────────────────────────────────────────────────
#
# A case definition is a rule that reads what has happened to a person and
# returns their classification (confirmed, probable, suspected, or whatever a
# surveillance system uses). It is a layer of a `ModelSpec`, held beside the
# progression and the interventions, because the events it reads come from
# all of them: symptoms and a test from the progression, a trace from the
# interventions.
#
# The classification does not feed back into the simulation, and draws nothing
# from the generator the run uses. It is worked out
# once the race has finished, from the event times recorded on each person. A
# rule is evaluated at every time a relevant event was recorded and sees only
# the events at or before that time, as it would if evaluated as each event
# happened. Events recorded late in the run, tracing among them, reach the
# classification like any other.

"""
Parent type of a case definition: a rule that gives each infected person a
classification from the events recorded on them. Pass one (or a plain function
`history -> label`) as the `case_definition` of a [`ModelSpec`](@ref).

A new definition subtypes this and extends [`EpiBranch.classify_case`](@ref),
and optionally [`EpiBranch.relevant_events`](@ref). Built in:
[`ClinicalCaseDefinition`](@ref) and [`FixedProportionConfirmed`](@ref).
"""
abstract type AbstractCaseDefinition end

"""
    CaseHistory

What a case definition can see of one person at one time: the person
(`history.individual`), the time (`history.time`), and `history.draw`, a
uniform random number on `[0, 1)` drawn once for the person and the same at
every time the definition is evaluated, for a rule with a random element.

Read the events through [`event_time`](@ref) and [`has_happened`](@ref), which
ignore any event after `history.time`, and anything else recorded on the
person through `history.individual` (`history.individual.state[:test_result]`, `history.individual.state[:age]`).
"""
struct CaseHistory{I <: Individual, T <: Real}
    individual::I
    time::T
    draw::Float64
end

# The time an event was recorded as happening to the person, as the line list
# would show it: an isolation that does not count as a detection reads as no
# event.
function _recorded_event_time(ind::Individual, name::Symbol)
    name === :infection && return ind.infection_time
    t = _reported_state(ind, _resolved_key(name))
    return t isa Real ? t : NaN
end

"""
    event_time(history, event)

When `event` happened to the person, if it had by `history.time`, and `Inf`
if it had not (or never does). `event` is `:infection`, or a state named as in
a [`Transition`](@ref)'s `from`: `:onset` for the time recorded under
`:onset_time`, `:isolation`, `:trace`, `:outcome`, or any time an intervention
or progression step of your own records.
"""
function event_time(h::CaseHistory, event::Symbol)
    t = _recorded_event_time(h.individual, event)
    return isfinite(t) && t <= h.time ? t : oftype(float(h.time), Inf)
end

"""
    has_happened(history, event)

Whether `event` (named as in [`event_time`](@ref)) had happened to the person
by `history.time`.
"""
has_happened(h::CaseHistory, event::Symbol) = isfinite(event_time(h, event))

"""
    EpiBranch.classify_case(definition, history::CaseHistory)

The classification of the person `history` describes under `definition`: any
value, usually a `Symbol`, with `missing` for a person who is not (yet) a case
under the definition. The engine calls it at each time one of the
[`EpiBranch.relevant_events`](@ref) was recorded, in time order, and keeps the
times at which the answer changes. A function `history -> label` serves as a
definition without a type of its own.

```julia
struct SevereOnly <: EpiBranch.AbstractCaseDefinition end
function EpiBranch.classify_case(::SevereOnly, h)
    return has_happened(h, :hospitalised) ? :severe : missing
end
```
"""
function classify_case end

# Any callable serves as a definition. A type of its own that has no method of
# its own is told so, where calling it would report that it is not callable.
classify_case(f, h) = f(h)
function classify_case(d::AbstractCaseDefinition, h)
    throw(ArgumentError("`$(typeof(d))` needs a method of `EpiBranch.classify_case`."))
end

"""
    EpiBranch.relevant_events(definition)

The events a definition reads, named as in [`event_time`](@ref), or `nothing`
(the default) when it should be evaluated at every event recorded on the
person. Naming them saves evaluations and keeps a change in an event the rule
never reads from being logged as a classification time.
"""
relevant_events(::Any) = nothing

"""
    ClinicalCaseDefinition(; test = :test, symptoms = :onset, contact = :trace,
                           confirmed = :confirmed, probable = :probable,
                           suspected = :suspected)

The usual three-level definition. A person is `confirmed` once they have a
positive test result, `probable` once they have symptoms and a known contact
with a case, and `suspected` once they have symptoms alone. Someone with a
contact but no symptoms is not a case (`missing`) until they develop some.

The three events are named as in [`event_time`](@ref). A person's test counts
as positive when the `:<test>_result` that [`Testing`](@ref) records is true, and
its result is available at the time `test` names: by default `:test`, the
sampling time `Testing` records as `:test_time`. For a named step such as
`Testing(name = :pcr)`, use `test = :pcr`. A model with no `Testing` step
has no confirmed cases under this definition. The contact is
`:trace`, the time [`ContactTracing`](@ref) reached the person. Because the
definition is evaluated as events happen, a case is `suspected` at onset,
becomes `probable` if tracing reaches them later, and `confirmed` at a
positive result.

# Examples

```julia
assay = Assay(value = load -> 40.0 - 3.3 * load, detect = 2.0)
model = ModelSpec(
    BranchingProcess(NegBin(2.0, 0.5), Gamma(2.0, 3.0));
    attributes = [
        clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
        within_host(acute),
    ],
    progression = [Testing(assay = assay, delay = Gamma(2.0, 1.0))],
    case_definition = ClinicalCaseDefinition(),
)
state = simulate(model; max_cases = 200)
case_classification_changes(state)
```
"""
struct ClinicalCaseDefinition <: AbstractCaseDefinition
    test::Symbol
    symptoms::Symbol
    contact::Symbol
    confirmed::Symbol
    probable::Symbol
    suspected::Symbol
end

function ClinicalCaseDefinition(;
        test::Symbol = :test, symptoms::Symbol = :onset,
        contact::Symbol = :trace, confirmed::Symbol = :confirmed,
        probable::Symbol = :probable, suspected::Symbol = :suspected
    )
    return ClinicalCaseDefinition(test, symptoms, contact, confirmed, probable, suspected)
end

function classify_case(def::ClinicalCaseDefinition, h::CaseHistory)
    result = Symbol(replace(String(def.test), r"_time$" => ""), :_result)
    get(h.individual.state, result, false)::Bool && has_happened(h, def.test) &&
        return def.confirmed
    has_happened(h, def.symptoms) || return missing
    return has_happened(h, def.contact) ? def.probable : def.suspected
end

relevant_events(def::ClinicalCaseDefinition) = (def.test, def.symptoms, def.contact)

"""
    FixedProportionConfirmed(proportion; from = :onset, confirmed = :confirmed,
                             otherwise = :probable)

A definition that classifies a fixed `proportion` of cases as `confirmed` and
the rest as `otherwise`, from the time the event `from` (named as in
[`event_time`](@ref)) happens. Each case is drawn once, so its classification
does not change after that.
"""
struct FixedProportionConfirmed <: AbstractCaseDefinition
    proportion::Float64
    from::Symbol
    confirmed::Symbol
    otherwise::Symbol
    function FixedProportionConfirmed(
            proportion::Real; from::Symbol = :onset,
            confirmed::Symbol = :confirmed, otherwise::Symbol = :probable
        )
        0 <= proportion <= 1 || throw(
            ArgumentError("proportion must be between 0 and 1, got $proportion")
        )
        return new(proportion, from, confirmed, otherwise)
    end
end

function classify_case(def::FixedProportionConfirmed, h::CaseHistory)
    has_happened(h, def.from) || return missing
    return h.draw < def.proportion ? def.confirmed : def.otherwise
end

relevant_events(def::FixedProportionConfirmed) = (def.from,)

# ── Applying a definition ───────────────────────────────────────────

const _CLASSIFICATION_KEY = :case_classification
const _CLASSIFICATION_CHANGES_KEY = :_case_classification_changes
const _CLASSIFICATION_DRAW_KEY = :_case_definition_draw
const _CLASSIFICATION_SEED_KEY = :_case_definition_seed
const _CLASSIFICATION_HORIZON_KEY = :_case_definition_horizon

# The times at which to evaluate the definition on one person: their
# infection, and each later time up to `horizon` an event the definition reads
# was recorded. A person's progression is drawn when they are infected, so
# events can be recorded past a `max_time` cut; the race did not simulate what
# would have followed them, tracing among it, so they are left out.
function _evaluation_times(def, ind::Individual, horizon = Inf)
    t0 = ind.infection_time
    times = typeof(t0)[t0]
    events = relevant_events(def)
    keys_ = events === nothing ?
        (k for k in keys(ind.state) if endswith(String(k), "_time") && !startswith(String(k), "_")) :
        (_resolved_key(e) for e in events if e !== :infection)
    for key in keys_
        t = _reported_state(ind, key)
        t isa Real && isfinite(t) && t0 < t <= horizon && push!(times, t)
    end
    return sort!(unique!(times))
end

function _classify_individual!(def, ind::Individual, state, horizon = Inf)
    delete!(ind.state, _CLASSIFICATION_KEY)
    delete!(ind.state, _CLASSIFICATION_CHANGES_KEY)
    # One draw per person, kept so that re-evaluating a continued run does not
    # reclassify a case already classified.
    draw = get!(ind.state, _CLASSIFICATION_DRAW_KEY) do
        rand(_classification_rng(state.rng, state, ind.id))
    end::Float64
    changes = Tuple{typeof(ind.infection_time), Any}[]
    current = missing
    for t in _evaluation_times(def, ind, horizon)
        label = classify_case(def, CaseHistory(ind, t, draw))
        isequal(label, current) && continue
        push!(changes, (t, label))
        current = label
    end
    isempty(changes) && return nothing
    ind.state[_CLASSIFICATION_CHANGES_KEY] = changes
    # A person classified and then declassified ends as `missing`, which the
    # line list leaves blank.
    ind.state[_CLASSIFICATION_KEY] = current
    return nothing
end

# The generator a person's classification draw comes from. It never draws
# from the run's own generator, which the user's `rng` is, so that adding a
# definition leaves later runs of a batch and a continued run unchanged.
# `CommonRandomNumbers` already keeps a stream per person and purpose. Any
# other generator gives a seed, taken once from a copy and kept with the
# state, from which each person's generator is derived. A generator that
# cannot be copied, such as `RandomDevice`, has no sequence to protect and
# gives the seed directly.
_classification_rng(rng::CommonRandomNumbers, state, id) =
    stream(rng, id, :case_definition)
function _classification_rng(rng::Random.AbstractRNG, state, id)
    seed = get!(state.scratch, _CLASSIFICATION_SEED_KEY) do
        rand(applicable(copy, rng) ? copy(rng) : rng, UInt64)
    end
    return Random.Xoshiro(hash((seed::UInt64, id)))
end

# No definition: nothing to do. A definition that is a function or an
# `AbstractCaseDefinition` classifies every infected person.
# `horizon` is the run's `max_time`, `Inf` when it had none.
_classify_cases!(state::SimulationState, ::Nothing, horizon = Inf) = state
function _classify_cases!(state::SimulationState, def, horizon = Inf)
    state.scratch[_CLASSIFICATION_HORIZON_KEY] = horizon
    for ind in state.individuals
        is_infected(ind) && _classify_individual!(def, ind, state, horizon)
    end
    return state
end

"""
    case_classification(individual)

The classification the model's case definition gives the person at the end of
the run (`missing` if none): at `max_time` when the run was given one, from the
events recorded up to then, and otherwise once every recorded event has
happened. See [`ModelSpec`](@ref)'s `case_definition`.
"""
case_classification(ind::Individual) = get(ind.state, _CLASSIFICATION_KEY, missing)

"""
    case_classification_changes(state::SimulationState; reference_date = Date(2020, 1, 1))

Every change of classification in a simulated outbreak, in order of time, as a
DataFrame with one row per change: `id` of the case, `time` (days since the
start of the outbreak), `date` (the same counted from `reference_date`) and
`classification` (`missing` where a case stopped being one under the
definition). A case that is probable at onset and confirmed at a positive
result has two rows. Empty if the model has no `case_definition`.
"""
function case_classification_changes(
        state::SimulationState; reference_date::Date = Date(2020, 1, 1)
    )
    rows = NamedTuple{(:id, :time, :date, :classification)}[]
    for ind in state.individuals
        for (t, label) in get(ind.state, _CLASSIFICATION_CHANGES_KEY, ())
            push!(rows, (; id = ind.id, time = t, date = _to_date(reference_date, t), classification = label))
        end
    end
    sort!(rows; by = r -> (r.time, r.id))
    return DataFrame(
        id = [r.id for r in rows], time = [r.time for r in rows],
        date = Union{Date, Missing}[r.date for r in rows],
        classification = [r.classification for r in rows]
    )
end
