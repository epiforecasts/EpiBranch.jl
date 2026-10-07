# ── Individual show + state accessors ───────────────────────────────
# Typed read/write helpers over `Individual.state`. Each is a small
# wrapper around `get(ind.state, ..., default)::T` so that the rest of
# the codebase doesn't sprinkle that pattern everywhere.

function Base.show(io::IO, ind::Individual)
    infected_str = is_infected(ind) ? "infected" : "contact-only"
    isolated_str = is_isolated(ind) ? ", isolated" : ""
    return print(
        io,
        "Individual(id=$(ind.id), gen=$(ind.generation), chain=$(ind.chain_id), t=$(round(ind.infection_time, digits = 1)), $(infected_str)$(isolated_str))"
    )
end

"""
    onset_time(ind)

Time of symptom onset, in days since the start of the outbreak (`NaN` if the
person is asymptomatic or has no onset time).

The functions that read a person's information, such as `onset_time`,
[`is_isolated`](@ref) or [`is_infected`](@ref), work on the people in a
simulated outbreak:

```julia
state = simulate(model)
count(is_isolated, state.individuals)                 # number isolated
onsets = [onset_time(i) for i in state.individuals if is_infected(i)]
```
"""
function onset_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :onset_time, T(NaN)))::T
end

"""
    incubation_period(ind)

Incubation period: days from infection to symptom onset (`NaN` if the person
is asymptomatic or has no onset time). Useful inside a `generation_time`
function that links a case's generation time to its own incubation period.
"""
incubation_period(ind::Individual) = onset_time(ind) - ind.infection_time

"""
    outcome_time(ind)

Time of the case's outcome, such as recovery or death, in days since the
start of the outbreak: the earliest step of the natural history that ends the
case (see [`Transition`](@ref)). `Inf` if there is none yet, or if the
natural history has no such step.
"""
function outcome_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :outcome_time, T(Inf)))::T
end

"""Whether the person was isolated in a way that counts as detecting them, so
that tracing and group vaccination start from them and the line list shows
it. An isolation that does not count as a detection (see
[`EpiBranch.records_isolation`](@ref)), such as one after death, still stops
transmission from [`isolation_time`](@ref) but leaves this `false`."""
is_isolated(ind::Individual) = _isolation_in_force(ind) && !_isolation_unrecorded(ind)

"""Time from which isolation or quarantine stops the person transmitting, in
days since the start of the outbreak (`Inf` if never), whether or not it
counts as a detection (see [`is_isolated`](@ref))."""
function isolation_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :isolation_time, T(Inf)))::T
end

"""Time at which the isolation or quarantine that began at
[`isolation_time`](@ref) ends, in days since the start of the outbreak (`Inf`
if it never does). It is finite when [`Isolation`](@ref) or the
[`Quarantine`](@ref) of [`ContactTracing`](@ref) has a duration, so a
quarantine that ended before the person was later infected does not stop
their own onward transmission. Set by [`set_isolated!`](@ref)."""
function isolation_release_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :isolation_release_time, T(Inf)))::T
end


# Whether an isolation or quarantine stands on the individual, recorded or not.
# Interventions layering one isolation over another read this.
_isolation_in_force(ind::Individual) = get(ind.state, :isolated, false)::Bool

# Whether the standing isolation removes the case from transmission without
# counting as a detection.
_isolation_unrecorded(ind::Individual) = get(ind.state, :_isolation_unrecorded, false)::Bool

# The time a detection reader sees: the isolation time, or `Inf` for an
# isolation that is not recorded.
function _recorded_isolation_time(ind::Individual{T}) where {T}
    return _isolation_unrecorded(ind) ? T(Inf) : isolation_time(ind)
end

"""Whether the person was found by contact tracing."""
is_traced(ind::Individual) = get(ind.state, :traced, false)::Bool

"""Whether the person was quarantined as a traced contact."""
is_quarantined(ind::Individual) = get(ind.state, :quarantined, false)::Bool

"""Whether the person received the vaccine dose named `dose_label` (by
default, the single dose of a vaccination without a `dose_label`)."""
function is_vaccinated(ind::Individual; dose_label::Symbol = :default)
    return get(ind.state, _vaccinated_key(dose_label), false)::Bool
end

"""Time at which immunity from the vaccine dose named `dose_label` develops,
in days since the start of the outbreak: the vaccination time plus the
dose's `delay_to_immunity` (`Inf` if not vaccinated). A natural-history step
can compare it with the time an outcome would take effect (for example
`onset_time(ind)` for the default [`Death`](@ref)) to decide whether the
dose's [`severity_efficacy`](@ref) applies: immunity that develops later
gives no protection."""
function immunity_time(ind::Individual{T}; dose_label::Symbol = :default) where {T}
    return convert(T, get(ind.state, _immunity_time_key(dose_label), T(Inf)))::T
end

"""Vaccine efficacy against severe outcomes for this person: the
proportional reduction in, for example, their probability of death once their
immunity has developed (0 if not vaccinated, or if the dose has no effect on
severity). Drawn once per vaccinated person from the `severity_efficacy` of
the dose's [`VaccineEffect`](@ref). Use it in the `probability` of a
natural-history step, once [`immunity_time`](@ref) has passed:

```julia
Death(delay = LogNormal(2.5, 0.4),
      probability = (rng, ind) ->
          immunity_time(ind) <= onset_time(ind) ?
              0.7 * (1 - severity_efficacy(ind)) : 0.7)
```
"""
function severity_efficacy(ind::Individual; dose_label::Symbol = :default)
    return get(ind.state, _severity_efficacy_key(dose_label), 0.0)::Float64
end

"""Whether the case never develops symptoms."""
is_asymptomatic(ind::Individual) = get(ind.state, :asymptomatic, false)::Bool

"""
    infection_aborted_time(ind)

Time at which the person's infection was stopped before symptom onset, for
example by post-exposure vaccination, in days since the start of the outbreak
(`Inf` if it was not). Set by
[`abort_infection!`](@ref EpiBranch.abort_infection!).
"""
function infection_aborted_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :infection_aborted_time, T(Inf)))::T
end

"""
    abort_infection!(ind, time)

Stop the person's infection at `time` (days), before symptom onset, as
post-exposure prophylaxis does; [`RingVaccination`](@ref) uses it for
post-exposure vaccination. For intervention authors: call it from an
intervention once the person has an infection time. If it is called more than
once, the earliest time counts.

In every transmission model the person stays a case but infects nobody from
`time` on, even after the intervention that stopped the infection is no longer
active, and all their transmission routes end then. They have no symptom onset
(`:onset_time` is `NaN`, `:asymptomatic` stays `false`), so nothing that
starts at onset happens. Natural-history steps that would take effect at or
after `time` are undone (see
[`resolve_transitions!`](@ref EpiBranch.resolve_transitions!)).

In a branching process, an intervention acting on contacts before it is
decided whether they are infected (in `apply_post_transmission!`) sees each
contact's earliest exposure time as its provisional infection time. If the
contact turns out not to be infected, or to be infected only at or after
`time`, the stop has no effect and the onset is restored.

Raises an error unless `time` is after the infection time and, for a person
with an incubation period, before symptom onset.
"""
function abort_infection!(ind::Individual, time::Real)
    ind.infection_time < time || throw(
        ArgumentError(
            "an infection can only be aborted after it starts (infection time " *
                "$(ind.infection_time), abort time $time)"
        )
    )
    incubation = get(ind.state, :incubation_period, NaN)
    isnan(incubation) || time < ind.infection_time + incubation || throw(
        ArgumentError(
            "an infection can only be aborted before symptom onset (onset " *
                "$(ind.infection_time + incubation), abort time $time)"
        )
    )
    ind.state[:infection_aborted_time] = min(infection_aborted_time(ind), time)
    _set_onset_from_incubation!(ind)
    return nothing
end

_infection_aborted(ind::Individual) = haskey(ind.state, :infection_aborted_time)

"""Whether the person develops symptoms: they are not asymptomatic and their
infection was not stopped before onset."""
_develops_symptoms(ind::Individual) = !is_asymptomatic(ind) && !_infection_aborted(ind)

"""Whether the case tested positive, as drawn by [`Isolation`](@ref) from its
`test_sensitivity`."""
is_test_positive(ind::Individual) = get(ind.state, :test_positive, false)::Bool

"""Whether the person was infected, as opposed to an exposed contact who
escaped infection."""
is_infected(ind::Individual) = get(ind.state, :infected, true)::Bool

"""
Time from which a person who has been infected can be infected again,
because their immunity has waned, in days since the start of the outbreak
(`Inf`, never, by default). Until then [`EpiBranch.HostImmunity`](@ref)
prevents reinfection. It matters only for a model that exposes people again
after infection through its own [`contacts_of`](@ref); none of the built-in
models does.

Set it with a step in the model's `progression`, timed from the state that
starts waning, for example
`Transition(:susceptible_again, from = :recovered, delay = Exponential(180))`."""
function susceptible_again_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :susceptible_again_time, T(Inf)))::T
end

"""
    is_settled(state, ind) -> Bool

For intervention authors, in the network and household models: whether
`ind`'s infection, and what interventions do to them, is final for this run,
so nothing found later will change it. `false` for a person whose infection
is still undecided, and for the case currently being processed (so its own
interventions can still be revised)."""
is_settled(state, ind::Individual) = get(ind.state, :_settled, false)::Bool

"""Which type a person is in a multi-type branching process, as a number
(1 in a single-type model)."""
individual_type(ind::Individual) = get(ind.state, :type, 1)::Int

# The key `set_isolated!` records a removal's history under. A component that
# removes a host through a path of its own records under its own key instead,
# and names it from `removal_gap_host_times`.
const REMOVAL_STRETCHES_KEY = :_removal_stretches

const _NO_STRETCHES = Tuple{Float64, Float64}[]

"""Record that a person is isolated from `time` (days) until `release_time`.
Used by interventions that isolate or quarantine people. `release_time` is
required: pass `Inf` for isolation that never ends, so that indefinite
isolation is always a stated choice.

A transmission route that isolation should end lists
[`EpiBranch.INTERVENTION_REMOVAL`](@ref) in its `until`, which also allows
for leaky isolation. `:isolated` in an `until` instead refers to a
`Transition(:isolated, …)` step in the natural history."""
function set_isolated!(ind::Individual, time::Real; release_time::Real)
    ind.state[:isolated] = true
    delete!(ind.state, :_isolation_unrecorded)
    ind.state[:isolation_time] = time
    ind.state[:isolation_release_time] = release_time
    record_removal!(ind, time, release_time)
    return release_time
end

"""
    record_removal!(ind, start, release; key = :_removal_stretches)

Record that `ind` could not transmit from `start` until `release` (days;
`Inf` if never released), for example while isolated or quarantined.
[`set_isolated!`](@ref) records under the default key, shared by isolation
and quarantine.

An intervention of your own that removes people in its own way passes its own
`key`, and names it in
[`removal_gap_host_times`](@ref EpiBranch.removal_gap_host_times). The
likelihood then leaves out the same periods of exposure that the simulation
blocked, so that `simulate` and `loglikelihood` agree.
"""
function record_removal!(
        ind::Individual, start::Real, release::Real;
        key::Symbol = REMOVAL_STRETCHES_KEY
    )
    (isfinite(start) && release > start) || return nothing
    # The stretch keeps the individual's own number type, so that an AD dual
    # isolation time flows through as the time accessors promise.
    stretch = promote(start, release)
    stretches = get!(() -> typeof(stretch)[], ind.state, key)
    push!(stretches, stretch)
    sort!(stretches; by = first)
    ind.state[key] = _merge_stretches(stretches)
    return nothing
end

# Overlapping and touching stretches folded into disjoint ones, so that no
# stretch counts twice. `stretches` must already be sorted by its starts.
function _merge_stretches(stretches)
    merged = similar(stretches, 0)
    for (a, b) in stretches
        if !isempty(merged) && a <= last(merged)[2]
            merged[end] = (last(merged)[1], max(last(merged)[2], b))
        else
            push!(merged, (a, b))
        end
    end
    return merged
end

"""
    removal_stretches(ind, key = :_removal_stretches)

Every period during which `ind` could not transmit, such as isolation or
quarantine, as sorted, non-overlapping `(start, release)` pairs in days; a
release of `Inf` means it never ended. Recorded by
[`record_removal!`](@ref EpiBranch.record_removal!).

[`isolation_time`](@ref) and [`isolation_release_time`](@ref) give only the
current isolation. This gives the whole history, which the likelihood needs:
for example a person quarantined, released, and isolated again later. Do not
change the returned vector.
"""
function removal_stretches(ind::Individual, key::Symbol = REMOVAL_STRETCHES_KEY)
    return get(ind.state, key, _NO_STRETCHES)
end

# The earliest removal of this host that never releases it. Such a removal
# closes the infectious window, which is where a likelihood takes it out of
# the exposure, so it is read apart from the stretches that lapse. The merge
# leaves at most one of them.
function permanent_removal_time(ind::Individual, key::Symbol = REMOVAL_STRETCHES_KEY)
    t = Inf
    for (start, release) in removal_stretches(ind, key)
        isfinite(release) || (t = min(t, start))
    end
    return t
end

"""Undo a person's isolation, reversing [`set_isolated!`](@ref)."""
function clear_isolated!(ind::Individual)
    ind.state[:isolated] = false
    delete!(ind.state, :_isolation_unrecorded)
    ind.state[:isolation_time] = Inf
    ind.state[:isolation_release_time] = Inf
    delete!(ind.state, REMOVAL_STRETCHES_KEY)
    return nothing
end
