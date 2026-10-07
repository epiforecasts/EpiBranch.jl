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

"""Symptom onset time (`NaN` if asymptomatic or not set); a dual under AD."""
function onset_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :onset_time, T(NaN)))::T
end

"""
Incubation period: time from infection to symptom onset (Float64, NaN if
asymptomatic or onset is not set). Useful inside a `generation_time`
function that links an individual's generation time to their own
incubation period.
"""
incubation_period(ind::Individual) = onset_time(ind) - ind.infection_time

"""
Time of the individual's terminal outcome — the earliest terminal
[`Transition`](@ref) to occur, e.g. recovery or death (`Inf` if none has
occurred, whether because the case is still ongoing or the progression has no
terminal transition); a dual under AD.
"""
function outcome_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :outcome_time, T(Inf)))::T
end

"""Whether the individual is recorded as isolated, which is what tracing,
group vaccination and the line list read as a detection. An isolation that
[`Isolation`](@ref) does not record (see
[`EpiBranch.records_isolation`](@ref)) still removes the case from
transmission at [`isolation_time`](@ref) but leaves this `false`."""
is_isolated(ind::Individual) = _isolation_in_force(ind) && !_isolation_unrecorded(ind)

"""Time from which isolation or quarantine removes the individual from
transmission (`Inf` if never), whether or not the isolation is recorded as a
detection (see [`is_isolated`](@ref)); a dual under AD."""
function isolation_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :isolation_time, T(Inf)))::T
end

"""Time from which the isolation block begun at [`isolation_time`](@ref)
lapses (`Inf` if it never does); a dual under AD. Set alongside `isolation_time` by
[`set_isolated!`](@ref); a duration configured on [`Isolation`](@ref) or
[`ContactTracing`](@ref)'s [`Quarantine`](@ref) action gives this a finite
value, so a quarantine that ended before a later, unrelated infection no
longer blocks that case's own onward transmission."""
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

"""Whether the individual was traced via contact tracing."""
is_traced(ind::Individual) = get(ind.state, :traced, false)::Bool

"""Whether the individual is quarantined."""
is_quarantined(ind::Individual) = get(ind.state, :quarantined, false)::Bool

"""Whether the individual is vaccinated under the given `dose_label`. The
default label reads the plain `:vaccinated` key; a non-default label reads the
namespaced key an `AbstractVaccination` with that `dose_label` writes."""
function is_vaccinated(ind::Individual; dose_label::Symbol = :default)
    return get(ind.state, _vaccinated_key(dose_label), false)::Bool
end

"""Time the individual's vaccine-induced immunity develops under the given
`dose_label` (`Inf` if not vaccinated); a dual under AD. This is
`vaccination_time + delay_to_immunity`, recorded by [`AbstractVaccination`](@ref)
at vaccination time so a clinical transition can check it without reaching
for the vaccination object, which it never sees. Compare against the time an
outcome would take effect (e.g. `onset_time(ind)` for the default `Death`) to
decide whether that dose's [`severity_efficacy`](@ref) applies: a dose whose
immunity develops after that time confers no protection."""
function immunity_time(ind::Individual{T}; dose_label::Symbol = :default) where {T}
    return convert(T, get(ind.state, _immunity_time_key(dose_label), T(Inf)))::T
end

"""Full-strength efficacy of the individual's dose against infection under the
given `dose_label`, as sampled when the dose was given — a per-exposure block
probability under `LeakyMode`, a responder status of `1.0` or `0.0` under
`AllOrNothingMode` — or `nothing` if no dose with that label has been
recorded. Unlike [`severity_efficacy`](@ref), there is no default of `0.0`:
`nothing` is what distinguishes an individual with no such dose from one
given a dose with `efficacy = 0.0`."""
function vaccine_efficacy(ind::Individual; dose_label::Symbol = :default)
    return get(ind.state, _vaccine_efficacy_key(dose_label), nothing)
end

"""Probability that the individual's own disease course is milder — e.g. a
lower chance of death — once their vaccine-induced immunity has developed
(`0.0` if not vaccinated, or if the dose carries no severity effect). Sampled
once per vaccinated individual, alongside `efficacy`, from the
`severity_efficacy` of the dose's [`VaccineEffect`](@ref). Read this from a
clinical transition's `probability`, gated on [`immunity_time`](@ref) having
passed:

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

"""Whether the individual is asymptomatic."""
is_asymptomatic(ind::Individual) = get(ind.state, :asymptomatic, false)::Bool

"""
    infection_aborted_time(ind)

Time at which the individual's infection was aborted before symptom onset
(`Inf` if it was not); a dual under AD. Recorded by
[`abort_infection!`](@ref EpiBranch.abort_infection!).
"""
function infection_aborted_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :infection_aborted_time, T(Inf)))::T
end

"""
    abort_infection!(ind, time)

End the individual's infection at `time`, before symptom onset, as a
post-exposure treatment would. For intervention authors: call it from any hook
once the individual has an infection time. Several aborts keep the earliest.

The engine then treats the infection as ended at `time` on every transmission
model: the individual stays a case but transmits nothing from `time` on, also
after the intervention that aborted it stops being active, and every route
window closes there. It has no onset (`:onset_time` is `NaN` while
`:asymptomatic` stays `false`), and nothing triggered by onset happens. Any
clinical transition that would take effect at or after `time` is undone (see
[`resolve_transitions!`](@ref EpiBranch.resolve_transitions!)).

On the generation-based engine an intervention acting before infection is
resolved, in `apply_post_transmission!`, sees each contact's provisional
infection time, its earliest exposure. If resolution leaves the contact
uninfected, or infected at or after `time`, the abort did not end that
infection and the engine discards it, restoring the onset.

Throws an `ArgumentError` unless `time` falls after the infection time and,
for an individual with a finite `:incubation_period`, before its onset.
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

"""Whether the individual develops symptoms: it is not asymptomatic and its
infection was not aborted before onset."""
_develops_symptoms(ind::Individual) = !is_asymptomatic(ind) && !_infection_aborted(ind)

"""Whether the individual tested positive."""
is_test_positive(ind::Individual) = get(ind.state, :test_positive, false)::Bool

"""Whether the individual was successfully infected (vs contact only)."""
is_infected(ind::Individual) = get(ind.state, :infected, true)::Bool

"""
Time at which a resolved infection's immunity has waned enough for the host
to be at risk of a new one (`Inf` if never, the default); a dual under AD.
Read by [`EpiBranch.HostImmunity`](@ref), the built-in risk source that keeps
an already-infected host out of reach of a new infection until this time, for
any model whose [`contacts_of`](@ref) offers one as a candidate contact
again.

Set it the same way `:infectious_time` or a progression's other `_time` keys
are set: list a [`Transition`](@ref) into a waned state in the model's
progression, timed from the state that starts the clock, e.g.
`Transition(:susceptible_again, from = :recovered, delay = Exponential(180))`
writes `:susceptible_again_time`, which this reads."""
function susceptible_again_time(ind::Individual{T}) where {T}
    return convert(T, get(ind.state, :susceptible_again_time, T(Inf)))::T
end

"""
    is_settled(state, ind) -> Bool

Whether `ind`'s own fate in a continuous-time race is already fixed, so that
no later discovery on this run will revisit it. Set by
`_apply_continuous_actions!` once a case's own round of action discovery has
run; `false` for a case still pending, and for the case currently being
settled during its own round (letting that round still revise an action it
had already admitted). An intervention consults this instead of
reconstructing the settled/pending distinction from the candidate list the
race handed it."""
is_settled(state, ind::Individual) = get(ind.state, :_settled, false)::Bool

"""Type index for multi-type branching processes (default 1)."""
individual_type(ind::Individual) = get(ind.state, :type, 1)::Int

# The key `set_isolated!` records a removal's history under. A component that
# removes a host through a path of its own records under its own key instead,
# and names it from `removal_gap_host_times`.
const REMOVAL_STRETCHES_KEY = :_removal_stretches

const _NO_STRETCHES = Tuple{Float64, Float64}[]

"""Mark an individual as isolated at the given time (any `Real`, so an AD
dual isolation time flows through), with the `release_time` from which the
block lapses. `release_time` is required: a removal that never releases its
host is a choice to state, `Inf` saying so, and no policy should inherit it
silently.

The time is stored under `:isolation_time`, the release under
`:isolation_release_time`. A route window that isolation should end lists
[`EpiBranch.INTERVENTION_REMOVAL`](@ref) in its `until`, which respects leaky
isolation. `:isolated` in an `until` refers to a `Transition(:isolated, …)`
in the natural history."""
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

Record that a removal took `ind` out of transmission from `start` until
`release`, which is `Inf` for one that never releases it.
[`set_isolated!`](@ref) records under the reserved key, which both built-in
removals share.

A removal of your own that keeps its own history passes its own `key` and
names that key from
[`removal_gap_host_times`](@ref EpiBranch.removal_gap_host_times). A
likelihood then takes the same stretches out of each pair's exposure as the
simulator blocked, which is what keeps `simulate` and `loglikelihood` in
agreement.
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

Every stretch a removal has taken `ind` out of transmission for, as sorted
disjoint `(start, release)` pairs, a release of `Inf` standing for a removal
that never ends. Recorded by
[`record_removal!`](@ref EpiBranch.record_removal!), which
[`set_isolated!`](@ref) calls with the reserved key.

[`isolation_time`](@ref) and [`isolation_release_time`](@ref) hold the removal
in force, which is what a detection reads; this holds the history, which is
what a likelihood needs, since one pair of times cannot say that a host was
quarantined, released, and isolated again later. Treat the returned vector as
read-only.
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

"""Clear an individual's isolation, the inverse of [`set_isolated!`](@ref)."""
function clear_isolated!(ind::Individual)
    ind.state[:isolated] = false
    delete!(ind.state, :_isolation_unrecorded)
    ind.state[:isolation_time] = Inf
    ind.state[:isolation_release_time] = Inf
    delete!(ind.state, REMOVAL_STRETCHES_KEY)
    return nothing
end
