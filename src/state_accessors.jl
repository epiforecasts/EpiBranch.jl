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
    return _state_real(ind.state, :onset_time, T(NaN))
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
    return _state_real(ind.state, :outcome_time, T(Inf))
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
    return _state_real(ind.state, :isolation_time, T(Inf))
end

"""Time at which the isolation or quarantine that began at
[`isolation_time`](@ref) ends, in days since the start of the outbreak (`Inf`
if it never does). It is finite when [`Isolation`](@ref) or the
[`Quarantine`](@ref) of [`ContactTracing`](@ref) has a duration, so a
quarantine that ended before the person was later infected does not stop
their own onward transmission. Set by [`set_isolated!`](@ref)."""
function isolation_release_time(ind::Individual{T}) where {T}
    return _state_real(ind.state, :isolation_release_time, T(Inf))
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
    return _state_real(ind.state, _immunity_time_key(dose_label), T(Inf))
end

"""The full-strength efficacy against infection drawn for this person's dose
named `dose_label`, as it was sampled when the dose was given: the probability
that an exposure is blocked under `LeakyMode`, or `1.0` or `0.0` for whether
they responded under `AllOrNothingMode`. `nothing` if no dose with that label
was recorded, which is what tells a person with no such dose apart from one
given a dose of `efficacy = 0.0`. [`severity_efficacy`](@ref) draws no such
distinction: it returns `0.0` for a person who was never dosed."""
function vaccine_efficacy(ind::Individual; dose_label::Symbol = :default)
    return get(ind.state, _vaccine_efficacy_key(dose_label), nothing)
end

"""Vaccine efficacy against severe outcomes for this person: the
proportional reduction in, for example, their probability of death once their
immunity has developed (0 if not vaccinated, or if the dose has no effect on
severity). Drawn once per vaccinated person, alongside `efficacy`, from the
`severity_efficacy` of the dose's [`VaccineEffect`](@ref). Use it in the
`probability` of a natural-history step, once [`immunity_time`](@ref) has
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
    return _state_real(ind.state, :infection_aborted_time, T(Inf))
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
decided whether they are infected (in [`trace_contacts!`](@ref
EpiBranch.trace_contacts!), or in an action it has admitted) sees each
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
    incubation = _state_real(ind.state, :incubation_period, NaN)
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
    return _state_real(ind.state, :susceptible_again_time, T(Inf))
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

# The key a removal that only blocks a fraction of the hazard (see
# `removal_block_probability`) records its stretches under, apart from
# `REMOVAL_STRETCHES_KEY`'s complete ones, so a likelihood never mistakes one
# for the other. `Isolation` writes here for itself when leaky.
const LEAKY_REMOVAL_STRETCHES_KEY = :_leaky_removal_stretches

const _NO_STRETCHES = Tuple{Float64, Float64}[]

"""Record that a person is isolated from `time` (days) until `release_time`.
Used by interventions that isolate or quarantine people. `release_time` is
required: pass `Inf` for isolation that never ends, so that indefinite
isolation is always a stated choice.

A transmission route that isolation should end lists
[`EpiBranch.INTERVENTION_REMOVAL`](@ref) in its `until`, which also allows
for leaky isolation. `:isolated` in an `until` instead refers to a
`Transition(:isolated, …)` step in the natural history.

`key` names the [`record_removal!`](@ref EpiBranch.record_removal!) key the
stretch goes under, the shared default key by default. A removal that only
blocks a fraction of the hazard passes its own key instead, so its stretch is
never mistaken for a complete removal's merely because some other component in
the model happens to share the default key."""
function set_isolated!(
        ind::Individual, time::Real;
        release_time::Real, key::Symbol = REMOVAL_STRETCHES_KEY
    )
    ind.state[:isolated] = true
    delete!(ind.state, :_isolation_unrecorded)
    ind.state[:isolation_time] = time
    ind.state[:isolation_release_time] = release_time
    record_removal!(ind, time, release_time; key = key)
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
blocked, so that `simulate` and `loglikelihood` agree. Every `key` that ever
records a stretch with a finite `release` here is checked against
[`removal_gap_host_times`](@ref EpiBranch.removal_gap_host_times) when the
infection layer is built, and the likelihood stops with an error for one that
no component names or withdraws.
"""
function record_removal!(
        ind::Individual{T}, start::Real, release::Real;
        key::Symbol = REMOVAL_STRETCHES_KEY
    ) where {T}
    (isfinite(start) && release > start) || return nothing
    # The stretch keeps the individual's own number type, so that an AD dual
    # isolation time flows through as the time accessors promise.
    S = promote_type(T, typeof(start), typeof(release))
    stretch = (convert(S, start), convert(S, release))
    old = get(ind.state, key, nothing)
    # A fresh vector each time rather than one grown in place: an archived
    # episode holds a copy of the state that shares the old one.
    ind.state[key] = old === nothing ? [stretch] : _with_stretch(old, stretch)
    # Only a stretch that releases its host needs a named key: one that never
    # does closes the infectious window instead (see `infectious_removal_time`),
    # which the likelihood already reads with no key of its own.
    isfinite(release) && _note_lapsing_removal_key!(ind, key)
    return nothing
end

# The keys under which `ind` has had at least one releasing stretch recorded,
# noted here so the likelihood can later tell a key nobody named from one
# that simply never lapsed (see `removal_gap_host_times_withdrawn`).
function _note_lapsing_removal_key!(ind::Individual, key::Symbol)
    noted = get(ind.state, :_lapsing_removal_keys, ())::Tuple{Vararg{Symbol}}
    key in noted || (ind.state[:_lapsing_removal_keys] = (noted..., key))
    return nothing
end

# Every key noted by `_note_lapsing_removal_key!` across `state`'s
# individuals, in first-seen order.
function _lapsing_removal_keys(state::SimulationState)
    keys = Symbol[]
    for ind in state.individuals, key in get(ind.state, :_lapsing_removal_keys, ())
        key in keys || push!(keys, key)
    end
    return keys
end

function _with_stretch(old::Vector{E}, stretch) where {E}
    new = Vector{E}(undef, length(old) + 1)
    copyto!(new, old)
    new[end] = stretch
    sort!(new; by = first)
    return _merge_stretches!(new)
end
_with_stretch(old, stretch) = _merge_stretches(sort!(push!(copy(old), stretch); by = first))

# Overlapping and touching stretches folded into disjoint ones, so that no
# stretch counts twice. `stretches` must already be sorted by its starts.
_merge_stretches(stretches) = _merge_stretches!(copy(stretches))

function _merge_stretches!(stretches::AbstractVector)
    n = 0
    for (a, b) in stretches
        if n > 0 && a <= stretches[n][2]
            stretches[n] = (stretches[n][1], max(stretches[n][2], b))
        else
            n += 1
            stretches[n] = (a, b)
        end
    end
    return resize!(stretches, n)
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
    # Leaky isolation's own record, apart from the shared history above; only
    # `Isolation` ever writes it, so clearing it unconditionally is safe.
    delete!(ind.state, LEAKY_REMOVAL_STRETCHES_KEY)
    return nothing
end
