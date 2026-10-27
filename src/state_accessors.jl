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

# Element types of the two handle-tracked entry vectors below, concrete in
# `Individual{T}`'s own number type rather than `Any`, so a `set_isolated!`
# call pushes one inline tuple instead of boxing it.
const _RemovalEntry{T} = Tuple{Int, Symbol, T, T}
const _IsolationSpanEntry{T} = Tuple{Int, T, T, Bool}

"""Record that a person is isolated from `time` (days) until `release_time`.
Used by interventions that isolate or quarantine people. `release_time` is
required: pass `Inf` for isolation that never ends, so that indefinite
isolation is always a stated choice.

A transmission route that isolation should end lists
[`EpiBranch.INTERVENTION_REMOVAL`](@ref) in its `until`, which also allows
for leaky isolation. `:isolated` in an `until` instead refers to a
`Transition(:isolated, …)` step in the natural history.

`key` names the [`record_removal!`](@ref EpiBranch.record_removal!) key the
stretch goes under, the shared key by default. A removal that only blocks a
fraction of the hazard passes its own key instead, apart from the shared one,
so a likelihood never mistakes one for the other.

`unrecorded` says whether this particular removal counts as a detection (see
[`is_isolated`](@ref)): `false` (the default) records it, `true` keeps it out
of the line list and of isolation-triggered tracing. [`Isolation`](@ref)
passes whatever its [`records_isolation`](@ref EpiBranch.records_isolation)
says for its own start; a quarantine is always recorded. Where this call
overlaps or touches a removal already standing, whichever of the two starts
earliest decides the combined span's detection status, the same way its own
start and release decide [`isolation_time`](@ref) and
[`isolation_release_time`](@ref)."""
function set_isolated!(
        ind::Individual, time::Real;
        release_time::Real, key::Symbol = REMOVAL_STRETCHES_KEY,
        unrecorded::Bool = false
    )
    _set_isolated!(ind, time; release_time, key, unrecorded)
    return release_time
end

# As `set_isolated!`, but returns the handle the entry was recorded under, so
# a component that isolates or quarantines someone can revise or withdraw
# exactly its own effect later (see `Isolation`'s `reset!` and
# `_undo_quarantine_isolation!`), without disturbing another's record left
# under the same key.
function _set_isolated!(
        ind::Individual{T}, time::Real;
        release_time::Real, key::Symbol = REMOVAL_STRETCHES_KEY,
        unrecorded::Bool = false
    ) where {T}
    S = promote_type(T, typeof(time), typeof(release_time))
    t, rt = convert(S, time), convert(S, release_time)
    handle = _next_removal_handle!(ind)
    _push_isolation_span_entry!(ind, handle, t, rt, unrecorded)
    _push_qualifying_removal!(ind, handle, key, t, rt)
    _refresh_isolation_span!(ind)
    return handle
end

"""
    record_removal!(ind, start, release; key = :_removal_stretches) -> handle

Record that `ind` could not transmit from `start` until `release` (days;
`Inf` if never released), for example while isolated or quarantined. Returns a
handle identifying this entry, or `nothing` if `start` and `release` describe
no removal at all (`start` not finite, or `release` not after it). Pass the
handle to [`revise_removal!`](@ref EpiBranch.revise_removal!) or
[`withdraw_removal!`](@ref EpiBranch.withdraw_removal!) to change or undo
exactly this entry later, leaving every other entry recorded under `key`
untouched, whatever it overlaps. [`set_isolated!`](@ref) records under the
default key, shared by isolation and quarantine.

An intervention of your own that removes people in its own way passes its own
`key`, and names it in
[`removal_gap_host_times`](@ref EpiBranch.removal_gap_host_times). The
likelihood then leaves out the same periods of exposure that the simulation
blocked, so that `simulate` and `loglikelihood` agree. Every `key` that ever
records a stretch with a finite `release` here is checked against
[`removal_gap_host_times`](@ref EpiBranch.removal_gap_host_times) when the
infection layer is built, and the likelihood stops with an error for one that
no component names or withdraws.

Each call is its own entry: recording a second stretch under the same `key`
never merges it into the first, whatever their overlap.
[`removal_stretches`](@ref) still reads the merged, non-overlapping view over
every entry recorded under `key`, recomputed whenever one is added, revised
or withdrawn.
"""
function record_removal!(
        ind::Individual{T}, start::Real, release::Real;
        key::Symbol = REMOVAL_STRETCHES_KEY
    ) where {T}
    (isfinite(start) && release > start) || return nothing
    # The stretch keeps the individual's own number type, so that an AD dual
    # isolation time flows through as the time accessors promise.
    S = promote_type(T, typeof(start), typeof(release))
    handle = _next_removal_handle!(ind)
    _push_qualifying_removal!(ind, handle, key, convert(S, start), convert(S, release))
    return handle
end

"""
    revise_removal!(ind, handle; release)

Change the release of the removal [`record_removal!`](@ref) recorded as
`handle` to `release` (days), ending it sooner or letting it run longer.
Does nothing if `handle` no longer names an entry, for example one already
withdrawn. [`removal_stretches`](@ref), [`isolation_time`](@ref) and
[`isolation_release_time`](@ref) reflect the change immediately, recomputed
from every entry that still stands. A `release` at or before the entry's own
start leaves it out of `removal_stretches` altogether, as a zero-length
removal already does; one strictly before the start leaves it out of the
isolation span too, as though withdrawn, rather than isolating for negative
time.

For extension authors: a policy that can shorten or lengthen a removal
already recorded, such as a negative test ending a quarantine early, revises
it directly here, rather than finding and re-combining whatever else was
recorded alongside it. [`set_isolated!`](@ref) records an entry the same
way but returns `release_time` instead of the handle. Only built-in
components of the package itself can revise or withdraw an entry it
recorded this way, because they keep the handle under a key of their own
(see the `:_isolation_handle` key in [`Isolation`](@ref)).
"""
function revise_removal!(ind::Individual, handle; release::Real)
    _revise_removal_entry!(ind, handle, release)
    _revise_isolation_span_entry!(ind, handle, release)
    return nothing
end

"""
    withdraw_removal!(ind, handle)

Undo the removal [`record_removal!`](@ref) recorded as `handle`, as though
it had never been recorded. Does nothing if `handle` no longer names an
entry. [`removal_stretches`](@ref), [`isolation_time`](@ref) and
[`isolation_release_time`](@ref) reflect the withdrawal immediately,
recomputed from whatever else still stands.

For extension authors: [`Scheduled`](@ref)'s own reset calls this to undo an
isolation or quarantine that falls outside its window; a policy that cancels
a removal outright calls it too. A policy that wants the same for its own
`set_isolated!` call cannot reach this: [`set_isolated!`](@ref) returns
`release_time` rather than the handle it recorded. An entry it made can
only be withdrawn wholesale, with [`clear_isolated!`](@ref).
"""
function withdraw_removal!(ind::Individual, handle)
    _withdraw_removal_entry!(ind, handle)
    _withdraw_isolation_span_entry!(ind, handle)
    return nothing
end

# A handle unique to `ind`, identifying one entry across every store a write
# to it can land in (the per-key stretch history below, and the isolation
# span one `_set_isolated!` keeps), so `revise_removal!`/`withdraw_removal!`
# can find it in whichever it was written to.
function _next_removal_handle!(ind::Individual)
    n = get(ind.state, :_removal_handle_counter, 0)::Int + 1
    ind.state[:_removal_handle_counter] = n
    return n
end

_find_removal_entry(entries::Nothing, handle) = nothing
_find_removal_entry(entries, handle) = findfirst(e -> e[1] == handle, entries)

# Every stretch ever recorded under any key, as `(handle, key, start,
# release)`, each entry kept for as long as it stands: the unmerged source
# `removal_stretches`' per-key cache below is recomputed from, so that
# `withdraw_removal!` can drop exactly one without disturbing another entry a
# `_merge_stretches!` once folded it together with.
function _push_removal_entry!(ind::Individual{T}, handle::Int, key::Symbol, start, release) where {T}
    old = get(ind.state, :_removal_entries, nothing)::Union{Nothing, Vector{_RemovalEntry{T}}}
    entry = (handle, key, convert(T, start), convert(T, release))
    # A fresh vector each time rather than one grown in place: an archived
    # episode holds a copy of the state that shares the old one.
    ind.state[:_removal_entries] = old === nothing ? _RemovalEntry{T}[entry] : push!(copy(old), entry)
    return nothing
end

# Record a qualifying stretch (start finite, release after it) as its own
# entry, and fold it into `key`'s merged cache the fast way `record_removal!`
# always has, by merging the one new stretch into whatever already stands
# rather than recomputing the whole cache from every entry (`revise_removal!`
# and `withdraw_removal!` do that instead, since they must also account for
# another entry changing or disappearing). A non-qualifying call (`set_isolated!`
# accepts one, such as a zero-length isolation) is dropped here, leaving
# `isolation_time` as the only place it is seen.
function _push_qualifying_removal!(ind::Individual, handle::Int, key::Symbol, start, release)
    (isfinite(start) && release > start) || return nothing
    _push_removal_entry!(ind, handle, key, start, release)
    old = get(ind.state, key, nothing)
    ind.state[key] = old === nothing ? [(start, release)] : _with_stretch(old, (start, release))
    # Only a stretch that releases its host needs a named key: one that never
    # does closes the infectious window instead (see `infectious_removal_time`),
    # which the likelihood already reads with no key of its own.
    isfinite(release) && _note_lapsing_removal_key!(ind, key)
    return nothing
end

function _revise_removal_entry!(ind::Individual, handle, release::Real)
    entries = get(ind.state, :_removal_entries, nothing)
    i = _find_removal_entry(entries, handle)
    i === nothing && return nothing
    h, key, start, _ = entries[i]
    S = promote_type(typeof(start), typeof(release))
    new_entries = copy(entries)
    new_entries[i] = (h, key, start, convert(S, release))
    ind.state[:_removal_entries] = new_entries
    _recompute_removal_cache!(ind, key)
    return nothing
end

function _withdraw_removal_entry!(ind::Individual, handle)
    entries = get(ind.state, :_removal_entries, nothing)
    i = _find_removal_entry(entries, handle)
    i === nothing && return nothing
    key = entries[i][2]
    new_entries = copy(entries)
    deleteat!(new_entries, i)
    ind.state[:_removal_entries] = new_entries
    _recompute_removal_cache!(ind, key)
    return nothing
end

# Rebuild `key`'s merged cache from scratch, from every entry still recorded
# under it: the only way to drop or revise one entry without assuming which
# other it was folded together with. Rare, called only by `revise_removal!`
# and `withdraw_removal!`, never by the fast path a plain `record_removal!`
# call takes, so it need not keep the specialised `Vector{Tuple{T,T}}` type
# that path does for `removal_stretches`' fast reader; one that wants it
# falls back to its own slower, generic path.
function _recompute_removal_cache!(ind::Individual, key::Symbol)
    entries = get(ind.state, :_removal_entries, nothing)
    pairs = Any[]
    lapses = false
    if entries !== nothing
        for (_, k, s, r) in entries
            k === key || continue
            # The same qualifying condition `_push_qualifying_removal!` applies
            # on the fast path: a revise that leaves a release at or before its
            # own start describes no removal at all, so it drops out here too
            # rather than leaving an inverted interval behind.
            (isfinite(s) && r > s) || continue
            push!(pairs, (s, r))
            isfinite(r) && (lapses = true)
        end
    end
    if isempty(pairs)
        delete!(ind.state, key)
    else
        sort!(pairs; by = first)
        ind.state[key] = _merge_stretches!(pairs)
    end
    lapses && _note_lapsing_removal_key!(ind, key)
    return nothing
end

# Drop every stretch entry filed under `key`, as though none had ever been
# recorded. Used by `clear_isolated!` and by `_reset_traced!` for a
# quarantine's own key, so that a fresh removal recorded under the same key
# later starts from nothing rather than resurrecting what a full reset undid.
function _withdraw_all_removals!(ind::Individual, key::Symbol)
    entries = get(ind.state, :_removal_entries, nothing)
    entries === nothing && return nothing
    any(e -> e[2] === key, entries) || return nothing
    ind.state[:_removal_entries] = filter(e -> e[2] !== key, entries)
    delete!(ind.state, key)
    return nothing
end

# Every `set_isolated!` call's own `(handle, start, release, unrecorded)`,
# kept apart from the per-key stretch history above: `isolation_time`,
# `isolation_release_time` and `is_isolated` read the merged view over these,
# whichever key (the shared one or the leaky one) each call's stretch also
# went to, so a leaky isolation still moves them even though it records no
# stretch under the shared key.
function _push_isolation_span_entry!(
        ind::Individual{T}, handle::Int, start, release, unrecorded::Bool
    ) where {T}
    old = get(ind.state, :_isolation_span_entries, nothing)::Union{Nothing, Vector{_IsolationSpanEntry{T}}}
    entry = (handle, convert(T, start), convert(T, release), unrecorded)
    ind.state[:_isolation_span_entries] = old === nothing ? _IsolationSpanEntry{T}[entry] : push!(copy(old), entry)
    return nothing
end

function _revise_isolation_span_entry!(ind::Individual, handle, release::Real)
    entries = get(ind.state, :_isolation_span_entries, nothing)
    i = _find_removal_entry(entries, handle)
    i === nothing && return nothing
    h, start, _, unrecorded = entries[i]
    S = promote_type(typeof(start), typeof(release))
    new_entries = copy(entries)
    new_entries[i] = (h, start, convert(S, release), unrecorded)
    ind.state[:_isolation_span_entries] = new_entries
    _refresh_isolation_span!(ind)
    return nothing
end

function _withdraw_isolation_span_entry!(ind::Individual, handle)
    entries = get(ind.state, :_isolation_span_entries, nothing)
    i = _find_removal_entry(entries, handle)
    i === nothing && return nothing
    new_entries = copy(entries)
    deleteat!(new_entries, i)
    ind.state[:_isolation_span_entries] = new_entries
    _refresh_isolation_span!(ind)
    return nothing
end

# The isolation span's current group, folded from `entries`: sorted by
# start, merged into disjoint groups the way `_merge_stretches!` merges
# stretches, with the earliest-starting entry of a group deciding whether the
# whole group is recorded (so a later isolation or quarantine landing on a
# standing one does not turn it into a detection by asking its own
# eligibility about a start the group's own earliest already settled). The
# last group — the one with the greatest start — is the current isolation;
# an earlier, disjoint one is history `removal_stretches` still holds, but no
# longer the removal in force. An entry whose release has been revised to
# before its own start is dropped, as though withdrawn: a zero-length span
# still isolates for an instant, but an inverted one isolates for none at all.
# `nothing` if no entry remains.
function _fold_isolation_span(entries)
    length(entries) == 0 && return nothing
    if length(entries) == 1
        _, s, r, u = entries[1]
        return r >= s ? (s, r, u) : nothing
    end
    spans = sort!(filter(e -> e[3] >= e[2], entries); by = e -> e[2])
    isempty(spans) && return nothing
    _, cur_start, cur_release, cur_unrecorded = spans[1]
    for e in @view spans[2:end]
        _, s, r, u = e
        if s <= cur_release
            cur_release = max(cur_release, r)
        else
            cur_start, cur_release, cur_unrecorded = s, r, u
        end
    end
    return (cur_start, cur_release, cur_unrecorded)
end

# Recompute `:isolated`, `:isolation_time`, `:isolation_release_time` and
# `:_isolation_unrecorded` from every surviving isolation span entry.
function _refresh_isolation_span!(ind::Individual{T}) where {T}
    entries = get(ind.state, :_isolation_span_entries, ())::Union{Tuple{}, Vector{_IsolationSpanEntry{T}}}
    span = _fold_isolation_span(entries)
    if span === nothing
        ind.state[:isolated] = false
        ind.state[:isolation_time] = T(Inf)
        ind.state[:isolation_release_time] = T(Inf)
        delete!(ind.state, :_isolation_unrecorded)
        return nothing
    end
    start, release, unrecorded = span
    ind.state[:isolated] = true
    ind.state[:isolation_time] = start
    ind.state[:isolation_release_time] = release
    if unrecorded
        ind.state[:_isolation_unrecorded] = true
    else
        delete!(ind.state, :_isolation_unrecorded)
    end
    return nothing
end

# The isolation span as it would read with `handle`'s own entry left out,
# without touching `ind`'s real state: what the line list reports for an
# uninfected contact in place of a self-isolation `Isolation` based on a
# provisional onset that never happened, since nothing ever calls `reset!`
# to withdraw that entry for a contact nobody schedules or rations.
# `nothing` if nothing would stand without it.
function _isolation_span_excluding(ind::Individual, handle)
    entries = get(ind.state, :_isolation_span_entries, nothing)
    entries === nothing && return nothing
    i = _find_removal_entry(entries, handle)
    i === nothing && return _fold_isolation_span(entries)
    others = copy(entries)
    deleteat!(others, i)
    return _fold_isolation_span(others)
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
    delete!(ind.state, :_isolation_span_entries)
    _withdraw_all_removals!(ind, REMOVAL_STRETCHES_KEY)
    # Leaky isolation's own record, apart from the shared history above; only
    # `Isolation` ever writes it, so clearing it unconditionally is safe.
    _withdraw_all_removals!(ind, LEAKY_REMOVAL_STRETCHES_KEY)
    return nothing
end
