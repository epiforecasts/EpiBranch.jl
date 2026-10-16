# ── Contact recorder ─────────────────────────────────────────────────
# An output concern, alongside `ObservationModel`: a composed model can hold
# one so a continuous-time (Sellke) race knows whether a pair's contact draws
# still matter once a standing block (`standing_block`) would otherwise end
# them. Tracing and ring construction read the standing contact relationship,
# not these draws, so they are unaffected either way. A dropped pair still
# costs the later contact *events* between it and its infector; this is the
# seam for recovering them, and `ExposedContactRecorder` is the built-in one
# that feeds them to `contacts`.

"""
Asks the continuous-time models (networks, households,
[`HomogeneousProcess`](@ref) and [`MixingProcess`](@ref)) to keep simulating
contacts between an infector
and a person whom an intervention has protected for good, such as a
vaccinated contact. By default those
contacts are no longer simulated, since they can no longer cause infection;
keep them when you want every contact event recorded. Tracing and ring
vaccination do not depend on it.

Attach one to a [`ModelSpec`](@ref) as `recorder = ...`. A new recorder type
defines [`records_contacts`](@ref EpiBranch.records_contacts).
"""
abstract type ContactRecorder end

"""The default: no contact recorder. Contacts between an infector and a person
protected for good are no longer simulated
([`records_contacts`](@ref EpiBranch.records_contacts) is `false` for every
pair)."""
struct NoContactRecorder <: ContactRecorder end

"""
    records_contacts(recorder::ContactRecorder, parent, contact, state, t) -> Bool

Whether a continuous-time model should keep simulating contacts between the
infector `parent` and `contact` after an intervention has protected `contact`
for good (see [`EpiBranch.standing_block`](@ref)). Asked on every proposal a
risk blocks, not only ones certain to recur, with `t` the time of that
proposal, so a recorder can log each blocked contact; only the answer given
for a proposal whose block is certain decides whether the pair's draws
continue. Default `false`.

Returning `true` for a certain block restricts which models can run: a model
whose infectious period never ends would simulate contacts for ever, so it
raises an error instead, as described in [`EpiBranch.standing_block`](@ref).
"""
records_contacts(::ContactRecorder, parent, contact, state, t) = false

# The key an `ExposedContactRecorder` keeps its log under in a run's
# `scratch`, and `contacts` reads it back from.
const EXPOSED_CONTACTS_KEY = (:exposed_contacts,)

# Every distinct (infector, contact) pair an `ExposedContactRecorder` has seen
# blocked, in the order first seen, with the time of that first block and the
# generation the exposure implies for the contact, kept until either
# `contacts` reads it or the same pair goes on to infect (`_retract_contact!`
# then drops it, since the pair is no longer one that escaped).
struct _ExposedContactLog
    pairs::Set{Tuple{Int, Int}}
    events::Vector{Tuple{Int, Int, Float64, Int}}
end
_ExposedContactLog() = _ExposedContactLog(Set{Tuple{Int, Int}}(), Tuple{Int, Int, Float64, Int}[])

"""
Logs every distinct infector-contact pair a continuous-time model's race
blocks, whether or not the block is certain to recur, keeping the time of the
first block. A pair logged from a block that is not certain to recur is
dropped again if the same pair later infects: the block was not this pair's
last outcome, so it is no longer an exposure that escaped. A blocked community
introduction, where the person proposed as their own infector stands in for a
source outside the population, is not a contact between two people and is not
logged. Attach one to a [`ModelSpec`](@ref) as `recorder =
ExposedContactRecorder()`; [`contacts`](@ref) then lists each pair still
logged as an exposure that did not lead to infection, the same way it already
does for a [`BranchingProcess`](@ref).

Unlike a recorder whose [`records_contacts`](@ref EpiBranch.records_contacts)
answers `true`, this one never asks the race to keep drawing a pair whose
block is certain, so attaching it adds no restriction on which models can
run.
"""
struct ExposedContactRecorder <: ContactRecorder end

function records_contacts(::ExposedContactRecorder, parent, contact, state, t)
    parent.id == contact.id && return false  # a blocked community introduction isn't an exposure between two people
    log = get!(_ExposedContactLog, state.scratch, EXPOSED_CONTACTS_KEY)::_ExposedContactLog
    key = (parent.id, contact.id)
    if key ∉ log.pairs
        push!(log.pairs, key)
        push!(log.events, (parent.id, contact.id, Float64(t), parent.generation + 1))
    end
    return false
end

# Drops `parent`-`contact` from a recorder's log once that pair infects: a
# block that was not certain to recur (`records_contacts` is asked about it
# again) can still be followed by a later, unblocked proposal for the same
# pair, and the earlier block is then not the exposure's last outcome. A
# recorder with nothing to retract, including the default, does nothing.
_retract_contact!(::ContactRecorder, parent, contact, state) = nothing

function _retract_contact!(::ExposedContactRecorder, parent, contact, state)
    log = get(state.scratch, EXPOSED_CONTACTS_KEY, nothing)
    log === nothing && return nothing
    log = log::_ExposedContactLog
    key = (parent.id, contact.id)
    key in log.pairs || return nothing
    delete!(log.pairs, key)
    idx = findfirst(e -> (e[1], e[2]) == key, log.events)
    idx === nothing || deleteat!(log.events, idx)
    return nothing
end
