# ── Contact recorder ─────────────────────────────────────────────────
# An output concern, alongside `ObservationModel`: a composed model can hold
# one so a continuous-time (Sellke) race knows whether a pair's contact draws
# still matter once a standing block (`standing_block`) would otherwise end
# them. Tracing and ring construction read the standing contact relationship,
# not these draws, so they are unaffected either way. A dropped pair still
# costs the later contact *events* between it and its infector; this is the
# seam for recovering them.

"""
Asks the network and household models to keep
simulating contacts between an infector and a person whom an intervention
has protected for good, such as a vaccinated contact. By default those
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
for good (see [`EpiBranch.standing_block`](@ref)). Asked each time such a
contact would be dropped, with `t` the time of that contact (days), so a
recorder can log the contact before answering. Default `false`.

Returning `true` restricts which models can run: a model whose infectious
period never ends would simulate contacts for ever, so it raises an error
instead, as described in [`EpiBranch.standing_block`](@ref).
"""
records_contacts(::ContactRecorder, parent, contact, state, t) = false
