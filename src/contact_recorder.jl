# ── Contact recorder ─────────────────────────────────────────────────
# An output concern, alongside `ObservationModel`: a composed model can hold
# one so a continuous-time (Sellke) race knows whether a pair's contact draws
# still matter once a standing block (`standing_block`) would otherwise end
# them. Tracing and ring construction read the standing contact relationship,
# not these draws, so they are unaffected either way. A dropped pair still
# costs the later contact *events* between it and its infector; this is the
# seam for recovering them.

"""
Abstract supertype for a composed model's contact recorder. Attached to a
[`ModelSpec`](@ref) as `recorder = ...`; participates through
[`records_contacts`](@ref EpiBranch.records_contacts), dispatched on the
recorder type.
"""
abstract type ContactRecorder end

"""No contact recorder. [`records_contacts`](@ref EpiBranch.records_contacts)
answers `false` for every pair, so a continuous-time race drops a
standing-blocked pair exactly as one with no recorder attached always has."""
struct NoContactRecorder <: ContactRecorder end

"""
    records_contacts(recorder::ContactRecorder, parent, contact, state, t) -> Bool

Whether `recorder` wants the continuous-time (Sellke) race to keep drawing
`parent`'s contacts with `contact`, after a [`EpiBranch.standing_block`](@ref)
has settled the pair for good. Asked every time such a block would otherwise
end the pair's draws — not only the first — with `t` the time the dropped
proposal fell at, so a recorder that wants the stream can log the event
before answering. Default `false`.

Declaring `true` narrows which models can run: resuming the draws puts the
pair back under the rejection-continuation guard described in
[`EpiBranch.standing_block`](@ref), so a model whose window never closes is
refused there, as it would have been had the block never been declared
standing, rather than silently dropping transmission the pair could still
produce.
"""
records_contacts(::ContactRecorder, parent, contact, state, t) = false
