`GroupVaccination`'s revise-earlier policy, moving an admitted dose to a
genuinely earlier trigger discovered later on a continuous-time race, was
written into the intervention body, with its own re-derivation of which
candidates a settled member excludes. It is now a dispatched admission trait,
`EpiBranch.may_revise(intervention, prior_trigger, new_trigger)`, defaulting
to `false` (an admitted dose keeps its date) and overridden by
`GroupVaccination`; a custom intervention can opt into the same pattern on its
own type. The settled/pending distinction it needs is now read from the
engine via `EpiBranch.is_settled(state, ind)`, kept alongside the candidate
list (`is_settled` is only ever set on a continuous-time race; the ordinary
generation engine still needs the candidate list to keep a revision within
the members a given call actually discovered). The two action builders for
giving a first dose and revising one already given are now one.
