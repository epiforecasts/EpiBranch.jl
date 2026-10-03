`EpiBranch.abort_infection!(ind, time)` lets any intervention end an infection
before symptom onset, as `RingVaccination`'s `post_exposure_efficacy` does, and
`EpiBranch.infection_aborted_time(ind)` reads when it ended. The engine applies
the same consequences on every transmission model whichever intervention
records the abort: no onward transmission from that time, no onset, and no
clinical transition from that time on.
