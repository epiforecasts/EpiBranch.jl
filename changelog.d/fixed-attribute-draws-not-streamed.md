`clinical_presentation`, `demographics` and `groups` now draw each person's
age, sex, symptom status, incubation period and group through
`EpiBranch.stream(rng, id, purpose)`, as `transmission_traits` and the
built-in interventions already do. Under a plain `rng` nothing changes; under
`CommonRandomNumbers`, these draws no longer shift with the order people are
created, which a model that creates them lazily (as a network or household
process could, rather than for the whole population up front) cannot
otherwise guarantee.
