On a `HouseholdProcess`, a count-gated `Scheduled(iv; start_after_cases = N)`
or a `CapacityConstrained` budget could act before its trigger, because
households race one after another and the running case count after one
household's race holds every case from it regardless of calendar time.
`EpiBranch.reads_population_state(intervention)` lets a component declare
whether its delivery depends on such population-wide state; the household
engine now puts every household on one shared clock whenever any composed
component does. The default is conservative (`true`); `Isolation`,
`ContactTracing`, plain vaccination and a time-only `Scheduled` declare
`false`, and a count-gated `Scheduled` and `CapacityConstrained` declare
`true`. Per-household races, the faster path, stay the default when every
component declares them safe.
