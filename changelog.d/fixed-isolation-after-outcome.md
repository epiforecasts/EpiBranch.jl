`Isolation` no longer records a case as isolated when its self-report or traced
isolation time falls at or after the case's own outcome (recovery, death, or
any other terminal `Transition`). It previously recorded
`:isolated`/`:isolation_time` regardless, so a case could be flagged as
isolated days after recovering, which `OnIsolation`, line lists and
detection-based endpoints all read as a genuine detection.

Whether such a time counts as a detection is now the isolation eligibility's
call, through `EpiBranch.records_isolation`. The default declines it; a policy
that records a detection arriving after the outcome, as post-mortem detection
does, overrides that method.
