`Isolation` no longer records a case as isolated when its self-report or
traced isolation time falls at or after the case's own outcome (recovery,
death, or any other terminal `Transition`). It previously recorded
`:isolated`/`:isolation_time` regardless, so a case could be flagged as
isolated days after recovering — `OnIsolation`, line lists and
detection-based endpoints all read that as a genuine detection.
