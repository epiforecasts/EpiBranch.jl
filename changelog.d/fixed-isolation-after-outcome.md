`Isolation` no longer records a detection when a case's self-report or traced
isolation time falls at or after the case's own outcome (recovery, death, or
any other terminal `Transition`). Such a case is still removed from
transmission at that time, exactly as before, but `is_isolated` is `false` for
it: the line list's `isolated` and `date_isolation` columns no longer count it
as detected, and tracing or group vaccination triggered by isolation no longer
starts from it. The new reserved state key `:isolation_unrecorded` marks
such a removal, and `outcome_time` reads the time of the outcome.

Whether such a time counts as a detection is the isolation eligibility's
call, through `EpiBranch.records_isolation`. The default declines it; a policy
that records a detection arriving after the outcome, as post-mortem detection
does, overrides that method.
