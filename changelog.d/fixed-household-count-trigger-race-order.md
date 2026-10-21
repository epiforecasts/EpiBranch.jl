On a `HouseholdProcess`, a count-gated `Scheduled(iv; start_after_cases = N)`
or a `CapacityConstrained` budget could act before its trigger, because
households race one after another and the running case count after one
household's race holds every case from it regardless of calendar time.
All households now run on one shared clock, so a running count or a budget
sees cases in calendar order.

A `CapacityConstrained` with a finite `period` on more than one household is
no longer rejected: the shared clock gives the chronological admission its
period windows need.
