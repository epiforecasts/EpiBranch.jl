`contacts()` returned an empty table on `HomogeneousProcess`, `HouseholdProcess`
and `NetworkProcess`: the continuous-time race set each case's `parent_id` but
never added it to the infector's `secondary_case_ids`, the field `contacts()`
reads. It now lists every infector-infectee pair the race settled, as it
already did on `BranchingProcess`. These models keep no record of the
contacts a case exposed without infecting them, so `contacts()` has no
uninfected rows to report for them, unlike on `BranchingProcess`.
