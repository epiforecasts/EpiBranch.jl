`contacts()` returned an empty table on `HomogeneousProcess`, `HouseholdProcess`
and `NetworkProcess`: the continuous-time race set each case's `parent_id` but
never registered it on the infector's `secondary_case_ids`, which is what
`contacts()` reads. It now lists every infector-infectee pair the race
settled, as it already did on `BranchingProcess`. These models do not keep a
record of the contacts a case exposed without infecting them, so, unlike on
`BranchingProcess`, `contacts()` has no uninfected rows to report for them.
