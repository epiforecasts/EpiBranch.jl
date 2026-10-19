`HouseholdProcess` (and, since the fix lives in the shared continuous-time
race, every other structure-driven model) now reinfects a household member
once their immunity has waned: a `progression` reaching `:susceptible_again`
lets a still-infectious household-mate or a repeated community introduction
reach them again, with the earlier infection archived onto
`Individual.episodes` rather than lost. Until now the race settled each host
once and never offered them as a target again, however the progression was
written. `household_infections` now warns when an outbreak has such hosts,
since the pairwise likelihood still reads only each host's current episode.
