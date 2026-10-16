The household tutorial shows how to fit the within-household contact interval
when infection times are not observed, only symptom onsets: a Turing model
puts a prior on each case's incubation delay, rebuilds `HouseholdInfections`
from the candidate infection times this implies, and conditions on
`EarliestInfected()` since a recruited index case need not be the first
infected.
