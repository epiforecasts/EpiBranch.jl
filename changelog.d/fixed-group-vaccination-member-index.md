On a continuous-time network or household race, `RingVaccination` and
`GroupVaccination` scanned every still-pending individual on every settled
case to find their candidate actions. `RingVaccination` now looks only at
the settling case's own traced contacts, and `GroupVaccination` only at its
own group's members, via a group-to-members index, so cost no longer grows
with population size on every case. That growth had dominated at city scale
and on a `HouseholdProcess` with many households.
