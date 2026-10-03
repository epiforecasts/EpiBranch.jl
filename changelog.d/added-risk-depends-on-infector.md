`EpiBranch.risk_depends_on_infector(intervention)` declares whether an
intervention's `competing_risk` reads the infector. A fixed-size pool with more
than one mixing group refuses those that do, and an intervention written
outside the package can declare that its risk acts on the contact alone.
Wrappers such as `CapacityConstrained` take the answer from the intervention
they wrap, so a capacity-limited `RingVaccination` or `GroupVaccination` is
accepted wherever the bare one is.
