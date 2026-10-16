`RingVaccination` with a finite `eligibility_window` now works on the
network and household models, rather than being ignored with a warning. A
dose for a contact not yet exposed is admitted as though the window were
open, and withdrawn once the race settles that contact's own infection and
the window turns out not to cover it, freeing any `CapacityConstrained`
budget the withdrawn dose had used.
