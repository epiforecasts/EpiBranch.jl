`RingVaccination`'s `post_exposure_efficacy` now works on `HouseholdProcess`,
`NetworkProcess` and `RoutedNetwork`: a dose given to a household or network
member while it is still pending (uninfected) is reconsidered once the race
settles that member's own infection, aborting it with the usual probability
when immunity falls between exposure and onset. Previously
`continuous_actions` refused a nonzero `post_exposure_efficacy` on these
models outright, so no dose was given. A route's window on these models
also now closes at a post-exposure abort, matching the `AbortedInfection`
risk that already blocks transmission from that time, rather than staying
open indefinitely when the abort undoes the removal state it would
otherwise have closed on.
