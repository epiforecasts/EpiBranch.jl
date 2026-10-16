A continuous-time model's race asks its `ContactRecorder` about every
blocked proposal now, not only ones certain to recur, so a recorder can log
an exposure that an intervention or the end of the infector's window stopped
from infecting. A new built-in `ExposedContactRecorder`, attached as a
`ModelSpec`'s `recorder`, logs these pairs for `contacts()` to list with
`infected = false` — the row a household member or network neighbour who
escaped infection previously had no way to get, unlike on `BranchingProcess`.
