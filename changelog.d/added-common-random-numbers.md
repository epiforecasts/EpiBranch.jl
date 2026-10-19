`CommonRandomNumbers(seed)` is a new `rng` to pass to `simulate` that keeps
two scenarios on the same draws until one of them changes what actually
happens, instead of every draw after the first difference moving to a new
value. Progression delays, infectiousness and susceptibility, acceptance of
an offer, and a pair's contact times now go through `EpiBranch.stream(rng,
id, purpose)`, a new dispatched seam a model or intervention can also draw
from for its own per-person or per-pair values. The default method, for any
ordinary RNG, returns it unchanged, so nothing about an existing run changes
unless `CommonRandomNumbers` is the one passed in.
