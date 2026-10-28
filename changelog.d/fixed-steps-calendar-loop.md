A `PairKernel` calendar built from `Steps` no longer loops forever on an
outbreak with cases before the first break, where converting a segment's end
back to time-since-opening and forward again could round short of the break.
