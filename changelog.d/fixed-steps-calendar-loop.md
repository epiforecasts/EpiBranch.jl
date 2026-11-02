A `PairKernel` calendar built from `Steps` no longer loops forever on an
outbreak with cases before the first break, where converting the end of a
segment back to time-since-opening and forward again could round short of the
break.
