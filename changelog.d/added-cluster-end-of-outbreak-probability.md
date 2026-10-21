`end_of_outbreak_probability` now has a method that takes every case's time
since onset and, optionally, the number of secondary cases each has already
caused, and combines them into the probability that the whole cluster is
over. The existing single-`τ` method is documented as a per-case value,
exact for a whole cluster only when its other cases can no longer transmit;
passing only the most recent case's `τ` for a larger cluster overstated the
probability that the cluster was over.
