The incidence time-series likelihood now covers multi-type processes built
from an offspring matrix. `IncidenceCounts` takes counts by type, one column
per type, or their total over types, with seeds by type, and
`incidence_moments` returns the mean and covariance of each type's counts.
Under-reporting applies to every type, as in the single-type case.
