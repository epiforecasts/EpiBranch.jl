`LognormalApproximation` no longer throws a `PosDefException` when its
log-scale covariance is not positive definite, as happens for some long
subcritical series. A small diagonal term is added so the likelihood is
returned, and the docstring says so.
