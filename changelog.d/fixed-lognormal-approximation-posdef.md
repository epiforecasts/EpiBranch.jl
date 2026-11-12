`LognormalApproximation` no longer throws a `PosDefException` when its
log-scale covariance is not positive definite, as happens for some long
subcritical series. Adding a small diagonal term to the covariance lets the
likelihood be returned. The docstring documents this.
