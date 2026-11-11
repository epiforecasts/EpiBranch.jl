`progression_loglik` now scores a line list that holds only the outcome that
happened. A `Death` or `Recovery` whose candidate time is absent because the
other outcome came first counts as not having happened by that outcome's time,
giving `f_D(t) S_R(t)` for a death at `t` and the reverse for a recovery,
rather than `-Inf`.
