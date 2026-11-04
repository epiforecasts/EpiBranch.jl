Differentiating an incidence likelihood with `ForwardDiff` through a `Dual`
generation-time parameter no longer throws a `MethodError`. The discretised
generation-time and reporting-delay pmfs, and the pgf that combines them with
the offspring law, now take their element type from the distributions passed
in rather than assuming `Float64`.
