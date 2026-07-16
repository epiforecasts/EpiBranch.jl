"""
    scale_distribution(d, factor::Real)

Scale an offspring distribution's mean by `factor`, preserving its family and
shape. Defined for the two offspring families the package scales: `Poisson`
(returns `Poisson(λ · factor)`) and `NegativeBinomial` (same `k`, mean scaled).
Any other family raises an `ArgumentError` naming it, rather than a bare
`MethodError`.
"""
function scale_distribution(d::Poisson, factor::Real)
    Poisson(mean(d) * factor)
end

function scale_distribution(d::NegativeBinomial, factor::Real)
    k = d.r
    new_mean = mean(d) * factor
    p = k / (k + new_mean)
    NegativeBinomial(k, p)
end

function scale_distribution(d::Distribution, ::Real)
    throw(ArgumentError(
        "scale_distribution is not defined for $(typeof(d)); only Poisson and " *
        "NegativeBinomial offspring distributions can be scaled."))
end
