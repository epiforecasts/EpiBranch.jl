"""
    scale_distribution(d, factor::Real)

Multiply the mean of an offspring distribution, the reproduction number, by
`factor`, keeping the same family and dispersion: for example to model a
control measure that cuts transmission by a fixed proportion. Defined for
`Poisson` (`Poisson(R * factor)`) and the negative binomial (same dispersion
`k`, mean `R * factor`); any other distribution raises an error.

# Examples
```julia
scale_distribution(NegBin(2.5, 0.16), 0.5)   # R = 1.25, k = 0.16
```
"""
function scale_distribution(d::Poisson, factor::Real)
    return Poisson(mean(d) * factor)
end

function scale_distribution(d::NegativeBinomial, factor::Real)
    k = d.r
    new_mean = mean(d) * factor
    p = k / (k + new_mean)
    return NegativeBinomial(k, p)
end

function scale_distribution(d::Distribution, ::Real)
    throw(
        ArgumentError(
            "scale_distribution is not defined for $(typeof(d)); only Poisson and " *
                "NegativeBinomial offspring distributions can be scaled."
        )
    )
end
