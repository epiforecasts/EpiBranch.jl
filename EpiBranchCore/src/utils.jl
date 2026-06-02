"""
    scale_distribution(d::Distribution, factor::Real)

Scale a distribution's mean by `factor`, preserving distribution family and shape.
For Poisson: a Poisson(λ * factor) is returned.
For NegativeBinomial: a NegativeBinomial with same k and scaled mean is returned.
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
