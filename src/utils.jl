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
scale_distribution(d::Poisson, factor::Real) = Poisson(mean(d) * factor)

scale_distribution(d::NegativeBinomial, factor::Real) =
    NegativeBinomial(d.r, d.r / (d.r + mean(d) * factor))

function scale_distribution(d::Distribution, ::Real)
    throw(
        ArgumentError(
            "scale_distribution is not defined for $(typeof(d)); only Poisson and " *
                "NegativeBinomial offspring distributions can be scaled."
        )
    )
end

# Number of the `k` marked items among `draws` taken without replacement from
# `k` marked and `m` unmarked ones. Inverts the distribution outwards from its
# mode, so the cost grows with its standard deviation rather than with `draws`
# as `rand(Hypergeometric(k, m, draws))` does.
function _rand_hypergeometric(rng::AbstractRNG, k::Int, m::Int, draws::Int)
    lo, hi = max(0, draws - m), min(draws, k)
    lo == hi && return lo
    mode = clamp(((draws + 1) * (k + 1)) ÷ (k + m + 2), lo, hi)
    u = rand(rng)
    up = down = pmf = pdf(Hypergeometric(k, m, draws), mode)
    u -= pmf
    u <= 0 && return mode
    x_up = x_down = mode
    while x_up < hi || x_down > lo
        if x_up < hi
            up *= (k - x_up) * (draws - x_up) / ((x_up + 1) * (m - draws + x_up + 1))
            x_up += 1
            u -= up
            u <= 0 && return x_up
        end
        if x_down > lo
            down *= x_down * (m - draws + x_down) / ((k - x_down + 1) * (draws - x_down + 1))
            x_down -= 1
            u -= down
            u <= 0 && return x_down
        end
    end
    return mode
end

# Split `total` items drawn without replacement from groups of `sizes` items
# between the groups (the multivariate hypergeometric distribution).
function _rand_multivariate_hypergeometric(rng::AbstractRNG, sizes, total::Int)
    drawn = zeros(Int, length(sizes))
    pool = sum(sizes)
    for i in eachindex(sizes)
        pool -= sizes[i]
        drawn[i] = if total == 0 || sizes[i] == 0
            0
        elseif pool == 0
            total
        else
            _rand_hypergeometric(rng, sizes[i], pool, total)
        end
        total -= drawn[i]
    end
    return drawn
end
