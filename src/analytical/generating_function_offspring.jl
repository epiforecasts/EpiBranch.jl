# ── Offspring laws given directly by their generating function ───────
#
# An intervention whose effect on the offspring distribution is a mixture over
# some random quantity (isolation's effect over the isolation time, say) is
# generally not a negative binomial or a Poisson. It is still a law the
# analytical functions can use, through its probability generating function
# alone: `extinction_probability` and `probability_contain` only ever read a
# law through `_pgf` and its mean.

"""
    GeneratingFunctionOffspring(pgf, mean)
    GeneratingFunctionOffspring(pmf::AbstractVector{<:Real})

An offspring law given directly by its probability generating function
`pgf(s)`, or built from a probability mass function `pmf` over `0, 1, 2, ...`
(`pmf[i]` is the probability of `i - 1`). `mean` is the law's reproduction
number, the mean number of secondary cases.

Return one of these from [`EpiBranch.analytic_offspring_effect`](@ref) for an
effect with no simpler closed form, such as a mixture over a random isolation
time: [`Isolation`](@ref) does this. [`extinction_probability`](@ref) and
[`probability_contain`](@ref) accept it directly, and so does
[`reproduction_number`](@ref). [`proportion_transmission`](@ref),
[`proportion_cases_individual`](@ref), [`proportion_cases_offspring`](@ref)
and `end_of_outbreak_probability` need a negative binomial or Poisson's own
dispersion or rate. A law known only through its generating function has
none, so these raise an `ArgumentError` pointing at simulation instead.

[`chain_size_distribution`](@ref) needs the pmf itself, not just the
generating function it sums to, so it only works when `GeneratingFunctionOffspring`
is built from `pmf`.

# Examples
```julia
# Poisson(1.5) thinned by keeping each secondary case with probability 0.6
g = GeneratingFunctionOffspring(s -> exp(1.5 * (0.6 * s + 0.4 - 1)), 1.5 * 0.6)
extinction_probability(g)

# the same law from its probability mass function, which also gives
# chain_size_distribution
using Distributions
pmf = [pdf(Poisson(0.9), n) for n in 0:200]
chain_size_distribution(GeneratingFunctionOffspring(pmf))
```
"""
struct GeneratingFunctionOffspring{G, P}
    pgf::G
    mean::Float64
    pmf::P
end

GeneratingFunctionOffspring(pgf, mean::Real) = GeneratingFunctionOffspring(pgf, float(mean), nothing)

function GeneratingFunctionOffspring(pmf::AbstractVector{<:Real})
    mean_ = sum((i - 1) * p for (i, p) in enumerate(pmf))
    pgf(s) = sum(p * s^(i - 1) for (i, p) in enumerate(pmf))
    return GeneratingFunctionOffspring(pgf, float(mean_), float.(pmf))
end

function Base.show(io::IO, o::GeneratingFunctionOffspring)
    return print(io, "GeneratingFunctionOffspring(mean=$(o.mean))")
end

_pgf(o::GeneratingFunctionOffspring, s) = o.pgf(s)
_law_mean(o::GeneratingFunctionOffspring) = o.mean

"""
    reproduction_number(o::GeneratingFunctionOffspring)

The law's own `mean`.
"""
reproduction_number(o::GeneratingFunctionOffspring) = o.mean

# A bare generating function has no case-by-case probabilities to read, so
# `pdf` and `chain_size_distribution` both refuse a GeneratingFunctionOffspring
# built that way rather than reaching `o.pmf[k]` as a confusing `MethodError`
# on `nothing`.
function _no_pmf(o::GeneratingFunctionOffspring)
    throw(
        ArgumentError(
            "chain_size_distribution needs a GeneratingFunctionOffspring built " *
                "from its probability mass function, GeneratingFunctionOffspring(pmf), " *
                "not a bare generating function, which carries no case-by-case " *
                "probabilities to read."
        )
    )
end

# `pdf` on a GeneratingFunctionOffspring built from a pmf is just that pmf,
# padded with zero past its stored length (the truncation the caller already
# chose when they built it). Needed only by `chain_size_distribution`'s
# hitting-time theorem, which reads an offspring law's secondary-case
# probabilities one at a time.
function Distributions.pdf(o::GeneratingFunctionOffspring, k::Integer)
    o.pmf === nothing && _no_pmf(o)
    return 0 <= k < length(o.pmf) ? o.pmf[k + 1] : 0.0
end

"""
    chain_size_distribution(o::GeneratingFunctionOffspring)

Exact chain size distribution for an offspring law given by its probability
mass function, from the hitting-time theorem
([`EpiBranch.HittingTimeChainSize`](@ref)). Needs `o` to have been built as
`GeneratingFunctionOffspring(pmf)`: a law given only as a bare generating
function has no case-by-case probabilities to read, and raises an
`ArgumentError` instead.
"""
function chain_size_distribution(o::GeneratingFunctionOffspring)
    o.pmf === nothing && _no_pmf(o)
    return HittingTimeChainSize(o)
end

"""
    extinction_probability(o::GeneratingFunctionOffspring; tol=1e-10, max_iter=1000)

Probability that transmission from a single introduced case dies out, for an
offspring law given by its generating function. Found by the same
fixed-point iteration as [`extinction_probability(d::Poisson)`](@ref).
"""
function extinction_probability(
        o::GeneratingFunctionOffspring; tol::Real = 1.0e-10,
        max_iter::Int = 1000
    )
    o.mean <= 1.0 && return 1.0

    q = 0.5
    for _ in 1:max_iter
        q_new = _pgf(o, q)
        abs(q_new - q) < tol && return q_new
        q = q_new
    end
    @warn_unconverged_extinction(max_iter, "the reproduction number")
    return q
end

"""
    probability_contain(o::GeneratingFunctionOffspring; n_initial=1, ind_control=0.0, pop_control=0.0)

Closed-form containment probability for an offspring law given by its
generating function, with the same keywords as
[`probability_contain(R, k)`](@ref). `pop_control`'s proportional reduction in
R is applied by thinning: keeping each secondary case with probability
`1 - pop_control`, which rescales the law's generating function the same way
it rescales a negative binomial's or a Poisson's mean.
"""
function probability_contain(
        o::GeneratingFunctionOffspring;
        n_initial::Int = 1,
        ind_control::Real = 0.0,
        pop_control::Real = 0.0,
        tol::Real = 1.0e-10, max_iter::Int = 1000
    )
    0.0 <= ind_control <= 1.0 || throw(ArgumentError("ind_control must be in [0, 1]"))
    0.0 <= pop_control <= 1.0 || throw(ArgumentError("pop_control must be in [0, 1]"))
    n_initial >= 1 || throw(ArgumentError("n_initial must be ≥ 1"))

    keep = 1.0 - pop_control
    keep * o.mean <= 1.0 && return 1.0

    q = 0.5
    for _ in 1:max_iter
        g = _pgf(o, 1.0 - keep * (1.0 - q))
        q_new = ind_control + (1.0 - ind_control) * g
        abs(q_new - q) < tol && return q_new^n_initial
        q = q_new
    end

    @warn_unconverged_extinction(
        max_iter,
        "the effective reproduction number times one minus `ind_control`"
    )
    return q^n_initial
end

# ── Functions a pgf-only law cannot support ───────────────────────────
#
# `extinction_probability`, `probability_contain` and `reproduction_number`
# read a law through `_pgf` and `_law_mean` alone and work for any
# `GeneratingFunctionOffspring` above. The superspreading-ranking functions
# need a negative binomial's own dispersion or a per-count pmf instead, and
# `end_of_outbreak_probability` needs the same dispersion. A law known only
# through its generating function has none of that, so these refuse
# explicitly rather than reaching `_offspring_through_interventions`'s caller
# as a raw `MethodError` and leaving the user without the usual pointer to
# simulation.
function _no_dispersion(fn::AbstractString)
    throw(
        ArgumentError(
            "$fn needs a NegativeBinomial or Poisson offspring distribution; " *
                "a GeneratingFunctionOffspring carries no such dispersion to use. " *
                "Compute it from simulated outbreaks, simulate(spec, n), instead."
        )
    )
end

proportion_transmission(::GeneratingFunctionOffspring; prop_cases::Real = 0.2) =
    _no_dispersion("proportion_transmission")
proportion_cases_individual(::GeneratingFunctionOffspring; prop_transmission::Real = 0.8) =
    _no_dispersion("proportion_cases_individual")
proportion_cases_offspring(::GeneratingFunctionOffspring; prop_transmission::Real = 0.8) =
    _no_dispersion("proportion_cases_offspring")
end_of_outbreak_probability(::GeneratingFunctionOffspring, ::Distribution, ::Real) =
    _no_dispersion("end_of_outbreak_probability")
