# ── Likelihood of final statuses, with no infection times ────────────
#
# A household study that only follows up who was ever infected, not when,
# still has a closed-form likelihood when the community hazard and the
# within-household kernel act independently across people and pairs (Ball
# 1986): a community contact reaches each person on its own, and an
# infector's contact with each household-mate over its infectious window
# either infects them or does not, independently of every other pair. The final
# infected set is then the set reachable, through those independent edges,
# from whoever the community reaches directly.

# One person's own trait value: a number shared by everyone, or a vector with
# one entry per person (as `HouseholdProcess`'s `external_hazard` is either).
_trait_value(x::Real, ::Int) = x
_trait_value(x::AbstractVector, j::Int) = x[j]

function _check_trait_length(x::AbstractVector, n::Int, name::String)
    length(x) == n || throw(
        ArgumentError("$name must have one entry per person ($n), got $(length(x))")
    )
    return nothing
end
_check_trait_length(::Real, ::Int, ::String) = nothing

# The within-household kernel between a specific ordered pair, for a shared
# distribution or a covariate function `(infector, susceptible) -> Distribution`,
# as `_rowkernel` reads a row's kernel for the pairwise survival likelihood. A
# `PairKernel` needs infection times and host state that a final-status
# likelihood, with no infection times at all, has no way to supply.
_pair_escape_kernel(k::ContinuousUnivariateDistribution, ::Int, ::Int) = k
_pair_escape_kernel(k, i::Int, j::Int) = k(i, j)
function _pair_escape_kernel(::PairKernel, ::Int, ::Int)
    throw(
        ArgumentError(
            "final_status_loglik needs a shared or (infector, susceptible) -> " *
                "Distribution kernel; a PairKernel reads infection times and host " *
                "state that final statuses alone do not carry"
        )
    )
end

# The probability that person `j` escapes the community hazard over `obs_end`,
# with their own susceptibility as the rate multiplier (as `_ext_draw` applies
# it to a community introduction, with no infectiousness term since there is
# no infector): 1 when they face no hazard at all.
function _community_escape(external_hazard, j::Int, obs_end::Real, susceptibility::Real)
    d = _ext_survival_of(external_hazard, j)
    d === nothing && return 1.0
    return Float64(ccdf(d, obs_end))^susceptibility
end

# ∏_{i ∈ A, j ∈ B} p[i, j], read off the bits of the two masks.
function _cross_escape(p::Matrix{Float64}, A::Int, B::Int)
    acc = 1.0
    a = A
    while a != 0
        i = trailing_zeros(a) + 1
        a &= a - 1
        b = B
        while b != 0
            j = trailing_zeros(b) + 1
            b &= b - 1
            acc *= p[i, j]
        end
    end
    return acc
end

function _status_mask(statuses::AbstractVector, value)
    mask = 0
    for (k, s) in enumerate(statuses)
        s === value && (mask |= 1 << (k - 1))
    end
    return mask
end

# The log-likelihood of one household's final statuses: the per-pair household
# escape probabilities `p` (`p[i, j]`, the probability infectious case `i`
# never infects susceptible `j` over the window) and per-person community
# escape probabilities `pi_esc` fix, for every subset of the household, the
# probability that exactly that subset is who ends up infected (Ball 1986);
# summing those probabilities over every subset consistent with `statuses`
# gives the probability of what was actually observed.
function _household_final_status_loglik(
        p::Matrix{Float64}, pi_esc::Vector{Float64}, statuses::AbstractVector
    )
    n = length(pi_esc)
    full = (1 << n) - 1

    # `piProd[mask + 1]`: the probability that every person in `mask` escapes
    # the community hazard, built up one person at a time from smaller masks.
    piProd = Vector{Float64}(undef, full + 1)
    piProd[1] = 1.0
    for mask in 1:full
        lowbit = mask & -mask
        piProd[mask + 1] = piProd[(mask ⊻ lowbit) + 1] * pi_esc[trailing_zeros(mask) + 1]
    end

    # `H[mask + 1]`: the probability that, confined to the people in `mask`
    # alone, every one of them ends up infected, whether seeded by the
    # community or by each other. Exactly one proper submask `c` of `mask` is
    # the set actually reached that way, with the rest of `mask` escaping both
    # the community and every contact from `c` — a partition of probability
    # that fixes `H[mask]` from the `H[c]` of its own (already known) subsets.
    H = Vector{Float64}(undef, full + 1)
    H[1] = 1.0
    for mask in 1:full
        acc = 0.0
        c = mask
        while c != 0
            c = (c - 1) & mask
            s = mask ⊻ c
            acc += piProd[s + 1] * _cross_escape(p, c, s) * H[c + 1]
        end
        H[mask + 1] = max(0.0, 1.0 - acc)
    end

    infected_mask = _status_mask(statuses, true)
    uninfected_mask = _status_mask(statuses, false)
    unknown_mask = full ⊻ infected_mask ⊻ uninfected_mask

    # Every final set consistent with what was observed: the known infected,
    # together with any subset of the unknown members.
    total = 0.0
    s = unknown_mask
    while true
        A = infected_mask | s
        Ac = full ⊻ A
        total += piProd[Ac + 1] * _cross_escape(p, A, Ac) * H[A + 1]
        s == 0 && break
        s = (s - 1) & unknown_mask
    end
    return log(total)
end

"""
    final_status_loglik(model::HouseholdProcess, statuses::AbstractVector;
                        window, susceptibility = 1.0, infectiousness = 1.0) -> Float64
    final_status_loglik(model::ModelSpec{<:HouseholdProcess}, statuses::AbstractVector;
                        susceptibility = 1.0, infectiousness = 1.0) -> Float64

Log-likelihood of a household study's final statuses — who in each household
was ever infected by the end of follow-up, with `missing` for anyone not
followed up — under `model`'s community hazard and within-household kernel,
with no infection times observed at all.

`statuses` holds one entry per person, in the same order as `model`'s members
(as `household_of` numbers them): `true` for someone infected by the end of
follow-up, `false` for someone who was not, `missing` for someone whose status
was not recorded. `window` is the length of every case's infectious period, a
fixed number or a distribution, as [`household_final_size`](@ref)'s `window`
is. For a `ModelSpec`, it instead comes from the progression, as
[`household_offspring`](@ref)'s exact calculation reads it, and reading it
that way fails the same way theirs does: with more than one delay closing the
infectious period, or a probability gate on the way there.

`susceptibility` and `infectiousness` give each person's own multiplier on the
rate of infection they face and cause, a number shared by everyone or a vector
with one entry per person, exactly as [`HouseholdProcess`](@ref) reads them
from its own population in simulation.

`model` needs a community `external_hazard` with a finite `obs_end`: without
one, nothing seeds a household's outbreak at all, since every household here
starts from scratch rather than from a fixed set of index cases.

# How it is calculated

A community contact reaches each susceptible independently over `obs_end`,
and an infectious case's contact with each household-mate over the window
either happens or not, independently of every other such contact (Ball 1986).
Together these fix, for any subset of a household, the probability that it is
exactly who ends up infected: the product of everyone outside the subset
escaping the community, nobody in the subset ever reaching outside it, and the
subset being fully reachable, by those within it, from whoever the community
reaches directly. The last piece is itself a triangular recursion over the
subset's own subsets, so computing it for a household of `n` costs about 3ⁿ;
scoring `statuses` sums the resulting probability over every subset consistent
with what was observed, `2^k` of them for `k` members of unknown status.

# Example

```julia
using EpiHouseholds, EpiBranch, Distributions

process = HouseholdProcess([3], Exponential(1 / 0.3); external_hazard = 0.1, obs_end = 1.0)
final_status_loglik(process, [true, false, missing]; window = 6.0)
```
"""
function final_status_loglik(
        model::HouseholdProcess, statuses::AbstractVector;
        window, susceptibility = 1.0, infectiousness = 1.0
    )
    n = length(model.household_of)
    length(statuses) == n || throw(
        ArgumentError("statuses must have one entry per person ($n), got $(length(statuses))")
    )
    _check_trait_length(susceptibility, n, "susceptibility")
    _check_trait_length(infectiousness, n, "infectiousness")
    _ext_active(model.external_hazard) || throw(
        ArgumentError(
            "final_status_loglik needs a community `external_hazard`: without one " *
                "nothing seeds a household's outbreak, since every household here " *
                "starts from scratch rather than from a fixed set of index cases"
        )
    )
    isfinite(model.obs_end) || throw(
        ArgumentError(
            "an external hazard needs a finite `obs_end` (an unbounded window seeds " *
                "every household member); build the process with e.g. `obs_end = 30.0`"
        )
    )

    ll = 0.0
    for mem in model.members
        m = length(mem)
        pi_esc = [
            _community_escape(
                model.external_hazard, mem[k], model.obs_end,
                _trait_value(susceptibility, mem[k])
            )
                for k in 1:m
        ]
        p = ones(Float64, m, m)
        for i in 1:m, j in 1:m
            i == j && continue
            kernel = _pair_escape_kernel(model.kernel, mem[i], mem[j])
            mult = _trait_value(infectiousness, mem[i]) * _trait_value(susceptibility, mem[j])
            p[i, j] = Float64(_escape(Float64, kernel, window, mult))
        end
        ll += _household_final_status_loglik(p, pi_esc, view(statuses, mem))
    end
    return ll
end

function final_status_loglik(
        model::ModelSpec{<:HouseholdProcess}, statuses::AbstractVector;
        susceptibility = 1.0, infectiousness = 1.0
    )
    window = _window_length_law(model)
    window === nothing && throw(
        ArgumentError(
            "final_status_loglik needs a single fixed-length (or fixed-distribution) " *
                "infectious window; pass `window` directly to " *
                "`final_status_loglik(model.process, statuses; window, ...)` instead, " *
                "or simplify the progression to a single delay out of the infectious state"
        )
    )
    return final_status_loglik(
        model.process, statuses; window, susceptibility, infectiousness
    )
end
