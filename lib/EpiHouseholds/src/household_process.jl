# ── HouseholdProcess ─────────────────────────────────────────────────
#
# A household-structured transmission model: the population is partitioned into
# households and, within a household, every infectious member can infect every
# susceptible household-mate. The time from a member's infectiousness onset to
# infectious contact with a household-mate — the contact interval (Kenah 2011) —
# is drawn from `kernel`, the one required input; transmission happens only
# while the infector is infectious, the window the `progression` opens at `from`
# and closes at the earliest of the `until` states.
#
# HouseholdProcess is a structure-driven EpiBranch `TransmissionModel`: it shares
# the natural-history timeline (`progression`/`Transition`), interventions,
# attributes and observation, and brings its own continuous-time (Sellke)
# simulator and pairwise likelihood — the part the generation-based engine
# cannot reproduce exactly for a finite, depleting clique.

"""
    HouseholdProcess(sizes, kernel; from = nothing, until = (:recovered, :died, :isolated),
                     external_hazard = 0.0, obs_end = Inf)

Household-structured transmission. `sizes` gives the size of each household (so
`sum(sizes)` individuals in `length(sizes)` households) and `kernel` is the
within-household **contact interval** — the one required input — any continuous
`Distributions.jl` distribution on the positive reals, or a callable
`(infector, susceptible) -> Distribution` for covariate models. The kernel times
each infectious contact from the infector's `from` state.

The process describes the transmission alone. The natural history is a `progression`
of EpiBranch `Transition`s attached with a [`ModelSpec`](@ref): a latent period
is `Transition(:infectious; from = :infection, delay = …)`, an infectious period
a terminal removal transition, and onset, testing and the rest are further
transitions the line list reads. `from` is the state the kernel times contacts
from; left as `nothing` it is derived from the progression (`:infectious` when a
latent period produces it, otherwise `:infection`). `until` names the removal
states that close the infectious window.

`external_hazard` is the community force of infection — a scalar for a constant
hazard or a calendar-time distribution for a time-varying one — and `obs_end`
bounds the window `[0, obs_end]` over which those community introductions emerge.

Interventions attach through the infectious window. An `Isolation` intervention
removes a case from transmission at its isolation time, shortening the window and
cutting secondary cases; an intervention whose effect is a per-contact competing
risk (contact tracing, leaky vaccination) has no representation on the
continuous-time path and is reported with a warning rather than applied.
Non-pharmaceutical control expressed as a removal `Transition` in the progression
always applies.

# Example

```julia
using EpiHouseholds, EpiBranch, Distributions
model = ModelSpec(HouseholdProcess([3, 4, 2], Weibull(1.5, 3.0));
    progression = [Transition(:recovered; from = :infection, delay = 6.0, terminal = true)])
```
"""
struct HouseholdProcess{K, E} <: TransmissionModel
    household_of::Vector{Int}        # household_of[i] = household id of individual i
    members::Vector{Vector{Int}}     # members[h] = individual ids in household h
    kernel::K                        # within-household contact interval (required)
    from::Union{Symbol, Nothing}     # infectious-window start; nothing → derive from progression
    until::Tuple                     # removal states that close the infectious window
    external_hazard::E               # community force of infection (0 = none)
    obs_end::Float64                 # end of the community-importation window
end

function HouseholdProcess(sizes::AbstractVector{<:Integer}, kernel;
        from = nothing,
        until = (:recovered, :died, :isolated),
        external_hazard = 0.0,
        obs_end = Inf)
    all(s -> s >= 1, sizes) || throw(ArgumentError("household sizes must be ≥ 1"))
    _valid_external(external_hazard) ||
        throw(ArgumentError("external_hazard must be a non-negative number or a continuous distribution"))

    household_of = Int[]
    members = Vector{Int}[]
    id = 0
    for (h, sz) in enumerate(sizes)
        mem = Int[]
        for _ in 1:sz
            id += 1
            push!(household_of, h)
            push!(mem, id)
        end
        push!(members, mem)
    end

    return HouseholdProcess(household_of, members, kernel, from, Tuple(until),
        _normalise_external(external_hazard), Float64(obs_end))
end

"""
    household_sizes(model) -> Vector{Int}

The size of each household in `model`.
"""
household_sizes(m::HouseholdProcess) = length.(m.members)

# Each household runs to extinction over its finite membership, so the
# termination controls do not apply; `simulate` warns if any is set.
_honours_termination_controls(::HouseholdProcess) = false

function Base.show(io::IO, m::HouseholdProcess)
    n = length(m.household_of)
    nh = length(m.members)
    from = m.from === nothing ? "" : ", from=:$(m.from)"
    print(io, "HouseholdProcess($nh households, $n individuals, ",
        "kernel=$(m.kernel isa Distribution ? nameof(typeof(m.kernel)) : "Function")",
        from,
        _ext_active(m.external_hazard) ? ", external_hazard=$(m.external_hazard))" : ")")
end

# ── Helpers (public-surface only) ────────────────────────────────────

# The external community source: a non-negative scalar (constant hazard) or any
# continuous distribution (a calendar-time hazard). `_ext_active` separates "no
# source" (a zero scalar) from a real one.
_valid_external(α::Real) = α >= 0
# A calendar-time hazard must live on the non-negative reals: introductions
# cannot happen before time 0, so reject distributions with negative support.
_valid_external(d::ContinuousUnivariateDistribution) = minimum(d) >= 0
_valid_external(_) = false
_normalise_external(α::Real) = Float64(α)
_normalise_external(d::ContinuousUnivariateDistribution) = d

_ext_active(α::Real) = α > 0
_ext_active(::ContinuousUnivariateDistribution) = true
