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
    HouseholdProcess(sizes, kernel; infectious_period = nothing, latent_period = nothing,
                     progression = [], from = nothing, until = (:recovered, :died, :isolated),
                     external_hazard = 0.0, interventions = [], attributes = NoAttributes(),
                     observation = NoObservation())

Household-structured transmission. `sizes` gives the size of each household (so
`sum(sizes)` individuals in `length(sizes)` households) and `kernel` is the
within-household **contact interval** — the one required input — any continuous
`Distributions.jl` distribution on the positive reals, or a callable
`(infector, susceptible) -> Distribution` for covariate models. The kernel times
each infectious contact from the infector's `from` state.

The infectious timeline is a flexible `progression` of EpiBranch `Transition`s,
exactly as for `BranchingProcess`: a latent period is
`Transition(:infectious, from = :infection, delay = …)`, an infectious period a
removal transition (`Transition(:recovered, from = :infectious, delay = …)`),
and symptom onset, testing and the rest are further transitions the line list
reads. `from` is the state the kernel times from (`:infectious` when the
timeline has a latent period, `:infection` otherwise); `until` names the removal
states that close the infectious window.

`infectious_period` and `latent_period` are sugar for the two common
transitions: each desugars into the corresponding `Transition` when no explicit
`progression` is given, and accepts a scalar (constant) or a `Distribution`
(per-case). `external_hazard` is the community force of infection — a scalar for
a constant hazard or a calendar-time distribution for a time-varying one.

# Example

```julia
using EpiHouseholds, Distributions
model = HouseholdProcess([3, 4, 2], Weibull(1.5, 3.0); infectious_period = 6.0)
```
"""
struct HouseholdProcess{K, E, A, O <: ObservationModel} <: TransmissionModel
    household_of::Vector{Int}        # household_of[i] = household id of individual i
    members::Vector{Vector{Int}}     # members[h] = individual ids in household h
    kernel::K                        # within-household contact interval (required)
    from::Symbol                     # state the kernel times contacts from
    until::Tuple                     # removal states that close the infectious window
    progression::Vector{AbstractClinicalTransition}  # natural-history timeline
    external_hazard::E               # community force of infection (0 = none)
    interventions::Vector{AbstractIntervention}
    attributes::A
    observation::O
end

function HouseholdProcess(sizes::AbstractVector{<:Integer}, kernel;
        infectious_period = nothing,
        latent_period = nothing,
        progression = AbstractClinicalTransition[],
        from = nothing,
        until = (:recovered, :died, :isolated),
        external_hazard = 0.0,
        interventions = AbstractIntervention[],
        attributes = NoAttributes(),
        observation::ObservationModel = NoObservation())
    all(s -> s >= 1, sizes) || throw(ArgumentError("household sizes must be ≥ 1"))
    _valid_external(external_hazard) ||
        throw(ArgumentError("external_hazard must be a non-negative number or a continuous distribution"))

    prog = convert(Vector{AbstractClinicalTransition}, progression)
    # Desugar the common timeline shorthands when no explicit progression given:
    # a latent period to :infectious, an infectious period to a recovery removal.
    if isempty(prog)
        latent_period === nothing ||
            push!(prog, Transition(:infectious; from = :infection, delay = _as_delay(latent_period)))
        anchor = latent_period === nothing ? :infection : :infectious
        infectious_period === nothing ||
            push!(prog,
                Transition(:recovered; from = anchor, delay = _as_delay(infectious_period),
                    terminal = true))
    end

    # The kernel times contacts from `from`: default to :infectious when the
    # timeline produces it (a latent period), otherwise from infection.
    kfrom = from === nothing ?
            (any(t -> _writes_state(t, :infectious), prog) ? :infectious : :infection) :
            from

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

    return HouseholdProcess(household_of, members, kernel, kfrom, Tuple(until),
        prog, _normalise_external(external_hazard), _intervention_list(interventions),
        attributes, observation)
end

"""
    household_sizes(model) -> Vector{Int}

The size of each household in `model`.
"""
household_sizes(m::HouseholdProcess) = length.(m.members)

# The interventions/attributes/observation the model carries.
interventions(m::HouseholdProcess) = m.interventions
attributes(m::HouseholdProcess) = m.attributes
observation(m::HouseholdProcess) = m.observation

function Base.show(io::IO, m::HouseholdProcess)
    n = length(m.household_of)
    nh = length(m.members)
    print(io, "HouseholdProcess($nh households, $n individuals, ",
        "kernel=$(m.kernel isa Distribution ? nameof(typeof(m.kernel)) : "Function"), ",
        "from=:$(m.from), $(length(m.progression)) transitions",
        _ext_active(m.external_hazard) ? ", external_hazard=$(m.external_hazard))" : ")")
end

# ── Helpers (public-surface only) ────────────────────────────────────

# A `Transition` delay is a `Distribution` or `(rng, ind) -> Real`; a scalar
# period becomes a constant-delay closure so the sugar accepts both.
_as_delay(d::Distribution) = d
_as_delay(x::Real) = (rng, ind) -> float(x)

# Does a clinical transition write the named state? Mirrors how EpiBranch
# decides which states a progression produces, using only the public field.
_writes_state(t, s::Symbol) = hasproperty(t, :state) && getfield(t, :state) === s

# Normalise the interventions argument to a vector without reaching into the
# EpiBranch internal `_intervention_vector`.
_intervention_list(iv::AbstractVector) = convert(Vector{AbstractIntervention}, iv)
_intervention_list(iv) = AbstractIntervention[iv]

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
