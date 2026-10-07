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

Transmission within households: each infectious person infects their household
members after a random delay, provided they are still infectious by then, and
infections from outside the household arrive at a community rate. `sizes`
gives the size of each household, so there are `sum(sizes)` people in
`length(sizes)` households.

`kernel` is the within-household **contact interval**, the delay in days from
a person becoming infectious to their infecting contact with a household
member. It can be any continuous distribution on the positive reals, a function
`(infector, susceptible) -> Distribution` of the two people's numbers for
covariates, or a [`PairKernel`](@ref), which can also use the infector's
infection time and each person's attributes and history.

The natural history is a `progression` of `Transition`s attached with a
[`ModelSpec`](@ref). A latent period is
`Transition(:infectious; from = :infection, delay = …)` and the infectious
period ends with a terminal removal transition; onset, testing and the rest are
further transitions that the line list reports. `from` is the state from which
the contact interval is measured. Left as `nothing`, it is `:infectious` when a
latent period produces that state and `:infection` otherwise. `until` names
the states that end the infectious period.

`external_hazard` is infection from the community: a constant rate per person
per day, or a distribution of the time (in days) at which each person would be
infected from outside, for a rate that changes over time. These introductions
happen only in the first `obs_end` days, which must be finite when
`external_hazard` is used. Without an external hazard, each household starts
with one index case at day 0.

!!! warning "What isolation means here"
    The only transmission this process represents is *within* a household;
    community infection enters as `external_hazard` introductions, and there is
    no contact between households. An intervention that ends a case's
    infectious period here therefore stops them infecting their own household,
    which physically means removing them from the household: hospitalisation,
    or transfer to an isolation facility. It does **not** model self-isolation
    at home, which would leave household transmission going and, in most
    settings, increase it. Nor is there a community route for a case to be
    isolated *from* while they stay infectious to the people they live with,
    which is what self-isolation does. Representing that needs transmission
    split into a household route and a community route, so a control measure
    can cut one and leave the other; see the route windows in the
    [design notes](@ref "Host timeline and transmission-route windows").
    Read isolation and quarantine on this process as removal from the
    household, and choose the parameters accordingly.

Interventions on this process:

- `Isolation` removes a case from their isolation time for its `duration`,
  cutting their secondary cases; with `duration = Inf` this ends their
  infectious period. Leaky isolation multiplies the rate at which each
  household contact is infected by `post_isolation_transmission` from the
  isolation time, which lowers their chance of infection by less than that.
- `ContactTracing` treats a case's household members as their contacts;
  quarantining a traced contact stops that contact's transmission for the
  quarantine's `duration`.
  Tracing can only reach household members infected after the case's own
  course of infection is known, which in a fast-spreading household means the
  ones infected later.
- A leaky vaccine's efficacy, and per-person susceptibility and
  infectiousness, reduce the rate at which each household contact is
  infected. An all-or-nothing vaccine instead fully protects its share of the
  vaccinated from their immunity date.
- `RingVaccination` and `GroupVaccination` work, including with `Scheduled`
  and `CapacityConstrained`, and a ring dose's `post_exposure_efficacy` can stop
  a household member's own infection before onset. A capacity limit is one
  budget shared by all households, renewed every `period` days of the
  outbreak.
- A removal `Transition` in the progression always applies.

Limits:

- `MassVaccination` is not supported.
- `RingVaccination` needs `eligibility_window = Inf`, because a household
  member's own infection time is not yet known when the dose is decided.
- Existing protection can instead be given through `transmission_traits`, a
  [`PairKernel`](@ref), or an intervention with its own `competing_risk`.

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

function HouseholdProcess(
        sizes::AbstractVector{<:Integer}, kernel;
        from = nothing,
        until = (:recovered, :died, :isolated),
        external_hazard = 0.0,
        obs_end = Inf
    )
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

    return HouseholdProcess(
        household_of, members, kernel, from, Tuple(until),
        _normalise_external(external_hazard), Float64(obs_end)
    )
end

"""
    household_sizes(model) -> Vector{Int}

The size of each household in `model`.
"""
household_sizes(m::HouseholdProcess) = length.(m.members)

# Each household runs over its finite membership until extinction or
# `max_time`; the other termination controls do not apply, and `simulate` warns
# if any is set.
_honours_termination_controls(::HouseholdProcess) = false

# See `_warn_uncovered_terminal_states` in EpiBranch's branching_process.jl.
function _validate_process_windows(m::HouseholdProcess, progression)
    return _warn_uncovered_terminal_states(m.until, progression; from = m.from)
end

# A case's contacts are its household-mates, so contact tracing has a set to act
# along here (see `EpiBranch.trace_contacts!`).
EpiBranch.supplies_contacts(::HouseholdProcess) = true

function Base.show(io::IO, m::HouseholdProcess)
    n = length(m.household_of)
    nh = length(m.members)
    from = m.from === nothing ? "" : ", from=:$(m.from)"
    return print(
        io, "HouseholdProcess($nh households, $n individuals, ",
        "kernel=$(m.kernel isa Distribution ? nameof(typeof(m.kernel)) : "Function")",
        from,
        _ext_active(m.external_hazard) ? ", external_hazard=$(m.external_hazard))" : ")"
    )
end
