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
`(infector, susceptible) -> Distribution` for covariate models, or a
[`ContextualKernel`](@ref) that also reads the infector's infection time,
or a [`StatefulKernel`](@ref) with sampled attributes and dated histories. The kernel times
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

!!! warning "What isolation means here"
    The only transmission this process represents is *within* a household;
    community infection enters as unstructured `external_hazard`
    introductions, and there is no between-household contact. So an
    intervention that closes a case's infectious window here stops it
    infecting its own household-mates, which physically means removing it
    from the household — hospitalisation, or transfer to an isolation
    facility. It does **not** model self-isolation at home, which would
    leave household transmission running and, in most settings, raise it.
    Nor is there a community route for a case to be isolated *from* while it
    stays infectious to the people it lives with, which is what
    self-isolation actually does. Representing that needs transmission
    separated into a household route and a community route, so a control
    measure can cut one and leave the other; see the route windows in the
    [design notes](@ref "Host timeline and transmission-route windows").
    Read isolation and quarantine on this process as removal from the
    household, and size the parameters accordingly.

With that reading, interventions attach through the infectious window. An
`Isolation` intervention removes a case at its isolation time, shortening the
window and cutting secondary cases. `ContactTracing` also applies: a case's
household-mates are its contacts, and quarantining a traced contact closes that
contact's window in turn. Because a case's trace time is only known once the
race has settled its timeline, tracing reaches the contacts that are not yet
themselves settled, which in a fast-mixing household means the ones infected
later. An intervention whose effect is a per-contact competing risk, such as a
leaky isolation or a vaccine's efficacy, is resolved against each infection the
race proposes between household members; a blocked proposal is declined and the
pair goes on meeting, so blocking a fraction of the contacts thins that pair's
hazard by the same fraction. Per-individual susceptibility and infectiousness
reach the same thinning through the contact-interval draw, which turns a pair's
survival `S(t)` into `S(t)^m`. Ring and group vaccination use candidate actions,
including scheduling and capacity admission. With several households, capacity
requires `period = Inf`: each household runs on its own clock, which prevents
chronological accounting of a shared periodic budget. Ring delivery requires an infinite
eligibility window and zero post-exposure efficacy. Mass vaccination remains
unsupported on this path. Existing protection can also use host traits,
a composed kernel or a user-defined competing risk. Non-pharmaceutical control expressed as a removal
`Transition` in the progression always applies.

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

# See `_warn_uncovered_terminal_states` in EpiBranch's branching_process.jl.
function _validate_process_windows(m::HouseholdProcess, progression)
    _warn_uncovered_terminal_states(m.until, progression; from = m.from)
end

# A case's contacts are its household-mates, so contact tracing has a set to act
# along here (see `EpiBranch.trace_contacts!`).
EpiBranch.supplies_contacts(::HouseholdProcess) = true

function Base.show(io::IO, m::HouseholdProcess)
    n = length(m.household_of)
    nh = length(m.members)
    from = m.from === nothing ? "" : ", from=:$(m.from)"
    print(io, "HouseholdProcess($nh households, $n individuals, ",
        "kernel=$(m.kernel isa Distribution ? nameof(typeof(m.kernel)) : "Function")",
        from,
        _ext_active(m.external_hazard) ? ", external_hazard=$(m.external_hazard))" : ")")
end
