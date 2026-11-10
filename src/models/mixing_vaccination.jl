# ── Vaccination in the approximations for mixing populations ─────────
#
# `MixingMeanFieldProcess` and `MixingLinearNoiseProcess`
# (mixing_population_dynamics.jl) take a `GroupVaccination` campaign from the
# `ModelSpec` they approximate and turn it into doses into each mixing
# group's compartments. Everything the intervention already defines is read
# from it (`group_key`, `efficacy` and `mode`, `coverage`, `dose_delay`, the
# window of a `Scheduled` wrapper and the budget of a `CapacityConstrained`
# one), so nothing is written down twice. What the exact model settles
# through individual cases, which a compartment cannot, is the one thing the
# caller chooses: how a budget of doses is shared out across groups, a
# `DoseAllocation`.

"""
    DoseAllocation

How a campaign's doses are shared across the groups of a
[`MixingMeanFieldProcess`](@ref) or [`MixingLinearNoiseProcess`](@ref). The
exact model gives each group's doses to its own members, once a case has
triggered the group, so a mean-field compartment has to be told how that
discrete trigger, and the sharing of a budget between triggered groups,
become a dose rate. The built-in choices are [`TriggeredInPriority`](@ref)
(the default, following the exact model), [`SplitByGroupSize`](@ref) and
[`SplitBySusceptibles`](@ref).

For extension authors: a new allocation is a subtype with methods of
`EpiBranch._dose_rate` (doses per day into each mixing group, for allocations
that give doses continuously), `EpiBranch._trigger_threshold` (the expected
number of cases that triggers a group, or `nothing` if triggers are not
read), `EpiBranch._next_dose_event` (the next time the integration must land
on exactly, such as a budget period's end) and `EpiBranch._check_allocation`
(what the campaign must hold for the allocation to be defined).
"""
abstract type DoseAllocation end

"""
    TriggeredInPriority(; threshold = 1.0)

The default [`DoseAllocation`](@ref), following the exact model. A group is
triggered once its expected number of eligible cases reaches `threshold`
(one, by default), counting every infection in the group as an eligible case
at the time it happens, and its members are all offered a dose `dose_delay`
days later. Triggered groups draw on a shared budget in the order they were
triggered, each taking as many doses as it has people left to reach and the
budget has left, as the exact model serves groups in the order their cases
are settled. A group that finds the budget spent is not offered again. Each
member is offered once, so a budget that cannot cover a group leaves the
rest of it unvaccinated.

Which cases are eligible, and how long an eligible case takes to count (to
onset or to a positive test), is not modelled: a mean field has no individual
cases to ask. The group's trigger time is the one thing `eligibility` and its
delays decide in the exact model, so a [`HybridProcess`](@ref) passes the
trigger times its exact phase already knows to this allocation when it
switches.

The priority function of a [`CapacityConstrained`](@ref) wrapper orders
members within a group, which makes no difference to a compartment of
exchangeable people, and groups are not interleaved by it, since a group's
members are all offered at the same time.
"""
struct TriggeredInPriority <: DoseAllocation
    threshold::Float64
    function TriggeredInPriority(threshold::Real)
        (isfinite(threshold) && threshold > 0) ||
            throw(ArgumentError("threshold must be finite and positive, got $threshold"))
        return new(Float64(threshold))
    end
end
TriggeredInPriority(; threshold::Real = 1.0) = TriggeredInPriority(threshold)

"""
    SplitByGroupSize()

A [`DoseAllocation`](@ref) that spreads the budget of a
[`CapacityConstrained`](@ref) campaign over every group with people left to
reach, in proportion to group size, as a daily rate. It ignores the trigger,
`dose_delay` and `eligibility`: doses flow into every group from the start of
the budget's window. It needs a campaign with a finite budget `period`, so a
budget per period is a daily rate to split.
"""
struct SplitByGroupSize <: DoseAllocation end

"""
    SplitBySusceptibles()

As [`SplitByGroupSize`](@ref), but in proportion to each group's number of
unvaccinated susceptibles, so the groups with the most people still at risk
get the most doses.
"""
struct SplitBySusceptibles <: DoseAllocation end

# ── What a campaign does to a dosed susceptible ──────────────────────

# How one dose acts on a susceptible: the share who become fully immune, the
# share who move to the vaccinated-susceptible compartment, and that
# compartment's hazard relative to an unvaccinated susceptible's. A leaky
# vaccine leaves everyone dosed susceptible at `1 - efficacy` of the hazard;
# an all-or-nothing one makes a share `efficacy` immune and leaves the rest
# exactly as exposed as before. A new mode defines a method of its own, and
# one without a method is refused rather than approximated by another.
function _dose_outcome(::LeakyMode, efficacy)
    return (immune = 0.0, vaccinated = 1.0, hazard = 1.0 - efficacy)
end
function _dose_outcome(::AllOrNothingMode, efficacy)
    return (immune = efficacy, vaccinated = 1.0 - efficacy, hazard = 1.0)
end
function _dose_outcome(mode, efficacy)
    throw(
        ArgumentError(
            "the approximations for mixing populations have no dose model for " *
                "`mode = $(mode)`; define `EpiBranch._dose_outcome` for it, or use " *
                "`LeakyMode` or `AllOrNothingMode`"
        )
    )
end

# ── The campaign read from a ModelSpec ───────────────────────────────

# A model with no vaccination: nothing flows between compartments.
struct NoCampaign end

# A `GroupVaccination`, with the groups its doses go to and the gates the
# wrappers around it put on them. `group_of[m]` is the vaccination group
# (a value of `group_key`) mixing group `m` belongs to, `members[v]` the
# mixing groups in vaccination group `v`. `budget` is the
# `CapacityConstrained` wrapper, or `nothing` for unlimited doses.
struct GroupCampaign{B, A <: DoseAllocation, GV <: GroupVaccination}
    vaccination::GV
    key::Symbol
    values::Vector{Any}
    group_of::Vector{Int}
    members::Vector{Vector{Int}}
    immune::Float64
    vaccinated::Float64
    hazard::Float64
    coverage::Float64
    delay::Float64
    window_start::Float64
    window_stop::Float64
    budget::B
    allocation::A
end

# Peel `Scheduled` and `CapacityConstrained` wrappers off an intervention,
# keeping what each says about when and how many doses are given.
function _unwrap_campaign(iv, window, budget)
    if iv isa Scheduled
        w = _clock_window(iv)
        return _unwrap_campaign(
            iv.intervention, (max(window[1], w.start), min(window[2], w.stop)), budget
        )
    elseif iv isa CapacityConstrained
        budget === nothing || throw(
            ArgumentError("a campaign with more than one CapacityConstrained is not supported")
        )
        iv.on_refusal isa DropRefused || throw(
            ArgumentError(
                "a campaign with `on_refusal = $(nameof(typeof(iv.on_refusal)))` is " *
                    "not supported: the approximations assume each member is offered " *
                    "a dose once, as under `DropRefused()`"
            )
        )
        return _unwrap_campaign(iv.intervention, window, iv)
    end
    return iv, window, budget
end

_clock_window(s::Scheduled) = _clock_window(s.window)
_clock_window(window::ClockWindow) = window
function _clock_window(::Nothing)
    throw(
        ArgumentError(
            "a Scheduled campaign gated on cases, a trigger or a predicate has no " *
                "calendar window for the approximation to read; give it `start_time` " *
                "and/or `end_time` alone"
        )
    )
end

_check_allocation(::TriggeredInPriority, budget) = nothing
function _check_allocation(alloc::Union{SplitByGroupSize, SplitBySusceptibles}, budget)
    (budget isa CapacityConstrained && isfinite(budget.period)) || throw(
        ArgumentError(
            "$(nameof(typeof(alloc))) shares a budget per period out as a daily " *
                "rate, so it needs the campaign under a CapacityConstrained with a " *
                "finite `period`"
        )
    )
    return nothing
end

_read_campaign(spec, mixing_by, groups, allocation, what) =
    _read_campaign(interventions(spec), mixing_by, groups, allocation, what)
_read_campaign(::Nothing, args...) = NoCampaign()
function _read_campaign(ivs::AbstractVector, mixing_by, groups, allocation, what)
    isempty(ivs) && return NoCampaign()
    length(ivs) == 1 || throw(
        ArgumentError(
            "$what reads one GroupVaccination from a model's interventions, " *
                "but this one has $(length(ivs)) interventions"
        )
    )
    iv, window, budget = _unwrap_campaign(only(ivs), (-Inf, Inf), nothing)
    iv isa GroupVaccination || throw(
        ArgumentError(
            "$what reads vaccination from a GroupVaccination, alone or under " *
                "Scheduled and CapacityConstrained, and cannot approximate a model " *
                "with a $(nameof(typeof(iv))), which would be silently left out"
        )
    )
    key = iv.group_key
    position = findfirst(==(key), mixing_by)
    position === nothing && throw(
        ArgumentError(
            "the GroupVaccination's group_key = :$key is not one of the model's " *
                "mixing_by keys $mixing_by, so its doses cannot be matched to the " *
                "mixing groups the compartments are kept for"
        )
    )
    _check_dose_inputs(iv)
    outcome = _dose_outcome(effect_mode(iv), efficacy(iv))
    _check_allocation(allocation, budget)
    values = Any[]
    group_of = Int[]
    for g in groups
        v = findfirst(isequal(g[position]), values)
        if v === nothing
            push!(values, g[position])
            v = length(values)
        end
        push!(group_of, v)
    end
    members = [findall(==(v), group_of) for v in eachindex(values)]
    return GroupCampaign(
        iv, key, values, group_of, members,
        Float64(outcome.immune), Float64(outcome.vaccinated), Float64(outcome.hazard),
        Float64(iv.coverage), Float64(iv.dose_delay), Float64(window[1]), Float64(window[2]),
        budget, allocation
    )
end

# The inputs a compartment can only take as numbers: a distribution or a
# function would vary person by person, which an average cannot stand in for
# without a choice the caller has not made.
function _check_dose_inputs(iv::GroupVaccination)
    for (name, x, lo, hi) in (
            ("efficacy", efficacy(iv), 0.0, 1.0), ("coverage", iv.coverage, 0.0, 1.0),
            ("dose_delay", iv.dose_delay, 0.0, Inf),
        )
        x isa Real && lo <= x <= hi || throw(
            ArgumentError(
                "the approximations read `$name` as one number between $lo and $hi, " *
                    "got $x"
            )
        )
    end
    delay = delay_to_immunity(iv)
    (delay isa Real && delay == 0) || throw(
        ArgumentError(
            "the approximations do not model `delay_to_immunity`: a dose protects " *
                "from the day it is given, so it must be 0, got $delay"
        )
    )
    waning(iv) === nothing || throw(
        ArgumentError("the approximations do not model `waning`, so it must be `nothing`")
    )
    return nothing
end

# ── Progress of the campaign through time ────────────────────────────

# Where a campaign has got to, held in each state so a continued run picks
# it up. `dose_time[v]` is when vaccination group `v`'s doses are due (`Inf`
# until it is triggered) and `fired[v]` whether they have been offered;
# `baseline[v]` is the expected number of cases already counted when the
# trigger started to be watched for; `log` holds each batch of doses as
# `(time, doses)`, which a budget with `carry_over = false` reads back.
struct CampaignProgress
    dose_time::Vector{Float64}
    fired::Vector{Bool}
    baseline::Vector{Float64}
    log::Vector{Tuple{Float64, Float64}}
end

_new_progress(::NoCampaign) = CampaignProgress(Float64[], Bool[], Float64[], Tuple{Float64, Float64}[])
function _new_progress(c::GroupCampaign)
    n = length(c.values)
    return CampaignProgress(fill(Inf, n), fill(false, n), zeros(n), Tuple{Float64, Float64}[])
end

function _copy(p::CampaignProgress)
    return CampaignProgress(copy(p.dose_time), copy(p.fired), copy(p.baseline), copy(p.log))
end

# ── Doses given continuously ─────────────────────────────────────────

# Doses per day into each mixing group at time `t`, for the allocations that
# give doses as a rate; none for `TriggeredInPriority`, which gives them in
# batches (see `_fire_doses!`).
_dose_rate(::NoCampaign, model, t, S, D) = zeros(length(S))
_dose_rate(c::GroupCampaign, model, t, S, D) = _dose_rate(c.allocation, c, model, t, S, D)
_dose_rate(::TriggeredInPriority, c, model, t, S, D) = zeros(length(S))

function _dose_rate(alloc::Union{SplitByGroupSize, SplitBySusceptibles}, c, model, t, S, D)
    G = length(S)
    c.window_start <= t <= c.window_stop || return zeros(G)
    cc = c.budget
    period_start = floor(t / cc.period + 1.0e-9) * cc.period
    total = _period_budget(cc.budget_per_period, period_start) / cc.period
    N = Float64.(model.group_sizes)
    reachable = c.coverage .* N .- D .> 1.0e-9
    weight = _split_weight(alloc, N, S) .* reachable
    sum(weight) > 0 || return zeros(G)
    return total .* weight ./ sum(weight)
end
_split_weight(::SplitByGroupSize, N, S) = N
_split_weight(::SplitBySusceptibles, N, S) = max.(S, 0.0)

_trigger_threshold(::NoCampaign) = nothing
_trigger_threshold(c::GroupCampaign) = _trigger_threshold(c.allocation)
_trigger_threshold(a::TriggeredInPriority) = a.threshold
_trigger_threshold(::Union{SplitByGroupSize, SplitBySusceptibles}) = nothing

# The next time after `t` that the integration should land on exactly,
# because something about the dosing changes there: a batch of doses falling
# due, or, for a rate, a new budget period or the edge of the window.
_next_dose_event(::NoCampaign, progress, t) = Inf
function _next_dose_event(c::GroupCampaign, progress, t)
    return _next_dose_event(c.allocation, c, progress, t)
end
function _next_dose_event(::TriggeredInPriority, c, progress, t)
    next = Inf
    for v in eachindex(progress.dose_time)
        progress.fired[v] && continue
        d = progress.dose_time[v]
        d > t + 1.0e-9 && (next = min(next, d))
    end
    return next
end
function _next_dose_event(::Union{SplitByGroupSize, SplitBySusceptibles}, c, progress, t)
    period = c.budget.period
    next = (floor(t / period + 1.0e-9) + 1.0) * period
    for edge in (c.window_start, c.window_stop)
        edge > t + 1.0e-9 && (next = min(next, edge))
    end
    return next
end

# ── Doses given in batches ───────────────────────────────────────────

# Doses left in the budget for a batch dated `d`, counted exactly as the
# `CapacityConstrained` wrapper counts them for one more person: what is
# available through the end of the period `d` falls in (with `carry_over`,
# everything unspent before it too), less what has been given.
_budget_left(::Nothing, log, d) = Inf
function _budget_left(cc::CapacityConstrained, log, d)
    period_start, period_end = _period_bounds(cc, d)
    if cc.carry_over
        available = isinf(cc.period) ? _period_budget(cc.budget_per_period, period_start) :
            _cumulative_budget(cc, period_end)
        used = sum((n for (t, n) in log if t < period_end); init = 0.0)
    else
        available = _period_budget(cc.budget_per_period, period_start)
        used = sum((n for (t, n) in log if period_start <= t < period_end); init = 0.0)
    end
    return max(available - used, 0.0)
end

# Give every group whose doses are due by `t` its batch, in the order they
# fell due. A group takes the people it can still reach (its `coverage` of
# the group, less those already dosed) or what the budget has left, and the
# batch is spread over the mixing groups in it in proportion to the people
# each can still reach. A dose reaches an unvaccinated susceptible with
# probability `S / (people not yet dosed)`, whatever the person's infection
# status, and then protects according to the vaccine's mode.
#
# `x` is the packed state of `_mixing_drift` and `Σ`, when there is one, the
# linear noise covariance, which a batch moves too.
_fire_doses!(::NoCampaign, model, x, Σ, progress, t) = nothing
function _fire_doses!(c::GroupCampaign, model, x, Σ, progress, t)
    _trigger_threshold(c) === nothing && return nothing
    G = length(model.groups)
    due = [
        v for v in eachindex(progress.dose_time)
            if !progress.fired[v] && progress.dose_time[v] <= t + 1.0e-9
    ]
    sort!(due; by = v -> (progress.dose_time[v], v))
    N = Float64.(model.group_sizes)
    for v in due
        progress.fired[v] = true
        d = progress.dose_time[v]
        c.window_start <= d <= c.window_stop || continue
        S, _, _, _, D = _unpack(x, G)
        reach = [max(c.coverage * N[m] - D[m], 0.0) for m in c.members[v]]
        given = min(sum(reach), _budget_left(c.budget, progress.log, d))
        given > 0 || continue
        for (k, m) in enumerate(c.members[v])
            _dose_group!(c, x, Σ, G, m, given * reach[k] / sum(reach), N[m])
        end
        push!(progress.log, (d, given))
    end
    return nothing
end

# `doses` people are dosed in mixing group `m`: the unvaccinated susceptibles
# among them move on, and the covariance takes the linear map of that move and
# the sampling noise of who is reached (a multinomial split of the susceptibles).
function _dose_group!(c::GroupCampaign, x, Σ, G, m, doses, size)
    S, _, V, P, D = _unpack(x, G)
    φ = doses / max(size - D[m], S[m], 1.0e-9)
    qv, qi = φ * c.vaccinated, φ * c.immune
    _jump_covariance!(Σ, G, m, S[m], qv, qi)
    moved = S[m] * (qv + qi)
    V[m] += S[m] * qv
    P[m] += S[m] * qi
    S[m] -= moved
    D[m] += doses
    return nothing
end

_jump_covariance!(::Nothing, G, m, S, qv, qi) = nothing
function _jump_covariance!(Σ::AbstractMatrix, G, m, S, qv, qi)
    n = 4G
    s, v, p = m, 2G + m, 3G + m
    # The linear map of the jump on the compartments (S, V, P) of this group.
    J = Matrix{Float64}(LinearAlgebra.I, n, n)
    J[s, s] = 1 - qv - qi
    J[v, s] = qv
    J[p, s] = qi
    # Y = (to V, to P) is multinomial from S, moving S by -(Y_v + Y_p).
    cov_yy = S .* [qv * (1 - qv) -qv * qi; -qv * qi qi * (1 - qi)]
    B = zeros(n, 2)
    B[s, :] .= -1.0
    B[v, 1] = 1.0
    B[p, 2] = 1.0
    Σ .= J * Σ * J' + B * cov_yy * B'
    Σ .= (Σ + Σ') / 2
    return nothing
end

# ── Triggering groups ────────────────────────────────────────────────

# Expected cases so far (everyone who has left the susceptibles other than
# by a dose or by having been immune), summed by vaccination group.
function _group_cases(c::GroupCampaign, model, x)
    G = length(model.groups)
    S, _, V, P, _ = _unpack(x, G)
    cases = Float64.(model.group_sizes) .- S .- V .- P
    return [sum(cases[m] for m in c.members[v]) for v in eachindex(c.values)]
end

# Schedule the doses of every group whose expected cases have crossed the
# trigger between the start and the end of a step, `dose_delay` after the
# crossing, found by interpolating linearly between the two.
_trigger_groups!(::NoCampaign, args...) = nothing
function _trigger_groups!(c::GroupCampaign, progress, before, after, t_before, t_after)
    threshold = _trigger_threshold(c)
    threshold === nothing && return nothing
    for v in eachindex(c.values)
        (progress.fired[v] || isfinite(progress.dose_time[v])) && continue
        b = before[v] - progress.baseline[v]
        a = after[v] - progress.baseline[v]
        a >= threshold || continue
        crossed = b >= threshold ? t_before :
            t_before + (t_after - t_before) * (threshold - b) / (a - b)
        progress.dose_time[v] = crossed + c.delay
    end
    return nothing
end

# ── Handing an exact run's campaign across a switch ──────────────────

# Whether `ind` has had a dose by `t` in an exact run, and whether it took
# (an all-or-nothing dose recorded as full protection), which is how a
# `HybridProcess` files the people already dosed in the exact phase.
_dose_given(::NoCampaign, ind, t) = false
function _dose_given(c::GroupCampaign, ind, t)
    dose_t = _dose_time(dose_label(c.vaccination), ind)
    return dose_t !== nothing && dose_t <= t
end

_dose_protects(::NoCampaign, ind) = false
function _dose_protects(c::GroupCampaign, ind)
    c.immune > 0 || return false
    eff = vaccine_efficacy(ind; dose_label = dose_label(c.vaccination))
    return eff isa Real && eff >= 1.0
end

# The campaign's progress as the exact run `es` knows it at `t`: a group the
# exact run has triggered keeps the trigger time it found (which is what
# `eligibility` and its delays decide), so its doses are due `dose_delay` after
# that, or already given if that is past. A group it has not triggered waits for
# one more expected case from here, rather than being triggered at once by the
# cases it has already seen.
_handover_progress(::NoCampaign, es, t, cases) = _new_progress(NoCampaign())
function _handover_progress(c::GroupCampaign, es, t, cases)
    progress = _new_progress(c)
    if _trigger_threshold(c) !== nothing
        for v in eachindex(c.values)
            trigger = _group_trigger_time(c.vaccination, es, c.values[v])
            if isfinite(trigger)
                progress.dose_time[v] = trigger + c.delay
                progress.fired[v] = progress.dose_time[v] <= t
            else
                progress.baseline[v] = sum(cases[m] for m in c.members[v])
            end
        end
    end
    given = Dict{Float64, Float64}()
    for ind in es.individuals
        dose_t = _dose_time(dose_label(c.vaccination), ind)
        dose_t !== nothing && dose_t <= t &&
            (given[dose_t] = get(given, dose_t, 0.0) + 1.0)
    end
    append!(progress.log, [(k, v) for (k, v) in sort!(collect(given))])
    return progress
end
