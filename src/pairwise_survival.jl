# ── Pairwise survival likelihood over a contact structure ────────────
#
# The contact-process density: the log-likelihood of an outbreak's *infection
# layer* (who is infected, and the latent infection time and infectious window
# of each) under a contact-interval kernel. It is the marginal pairwise
# likelihood of Kenah (2011). Who infected whom and the order of infections are
# both unobserved, and each infected susceptible's contribution sums the
# contact-interval hazard over every possible infector, with no ordering assumed.
#
# The density is a product over (susceptible, possible infector) pairs and does
# not depend on the kind of contact structure. Households, contact networks and
# any other relation of who could have infected whom use the same code and
# differ only in how the pairs are enumerated.

"""
    PairwiseSurvivalData(sus, start, stop, event)

Exposure data in survival-analysis form for [`pairwise_surv_loglik`](@ref): one
row per susceptible person and possible infector, giving the time over which
the person was exposed to that infector and whether they were infected at the
end of it. Row `r` covers the exposure interval `(start[r], stop[r]]` (days)
for susceptible person `sus[r]`, and `event[r]` is true if they were infected
at `stop[r]`. The rows record only exposure and infection, without who lives
where or the order of infections.
"""
struct PairwiseSurvivalData{T <: Real}
    sus::Vector{Int}
    start::Vector{T}
    stop::Vector{T}
    event::Vector{Bool}
    function PairwiseSurvivalData{T}(sus, start, stop, event) where {T <: Real}
        n = length(sus)
        (length(start) == n && length(stop) == n && length(event) == n) ||
            throw(ArgumentError("sus, start, stop and event must be the same length"))
        all(start[r] <= stop[r] for r in 1:n) ||
            throw(ArgumentError("each row needs start ≤ stop"))
        return new{T}(Int.(sus), Vector{T}(start), Vector{T}(stop), Bool.(event))
    end
end

# Promote the time type so callers don't have to spell it out; integer inputs
# widen to Float64.
function PairwiseSurvivalData(sus, start, stop, event)
    T = promote_type(eltype(start), eltype(stop), Float64)
    return PairwiseSurvivalData{T}(sus, start, stop, event)
end

Base.length(d::PairwiseSurvivalData) = length(d.sus)

# Row r's contact-interval distribution: a shared distribution, or a per-row
# callable `r -> Distribution` through which covariates enter.
_rowkernel(k::ContinuousUnivariateDistribution, r) = k
_rowkernel(k, r) = k(r)
function _rowkernel(::PairKernel, r)
    throw(
        ArgumentError(
            "PairKernel requires an InfectionLayer with source times; " *
                "for counting-process rows, supply a row-indexed kernel with those data"
        )
    )
end

"""
    pairwise_surv_loglik(kernel, data::PairwiseSurvivalData) -> Float64

Log-likelihood of exposure data in survival form, summed over who could have
infected whom: use it to estimate the contact interval from rows of
[`PairwiseSurvivalData`](@ref). Each infected person contributes the log of
the total hazard from their possible infectors at their infection time, and
every row subtracts the cumulative hazard over its exposure interval:

    ll = Σ_susceptible log Σ_{event rows} hazard(stop)
         − Σ_rows [cumhazard(stop) − cumhazard(start)]

People never infected are right-censored: they contribute only the probability
of escaping infection. `kernel` is the contact-interval distribution shared by
every row, or a function `r -> Distribution` of the row number for covariates.
The result is differentiable in the kernel's parameters, so it can be maximised
with Optim.jl or added to a Turing model with `@addlogprob!`.
"""
function pairwise_surv_loglik(kernel, data::PairwiseSurvivalData)
    groups = Dict{Int, Vector{Int}}()
    for r in eachindex(data.event)
        data.event[r] && push!(get!(groups, data.sus[r], Int[]), r)
    end

    ll = 0.0
    for (_, g) in groups
        ll += logsumexp([loghazard(_rowkernel(kernel, r), data.stop[r]) for r in g])
    end
    for r in eachindex(data.stop)
        kr = _rowkernel(kernel, r)
        ll -= cumhazard(kr, data.stop[r])
        data.start[r] > 0 && (ll += cumhazard(kr, data.start[r]))
    end
    return ll
end

# ── The infection layer ──────────────────────────────────────────────

"""
    InfectionLayer

The record of who was infected when in an outbreak, together with who could
have infected whom (household members, or neighbours in a contact network).
This is the data [`pairwise_surv_loglik`](@ref) scores to estimate the contact
interval. A subtype holds, for each person `i` (numbered `1:n`):

- `infection_time[i]`: the infection time in days, `NaN` if never infected;
- `infectious_time[i]`: when they became infectious;
- `removal_time[i]`: when they stopped being infectious (recovery, death or
  isolation);
- `is_index[i]`: whether they were infected from outside the households or
  network;

and `obs_end`, the day after which no more infections from outside arrive (only
used when there is a community hazard); spread within the households or network
continues after it. A subtype may also hold `host_times`, a named tuple of
further per-person time vectors such as `onset_time`, which a
[`PairKernel`](@ref) `state` function or a vaccine effect reads in the
likelihood as it reads `individual.state` in simulation. `missing` marks a
person without that time; `NaN` is a recorded value, as for the onset of an
asymptomatic case in a simulation.

Follow-up ends at [`followup_end`](@ref EpiBranch.followup_end): a
`followup_end` field when the subtype has one, and `Inf` otherwise. The
likelihood ignores everything after it. A person infected later counts as
uninfected until then, exposure stops there, and a case still infectious at the
end of follow-up can keep a removal time of `Inf`. Simulated outbreaks run to
the end and need no end of follow-up.

In real data the infection times are usually unobserved; they are known
exactly after a simulation and imputed (data augmentation) in inference. What
is observed, such as onsets and test results, comes from the natural history
and is scored separately by [`progression_loglik`](@ref); the sum of the two is
the full log-likelihood of the augmented data. There is no closed-form
likelihood of the onsets alone, because the unobserved infection times cannot
be integrated out exactly.

`household_infections` (in `EpiHouseholds`) and `network_infections` (in
`EpiNetwork`) read a simulated outbreak into an `InfectionLayer`. Each infected
person's infectious period starts at the process's `from` state and ends at
the earliest of its `until` states and the time the model's interventions take
them out of transmission, such as isolation or quarantine after tracing; these
are the same periods the simulation used. Any other change to transmission
must be built into the kernel passed to the likelihood, except changes to a
person's own susceptibility, which the model's interventions declare through
[`susceptibility_components`](@ref EpiBranch.susceptibility_components).
`loglikelihood(data, model)` for households and networks checks the model's
interventions with [`infection_likelihood_compatible`](@ref); where it
refuses, call `pairwise_surv_loglik` with a kernel that includes the extra
effects. Passing `followup_end` to `household_infections` or
`network_infections` evaluates the outbreak as if observation had stopped then.

To define a new kind of contact structure, a subtype also defines
[`contact_structure`](@ref EpiBranch.contact_structure);
[`compile_contact_pairs`](@ref) and [`pairwise_surv_loglik`](@ref) then work
on it with no further methods. `HouseholdInfections` and `NetworkInfections`
are the worked examples.
"""
abstract type InfectionLayer end

"""
    contact_structure(data::InfectionLayer)

Who could have infected whom in `data`, in a form
[`compile_contact_pairs`](@ref) accepts: a membership vector (people sharing a
label can all infect one another, as in households) or a list of contacts
(`contacts[i]` lists the people `i` can infect, as in a directed contact
network). An [`InfectionLayer`](@ref) subtype defines this.
"""
function contact_structure(data::InfectionLayer)
    throw(
        ArgumentError(
            "$(nameof(typeof(data))) needs a method for " *
                "`EpiBranch.contact_structure` naming who could have infected whom"
        )
    )
end

"""
    followup_end(data::InfectionLayer)

The day observation of `data` stopped. The pairwise likelihood covers
infections and exposure up to it and ignores everything after it. The default
reads a `followup_end` field when the [`InfectionLayer`](@ref) subtype has
one, and is `Inf` otherwise; a subtype that stores it elsewhere defines a
method.
"""
function followup_end(data::InfectionLayer)
    return hasproperty(data, :followup_end) ?
        data.followup_end : Inf
end

"""
    host_times(data::InfectionLayer)

Each person's recorded event times in `data` beyond the start and end of their
infectious period (such as `onset_time`), as a named tuple of vectors, with
`missing` for a person without that time. A [`PairKernel`](@ref) `state`
function or a vaccine effect reads these in the likelihood as it reads
`individual.state` in simulation. The default reads a `host_times` field when
the [`InfectionLayer`](@ref) subtype has one, and is empty otherwise; a
subtype that stores them elsewhere defines a method.
"""
function host_times(data::InfectionLayer)
    return hasproperty(data, :host_times) ? data.host_times : (;)
end

# ── Removals that lapse ──────────────────────────────────────────────
#
# A removal with a release takes its host out of transmission for a stretch and
# hands it back, leaving the host infectious on both sides of it, and a host can
# be removed and handed back more than once. An infectious window holds one
# closing time and cannot reopen, and `infectious_removal_time` therefore
# leaves such removals alone (see `Isolation`): the window runs to the host's
# natural-history close and every stretch comes out of each pair's exposure
# here instead. Cumulative hazards add, so a stretch comes out by evaluating
# the pair's own kernel at its two ends.

"""
    removal_gap_host_times(component) -> Tuple of Symbol

For an intervention that removes a case from transmission for a while and then
releases them (isolation that ends, quarantine that expires): the
`individual.state` keys under which it records those periods.
`household_infections` and `network_infections` add them to the infection
record's `host_times`, and the likelihood removes each recorded period from
the exposure of everyone the case could have infected. The default is `()`,
for an intervention that never releases anyone.

Both built-in removals record through [`record_removal!`](@ref
EpiBranch.record_removal!): [`Isolation`](@ref) uses the reserved
`:_removal_stretches` key for perfect isolation with a duration that can end,
and [`ContactTracing`](@ref) uses the quarantine's own key for any
[`Quarantine`](@ref), whatever its duration. A period with no release ends the
exposure where it starts. An intervention of your own names the key it records
under, whose value is a vector of `(start, release)` pairs. A wrapper that can
lift the removal part-way through a period, such as a [`Scheduled`](@ref) with
an end time, names none. The infectious period then ends at the first removal
only if that removal never releases. A removal with a release is left out of
the record, and in simulation acts contact by contact instead.
"""
removal_gap_host_times(component) = ()

# The keys every removal in `model` records its stretches under.
function _removal_gap_keys(model::ModelSpec)
    keys = Symbol[]
    for component in model.interventions, key in removal_gap_host_times(component)
        key in keys || push!(keys, key)
    end
    return keys
end

# Every stretch each host was removed for, as one column of sorted disjoint
# `(start, release)` pairs, or `nothing` when the layer records none. Several
# removals' keys are already merged into the one column (see
# `_layer_host_times`), a host removed by either being removed.
function _removal_gaps(data)
    times = host_times(data)
    haskey(times, REMOVAL_STRETCHES_KEY) || return nothing
    return times[REMOVAL_STRETCHES_KEY]
end

# A host that never had the key written reads as `missing`, which stands for a
# host no removal reached.
function _host_stretches(gaps, i)
    gaps === nothing && return _NO_STRETCHES
    return coalesce(gaps[i], _NO_STRETCHES)
end

# The pair's cumulative hazard over the exposure `[0, stop]`, in elapsed time
# since the infector became infectious, with every stretch it was removed for
# taken out. Cumulative hazards add, so each stretch comes out by evaluating
# the pair's own kernel at its two ends.
#
# Summed as the stretches that survive, never as the whole exposure less the
# gaps: a kernel of bounded support has an infinite cumulative hazard past its
# support, and one infinity less another gives a `NaN` where the head alone is
# the answer.
function _gapped_cumhazard(H, stretches, oi, stop)
    isempty(stretches) && return H(stop)
    zero_t = zero(stop)
    total = H(zero_t)
    u = zero_t
    for (a, b) in stretches
        lo = clamp(a - oi, zero_t, stop)
        # A removal that never releases ends the exposure where it starts. The
        # built-in removals close the infectious window there through
        # `infectious_removal_time`, so `stop` has already accounted for it and
        # `lo` is `stop`, which adds nothing; one written outside the package
        # that leaves its window open is still fitted on the exposure it
        # offered rather than on the days it blocked.
        hi = isfinite(b) ? clamp(b - oi, zero_t, stop) : stop
        hi > lo || continue
        total += _surviving_cumhazard(H, u, lo)
        u = max(u, hi)
    end
    return total + _surviving_cumhazard(H, u, stop)
end

# One surviving stretch's share of the exposure. Past the time the pair's
# survival reaches zero the kernel has no mass left, so a stretch beginning
# there contributes nothing, where the difference of two infinities would give
# a `NaN`.
function _surviving_cumhazard(H, lo, hi)
    h_lo = H(lo)
    (hi > lo && isfinite(h_lo)) || return zero(h_lo)
    return H(hi) - h_lo
end

# Whether the infector was removed at `t`, so it cannot be what infected this
# susceptible.
function _removed_at(gaps, i, t)
    for (a, b) in _host_stretches(gaps, i)
        a <= t < b && return true
    end
    return false
end

# The per-host fields of an `InfectionLayer` subtype over `n` hosts, in field
# order after the contact structure: the three time vectors, `is_index`,
# `obs_end`, `followup_end` and `host_times`. Every time shares one number type,
# at least `Float64`, which lets a constructor take integers or AD values.
function _infection_layer_fields(
        n, infection_time, infectious_time, removal_time,
        is_index; obs_end, followup_end, host_times = (;)
    )
    host_times isa NamedTuple ||
        throw(ArgumentError("host_times must be a named tuple of per-host vectors"))
    all(
        length(v) == n
            for v in (
                infection_time, infectious_time, removal_time, is_index,
                values(host_times)...,
            )
    ) ||
        throw(
        ArgumentError(
            "the contact structure and the per-host vectors must " *
                "cover the same hosts"
        )
    )
    T = promote_type(
        eltype(infection_time), eltype(infectious_time),
        eltype(removal_time), typeof(obs_end), typeof(followup_end),
        map(
            v -> nonmissingtype(eltype(v)),
            filter(_numeric_host_times, values(host_times))
        )..., Float64
    )
    return (
        Vector{T}(infection_time), Vector{T}(infectious_time),
        Vector{T}(removal_time), Vector{Bool}(is_index), T(obs_end), T(followup_end),
        map(v -> _host_time_column(v, T), host_times),
    )
end

# A per-host column of numbers shares the layer's time type, so that an AD value
# threads through it. One holding anything else, such as the stretches a removal
# recorded, is taken as it stands.
_numeric_host_times(v) = nonmissingtype(eltype(v)) <: Real
function _host_time_column(v, ::Type{T}) where {T}
    _numeric_host_times(v) || return v
    return Vector{Missing <: eltype(v) ? Union{Missing, T} : T}(v)
end

# The named per-host times of a simulated `state`, read from each individual's
# state under the given keys, `missing` where a host has none. A key no
# individual holds gives an all-`missing` column, since a run in which a policy
# never triggered still has to be evaluated.
function _host_time_columns(state::SimulationState, keys)
    names = Tuple(Symbol(key) for key in keys)
    columns = map(names) do key
        [get(ind.state, key, missing) for ind in state.individuals]
    end
    return NamedTuple{names}(columns)
end

# The per-host columns of an infection layer, read out of a `state` simulated
# from `model`, whose process runs one Sellke race with a `from` state and
# `until` states (as `HouseholdProcess` and `NetworkProcess` do). Each window is
# the one that race used, closed by the model's interventions as well, which
# makes the `simulate → loglikelihood` round trip exact.
function _infection_layer_columns(state::SimulationState, model::ModelSpec)
    process = model.process
    from = _resolve_infectious_from(process.from, model.progression)
    window = _shorthand_window(from, process.until)
    n = length(state.individuals)
    infection_time = fill(NaN, n)
    infectious_time = fill(NaN, n)
    removal_time = fill(Inf, n)
    is_index = falses(n)
    for (k, ind) in enumerate(state.individuals)
        get(ind.state, :infected, false) || continue
        infection_time[k] = ind.infection_time
        infectious_time[k] = window_open(ind, window)
        removal_time[k] = window_close(ind, window, model.interventions)
        is_index[k] = get(ind.state, :index, false)
    end
    return (; infection_time, infectious_time, removal_time, is_index)
end

"""
    infection_likelihood_compatible(component) -> Bool

Whether the pairwise likelihood can account for this intervention (or other
model component) from the recorded infectious periods alone: `true` when it
changes transmission only by starting or ending people's infectious periods,
changes a person's susceptibility only through [`susceptibility_components`](@ref
EpiBranch.susceptibility_components), or does not affect infection at all. The
default is `false`, so `loglikelihood(data, model)` refuses a model with an
intervention that has not declared this. Partial blocking (leaky isolation),
changes to infectiousness, and changes to the contact interval need a kernel
that includes them, passed to `pairwise_surv_loglik` directly.

Declaring `true` is a promise by the intervention's author; nothing checks it.
The infection likelihood takes the infection record as given and does not
include the probability of clinical outcomes, of who received interventions,
of people's characteristics, or of observation.
"""
infection_likelihood_compatible(component) = false
infection_likelihood_compatible(::NoAttributes) = true
infection_likelihood_compatible(::ClinicalPresentation) = true
function infection_likelihood_compatible(components::Union{Tuple, AbstractVector})
    return all(infection_likelihood_compatible, components)
end
infection_likelihood_compatible(::Transition) = true
infection_likelihood_compatible(::Union{Reporting, Hospitalisation, Recovery, Death}) = true
infection_likelihood_compatible(iso::Isolation) = iso.post_isolation_transmission == 0
function infection_likelihood_compatible(ct::ContactTracing)
    return infection_likelihood_compatible(ct.action)
end
infection_likelihood_compatible(::Union{Quarantine, FlagOnly}) = true
# A vaccination's susceptibility risk (`efficacy`) reaches the likelihood
# through `susceptibility_components`. A labelled dose belongs to a schedule
# whose doses block exposures as separate competing risks, which one effect per
# host does not combine. Any other hazard effect a vaccination has is not
# represented either: `RingVaccination`'s `onward_efficacy` acts on the parent's
# own transmission and `post_exposure_efficacy` can abort an existing
# infection, neither of which the likelihood's kernel sees.
function infection_likelihood_compatible(v::Union{MassVaccination, GroupVaccination})
    return dose_label(v) === :default
end
function infection_likelihood_compatible(rv::RingVaccination)
    return dose_label(rv) === :default && !_maybe_positive(rv.onward_efficacy) &&
        !_maybe_positive(rv.post_exposure_efficacy)
end
function infection_likelihood_compatible(w::Union{Scheduled, CapacityConstrained})
    return infection_likelihood_compatible(w.intervention)
end

# ── Susceptibility effects ───────────────────────────────────────────

"""
    HazardScaling(start, factor)

A change in one person's risk of infection from a given day, such as protection
from a vaccine: from day `start` on, every hazard of infection they face, from
each possible infector and from the community alike, is multiplied by
`factor`. `factor` is a non-negative number, or a function `dt -> Real` of the
days since `start` for protection that changes over time, such as waning.
Before `start` the hazard is unchanged.

One component of what [`susceptibility_components`](@ref
EpiBranch.susceptibility_components) returns.
"""
struct HazardScaling{S <: Real, F}
    start::S
    factor::F
    function HazardScaling(start::S, factor::F) where {S <: Real, F}
        _check_scaling_factor(factor)
        return new{S, F}(start, factor)
    end
end

function _check_scaling_factor(factor::Union{AbstractFloat, Integer, Rational})
    factor >= 0 || _negative_scaling_factor(factor)
    return nothing
end
# An AD number at zero compares by the sign of its derivative, so `factor < 0`
# would reject a factor of exactly zero that is being differentiated, such as
# `1 - efficacy` at efficacy 1. A threshold just below zero tests the value alone.
function _check_scaling_factor(factor::Real)
    factor < -floatmin(Float64) && _negative_scaling_factor(factor)
    return nothing
end
_check_scaling_factor(factor) = nothing
function _negative_scaling_factor(factor)
    throw(ArgumentError("a hazard scaling factor must be non-negative, got $factor"))
end

_scaling_at(factor::Real, dt) = factor
# A function factor is checked at each value it returns, since no single value
# stands for it when it is built.
function _scaling_at(factor, dt)
    value = factor(dt)
    _check_scaling_factor(value)
    return value
end

"""
    susceptibility_components(effect, host) -> components or nothing

How an intervention such as vaccination changes one person's risk of infection
in the pairwise likelihood (the `susceptibility` keyword of
[`pairwise_surv_loglik`](@ref)). `host` is a [`LayerHost`](@ref): the person's
`id`, `infection_time`, and the recorded event times under `host.state`.

It returns `nothing`, the default, when `effect` leaves the person's risk
unchanged. Otherwise it returns `weight => modifier` pairs, each modifier a
[`HazardScaling`](@ref EpiBranch.HazardScaling) or `nothing` (no change), with
weights that are probabilities summing to one. The person's likelihood
contribution is the weighted mixture of the likelihood of their escapes and
infection under each modifier. One component describes an effect every
exposure shares; several describe an unobserved state drawn once per person
that then governs all their exposures together.

For a vaccination's `VaccineEffect` under `LeakyMode` this is one component,
`1 => HazardScaling(τ, 1 - efficacy)` from the person's immunity time `τ`
(with `waning`, the factor is `dt -> 1 - efficacy * waning(dt)`). Under
`AllOrNothingMode` it is two: `efficacy => HazardScaling(τ, 0.0)` for someone
the vaccine fully protects and `1 - efficacy => nothing` for someone it does
not. The immunity time is read from `:immunity_time` (`:immunity_time_<label>`
for a labelled dose), and a person without one is unchanged. Every
`AbstractVaccination` returns its `VaccineEffect`'s answer, and an
`InterventionWrapper` that of the intervention it wraps.

For a collection, such as a model's interventions, the one non-`nothing`
answer among them is used, and an `ArgumentError` is raised if more than one
changes the same person. To support a new kind of effect, define a method for
it and [`susceptibility_host_times`](@ref EpiBranch.susceptibility_host_times)
for the event times it reads.
"""
susceptibility_components(effect, host) = nothing

function susceptibility_components(components::Union{Tuple, AbstractVector}, host)
    found = nothing
    for component in components
        mixture = susceptibility_components(component, host)
        mixture === nothing && continue
        found === nothing || throw(
            ArgumentError(
                "more than one composed component modifies the susceptibility of " *
                    "host $(host.id); define `susceptibility_components` for a single " *
                    "effect combining them"
            )
        )
        found = mixture
    end
    return found
end

function susceptibility_components(effect::VaccineEffect, host)
    efficacy = _fitted_efficacy(effect.efficacy)
    τ = get(host.state, _immunity_time_key(effect.dose_label), Inf)
    isfinite(τ) || return nothing
    return _dose_components(effect.mode, efficacy, effect.waning, τ)
end

function susceptibility_components(v::AbstractVaccination, host)
    return susceptibility_components(vaccine_effect(v), host)
end

function susceptibility_components(w::InterventionWrapper, host)
    return susceptibility_components(w.intervention, host)
end

# The likelihood evaluates one population-level efficacy. A `Distribution` or
# function draws a value per vaccinated individual in simulation, and the
# likelihood has no per-host draw to read it from.
function _fitted_efficacy(efficacy::Union{AbstractFloat, Integer, Rational})
    0 <= efficacy <= 1 || _efficacy_out_of_range(efficacy)
    return efficacy
end
# An AD number at a bound compares by the sign of its derivative, as for a
# scaling factor, so thresholds just outside [0, 1] test the value alone.
function _fitted_efficacy(efficacy::Real)
    (efficacy < -floatmin(Float64) || efficacy > 1 + eps(Float64)) &&
        _efficacy_out_of_range(efficacy)
    return efficacy
end
function _efficacy_out_of_range(efficacy)
    throw(ArgumentError("a vaccine's efficacy must lie in [0, 1], got $efficacy"))
end
function _fitted_efficacy(efficacy)
    throw(
        ArgumentError(
            "a vaccine's efficacy must be a Real for the likelihood; a Distribution " *
                "or Function describes a simulation draw, not a fitted value"
        )
    )
end

# Each mode says how a dose decomposes into mixture components. There is no
# generic answer: the two built-ins differ in kind, and a mode of a user's own
# (a partial-responder mode, say) differs again, so one that reaches the
# likelihood without a method is a missing contract rather than a bad value.
function _dose_components(mode::AbstractEffectMode, efficacy, waning, τ)
    throw(
        ArgumentError(
            "a dose under $(nameof(typeof(mode))) has no decomposition into " *
                "susceptibility mixture components, so a likelihood cannot " *
                "evaluate it. Give the effect holding this mode a " *
                "`susceptibility_components` method"
        )
    )
end

function _dose_components(::LeakyMode, efficacy, ::Nothing, τ)
    return (one(efficacy) => HazardScaling(τ, 1 - efficacy),)
end
function _dose_components(::LeakyMode, efficacy, waning, τ)
    return (one(efficacy) => HazardScaling(τ, dt -> 1 - efficacy * waning(dt)),)
end
# Responder status is drawn once per vaccinated individual and governs every
# exposure it faces, so the escape from all infectors sits inside the mixture.
function _dose_components(::AllOrNothingMode, efficacy, waning, τ)
    return (efficacy => HazardScaling(τ, zero(efficacy)), (1 - efficacy) => nothing)
end

"""
    susceptibility_host_times(component) -> Tuple of Symbols

The event times (keys of `individual.state`) that `component`'s
[`susceptibility_components`](@ref EpiBranch.susceptibility_components) reads,
such as a vaccinee's immunity time. `household_infections` and
`network_infections` record them in the infection record's `host_times`
whenever the model includes `component`. The default is `()`. An
`AbstractVaccination` reads its dose's immunity time, and an
`InterventionWrapper` the times of the intervention it wraps.
"""
susceptibility_host_times(component) = ()
function susceptibility_host_times(v::AbstractVaccination)
    return (_immunity_time_key(dose_label(v)),)
end
function susceptibility_host_times(w::InterventionWrapper)
    return susceptibility_host_times(w.intervention)
end

# The host times an infection layer read out of a `state` simulated from
# `model` records: those the caller names, and those the model's composed
# components read.
function _layer_host_time_keys(model::ModelSpec, host_times)
    keys = Symbol[Symbol(key) for key in host_times]
    for component in model.interventions, key in susceptibility_host_times(component)
        key in keys || push!(keys, key)
    end
    for key in _removal_gap_keys(model)
        key in keys || push!(keys, key)
    end
    return keys
end

# The host-time columns an infection layer reads out of a simulated `state`,
# with every removal's stretches merged into the one column the likelihood
# takes out of each exposure. A host removed by either of two removals is
# removed, so the merge is their union.
function _layer_host_times(state::SimulationState, model::ModelSpec, host_times)
    columns = _host_time_columns(state, _layer_host_time_keys(model, host_times))
    gap_keys = _removal_gap_keys(model)
    isempty(gap_keys) && return columns
    merged = map(eachindex(state.individuals)) do i
        stretches = mapreduce(
            key -> coalesce(columns[key][i], _NO_STRETCHES), vcat, gap_keys;
            init = _NO_STRETCHES
        )
        return _merge_stretches(sort(stretches; by = first))
    end
    return merge(columns, NamedTuple{(REMOVAL_STRETCHES_KEY,)}((merged,)))
end

function _validate_infection_likelihood(model::ModelSpec)
    for component in (model.attributes, model.progression, model.interventions)
        infection_likelihood_compatible(component) && continue
        throw(
            ArgumentError(
                "infection-layer likelihood cannot represent all effects of " *
                    "$(typeof(component)). Use pairwise_surv_loglik with an explicit " *
                    "effective kernel (and external_hazard where needed), or define " *
                    "infection_likelihood_compatible for an external component whose " *
                    "effects are fully represented by the layer's infectious windows."
            )
        )
    end
    return nothing
end

# ── Compiled pair layout ─────────────────────────────────────────────
#
# In inference the contact structure, the index cases and the set of
# ever-infected hosts are fixed across gradient evaluations; only the augmented
# infection and infectious times move. `ContactPairsLayout` holds everything
# that does not depend on those times: row → (susceptible, infector, the edge it
# travels along) and a susceptible-grouped index for the log-sum-exp. An
# evaluation is then a single pass over rows with no `Dict` and a streaming
# log-sum-exp, which keeps a reverse-mode AD tape short.

"""
    ContactPairsLayout

The list of who could have infected whom that the pairwise likelihood is
evaluated over, built once with [`compile_contact_pairs`](@ref) and reused
while infection times change during inference. Each row is one ordered
(susceptible, possible infector) pair, plus, when there is a community hazard,
one row per person for infection from outside. Rows whose exposure periods do
not overlap are kept and skipped when evaluating, so one layout works for every
set of infection times with the same contact structure and the same people
infected.

`component` gives each person's group in the contact structure (their
household, or their connected part of a contact network), numbered
`1:ncomponents`. [`pairwise_surv_loglik_by_component`](@ref) uses it to split
the likelihood by group, for a sampler that updates infection times one group
at a time.
"""
struct ContactPairsLayout
    sus::Vector{Int}                       # row r → susceptible host id
    infector::Vector{Int}                  # row r → infector host id (0 = external)
    contact_index::Vector{Int}             # row r → position of sus in the infector's contact list (0 = none)
    is_ext::Vector{Bool}
    sus_unique::Vector{Int}                # susceptibles that have ≥1 row
    sus_row_ranges::Vector{UnitRange{Int}} # row indices in `sus_row_order`
    sus_row_order::Vector{Int}             # row indices, susceptible-grouped
    no_rows::Vector{Int}                   # hosts not conditioned on that have no row
    external::Bool
    nhosts::Int                            # population the layout was compiled for
    component::Vector{Int}                 # host → connected-component id (1:ncomponents)
    ncomponents::Int                       # number of connected components of the contact structure
end

Base.length(L::ContactPairsLayout) = length(L.sus)

function _check_host_masks(n, is_index, infected)
    (length(is_index) == n && length(infected) == n) || throw(
        ArgumentError(
            "the contact structure, is_index and infected must cover the same hosts"
        )
    )
    return nothing
end

# The 1:ncomponents grouping of a contact structure, so that
# `pairwise_surv_loglik_by_component` can attribute rows to the household or
# network component they fall in. A membership vector's labels are already the
# components; an adjacency list needs its connectivity found, undirected,
# since a susceptible and its possible infectors must be updated together
# whichever way the edge between them points.
function _structure_components(membership::AbstractVector{<:Integer})
    ids = Dict{Int, Int}()
    component = Vector{Int}(undef, length(membership))
    for i in eachindex(membership)
        component[i] = get!(ids, membership[i], length(ids) + 1)
    end
    return component, length(ids)
end

function _structure_components(contacts::AbstractVector{<:AbstractVector{<:Integer}})
    n = length(contacts)
    parent = collect(1:n)
    function _root(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    for i in 1:n, j in contacts[i]
        1 <= j <= n || continue
        ri, rj = _root(i), _root(j)
        ri == rj || (parent[ri] = rj)
    end
    ids = Dict{Int, Int}()
    component = Vector{Int}(undef, n)
    for i in 1:n
        component[i] = get!(ids, _root(i), length(ids) + 1)
    end
    return component, length(ids)
end

# Group rows by susceptible for the per-susceptible log-sum-exp, and wrap up.
# Hosts that are explained (all of them with a community hazard, all but the
# index cases without one) and have no possible infector are listed apart, since
# an infection of one has zero density.
function _contact_pairs_layout(
        sus, infector, contact_index, is_ext, external,
        is_index, n, component, ncomponents
    )
    n_rows = length(sus)
    sus_row_order = sortperm(sus)
    sus_unique = Int[]
    sus_row_ranges = UnitRange{Int}[]
    if n_rows > 0
        s_prev = sus[sus_row_order[1]]
        push!(sus_unique, s_prev)
        range_lo = 1
        for k in 2:n_rows
            s_k = sus[sus_row_order[k]]
            if s_k != s_prev
                push!(sus_row_ranges, range_lo:(k - 1))
                push!(sus_unique, s_k)
                range_lo = k
                s_prev = s_k
            end
        end
        push!(sus_row_ranges, range_lo:n_rows)
    end
    has_rows = falses(n)
    has_rows[sus_unique] .= true
    no_rows = [j for j in 1:n if !has_rows[j] && (external || !is_index[j])]
    return ContactPairsLayout(
        sus, infector, contact_index, is_ext, sus_unique,
        sus_row_ranges, sus_row_order, no_rows, external, n,
        component, ncomponents
    )
end

"""
    compile_contact_pairs(membership::AbstractVector{<:Integer}, is_index, infected; external = false)
    compile_contact_pairs(contacts::AbstractVector{<:AbstractVector{<:Integer}}, is_index, infected; external = false)
    compile_contact_pairs(data::InfectionLayer; external = false)

List once who could have infected whom, as a [`ContactPairsLayout`](@ref) to
reuse across likelihood evaluations during inference.

The contact structure is either a membership vector, where people sharing a
label can all infect one another (such as households), or a list of contacts,
where `contacts[i]` lists the people `i` can infect (a directed network; list
each contact both ways for an undirected one). A person's possible infectors
are their household members or the people listing them. A contact listed twice
counts as two separate contacts.

`infected` is true for each person infected in every set of infection times
the layout will be used with (their infection time may still be imputed). Only
infected people can infect others. `is_index` marks people infected from
outside: without a community hazard the likelihood conditions on them and they
appear only as infectors. With `external = true` every infection is explained,
and each person gets an extra row for infection from outside. The
single-argument form reads the contact structure from `data` with
[`contact_structure`](@ref EpiBranch.contact_structure) and takes as infected
everyone with a non-`NaN` `infection_time`.
"""
function compile_contact_pairs(
        membership::AbstractVector{<:Integer},
        is_index::AbstractVector{Bool}, infected::AbstractVector{Bool};
        external::Bool = false
    )
    n = length(membership)
    _check_host_masks(n, is_index, infected)
    isempty(membership) && return _contact_pairs_layout(
        Int[], Int[], Int[], Bool[], external, is_index, 0, Int[], 0
    )

    # Bucket hosts by label into a `Vector{Vector{Int}}` indexed by label offset,
    # which avoids hashing; offsetting by `lo` allows any integer labels.
    lo, hi = extrema(membership)
    n_buckets = hi - lo + 1
    buckets = [Int[] for _ in 1:n_buckets]
    for i in 1:n
        push!(buckets[membership[i] - lo + 1], i)
    end

    sus = Int[]
    infector = Int[]
    is_ext = Bool[]
    for h in 1:n_buckets
        mem = buckets[h]
        isempty(mem) && continue
        for j in mem
            (!external && is_index[j]) && continue
            if external
                push!(sus, j)
                push!(infector, 0)
                push!(is_ext, true)
            end
            for i in mem
                infected[i] || continue
                i == j && continue
                push!(sus, j)
                push!(infector, i)
                push!(is_ext, false)
            end
        end
    end
    # A group has no per-edge list for a kernel to index into.
    contact_index = zeros(Int, length(sus))
    component, ncomponents = _structure_components(membership)
    return _contact_pairs_layout(
        sus, infector, contact_index, is_ext, external,
        is_index, n, component, ncomponents
    )
end

function compile_contact_pairs(
        contacts::AbstractVector{<:AbstractVector{<:Integer}},
        is_index::AbstractVector{Bool}, infected::AbstractVector{Bool};
        external::Bool = false
    )
    n = length(contacts)
    _check_host_masks(n, is_index, infected)

    # Invert the out-lists of infected hosts into in-lists (compressed rows), so
    # each susceptible's possible infectors come out together, in infector order.
    indeg = zeros(Int, n + 1)
    for i in 1:n
        infected[i] || continue
        for j in contacts[i]
            1 <= j <= n || throw(
                ArgumentError(
                    "host $i lists contact $j, outside the $n hosts in the structure"
                )
            )
            j == i || (indeg[j + 1] += 1)
        end
    end
    ptr = cumsum(indeg) .+ 1                      # in-edges of j: ptr[j]:(ptr[j+1]-1)
    src = Vector{Int}(undef, ptr[end] - 1)
    pos = Vector{Int}(undef, ptr[end] - 1)
    fill_at = ptr[1:n]
    for i in 1:n
        infected[i] || continue
        for (k, j) in enumerate(contacts[i])
            j == i && continue
            src[fill_at[j]] = i
            pos[fill_at[j]] = k
            fill_at[j] += 1
        end
    end

    n_rows = 0
    for j in 1:n
        (!external && is_index[j]) && continue
        n_rows += (ptr[j + 1] - ptr[j]) + external
    end
    sus = Vector{Int}(undef, n_rows)
    infector = Vector{Int}(undef, n_rows)
    contact_index = Vector{Int}(undef, n_rows)
    is_ext = Vector{Bool}(undef, n_rows)
    r = 0
    for j in 1:n
        (!external && is_index[j]) && continue
        if external
            r += 1
            sus[r] = j
            infector[r] = 0
            contact_index[r] = 0
            is_ext[r] = true
        end
        for e in ptr[j]:(ptr[j + 1] - 1)
            r += 1
            sus[r] = j
            infector[r] = src[e]
            contact_index[r] = pos[e]
            is_ext[r] = false
        end
    end
    component, ncomponents = _structure_components(contacts)
    return _contact_pairs_layout(
        sus, infector, contact_index, is_ext, external,
        is_index, n, component, ncomponents
    )
end

function compile_contact_pairs(data::InfectionLayer; external::Bool = false)
    infected = .!isnan.(data.infection_time)
    return compile_contact_pairs(
        contact_structure(data), data.is_index, infected;
        external
    )
end

# ── The community hazard ─────────────────────────────────────────────
#
# A model with a contact structure can also introduce cases from outside it: a
# non-negative rate (a constant hazard) or a continuous distribution on the
# non-negative reals (a calendar-time hazard). Its simulators and this likelihood
# share these helpers and agree on when the term applies and what it is.

# A distribution with negative support is rejected, because introductions cannot
# happen before time 0.
_valid_external(α::Real) = α >= 0
_valid_external(d::ContinuousUnivariateDistribution) = minimum(d) >= 0
_valid_external(_) = false
_normalise_external(α::Real) = Float64(α)
_normalise_external(d::ContinuousUnivariateDistribution) = d

# The community hazard is off at a zero rate; a distribution is always on.
_ext_active(α::Real) = α > 0
_ext_active(::ContinuousUnivariateDistribution) = true

# The community hazard as a calendar-time survival distribution: a constant rate
# α is `Exponential(1/α)` (hazard α, cumulative α·t); a distribution is itself.
_ext_survival(α::Real) = Exponential(1 / α)
_ext_survival(d::ContinuousUnivariateDistribution) = d

# A community introduction time drawn under the hazard.
_ext_draw(rng::AbstractRNG, source) = rand(rng, _ext_survival(source))
function _ext_draw(rng::AbstractRNG, source, susceptibility)
    return _traits_scaled_draw(rng, _ext_survival(source), susceptibility)
end

# ── Evaluation ───────────────────────────────────────────────────────

# Row r's contact-interval distribution: a shared distribution, a per-edge
# vector parallel to the adjacency the layout was compiled from, or a callable
# `(infector, susceptible) -> Distribution` for covariates.
_pair_kernel(k::ContinuousUnivariateDistribution, layout::ContactPairsLayout, r, data) = k
function _pair_kernel(k::AbstractVector{<:AbstractVector}, layout::ContactPairsLayout, r, data)
    c = layout.contact_index[r]
    c > 0 || throw(
        ArgumentError(
            "a per-edge kernel needs a layout compiled from an adjacency list"
        )
    )
    return k[layout.infector[r]][c]
end
function _pair_kernel(k, layout::ContactPairsLayout, r, data)
    i = layout.infector[r]
    return pair_kernel(k, i, layout.sus[r], data.infection_time[i], data.infectious_time[i])
end

# A live PairKernel reads each host through its projection, applied here to the
# host as the infection layer records it.
function _pair_kernel(k::PairKernel, layout::ContactPairsLayout, r, data)
    i = layout.infector[r]
    j = layout.sus[r]
    result = k.callback(
        PairContext(i, j, data.infection_time[i]),
        k.state(_layer_host(data, i)), k.state(_layer_host(data, j))
    )
    return _finish_kernel(k, result, data.infectious_time[i])
end
function _pair_kernel(k::PairKernel{F, Nothing}, layout::ContactPairsLayout, r, data) where {F}
    i = layout.infector[r]
    return pair_kernel(k, i, layout.sus[r], data.infection_time[i], data.infectious_time[i])
end
function _pair_kernel(
        k::PairKernel{F, <:AbstractVector}, layout::ContactPairsLayout,
        r, data
    ) where {F}
    i = layout.infector[r]
    return pair_kernel(k, i, layout.sus[r], data.infection_time[i], data.infectious_time[i])
end

# Streaming logsumexp, so the per-susceptible reduction allocates no
# intermediate vector for reverse-mode AD to track. A -Inf term (a zero hazard)
# adds nothing to the sum and is skipped. An accumulator that saw only zero
# hazards then gives -Inf without taking -Inf - (-Inf). The test reads the value
# alone, since an AD dual at -Inf can hold NaN partials and so compare unequal.
mutable struct _LogSumExpAcc{T}
    m::T
    s::T
    nseen::Int
end
_LogSumExpAcc{T}() where {T} = _LogSumExpAcc{T}(T(-Inf), zero(T), 0)
function _push!(acc::_LogSumExpAcc{T}, x) where {T}
    _is_minus_inf(x) && return acc
    if acc.nseen == 0
        acc.m = T(x)
        acc.s = one(T)
    elseif x > acc.m
        acc.s = acc.s * exp(acc.m - x) + one(T)
        acc.m = T(x)
    else
        acc.s += exp(x - acc.m)
    end
    acc.nseen += 1
    return acc
end
_is_minus_inf(x) = isinf(x) && x < 0
_value(acc::_LogSumExpAcc{T}) where {T} = acc.nseen == 0 ? T(-Inf) : acc.m + log(acc.s)

# The parameter float type the kernel adds to the accumulator. In inference the
# fitted parameters are AD duals inside the kernel. The data's float type alone
# cannot hold them, and the streaming accumulator is typed to include them.
# A distribution gives its parameter type through `partype`; a per-edge or
# covariate kernel is probed on the first internal pair. With no internal pair
# the type falls back to `T`.
function _kernel_partype(
        kernel::ContinuousUnivariateDistribution, layout, data, ::Type{T}
    ) where {T}
    return Distributions.partype(kernel)
end
function _kernel_partype(kernel, layout, data, ::Type{T}) where {T}
    for r in eachindex(layout.is_ext)
        layout.is_ext[r] && continue
        return Distributions.partype(_pair_kernel(kernel, layout, r, data))
    end
    return T
end

"""
    pairwise_surv_loglik(kernel, data::InfectionLayer, layout::ContactPairsLayout;
                         external_hazard = 0.0, susceptibility = nothing) -> Real
    pairwise_surv_loglik(kernel, data::InfectionLayer; external_hazard = 0.0,
                         susceptibility = nothing) -> Real

Log-likelihood of an outbreak in households or on a contact network, given who
was infected when and who could have infected whom: use it to estimate the
contact interval (and so the transmission rate) from such data. The data are
an [`InfectionLayer`](@ref), for example from `household_infections` or
`network_infections`. The likelihood sums over who infected whom, so the
transmission tree need not be known.

Each person accumulates hazard from every possible infector while that
infector is infectious and they are still uninfected, and each infected person
adds the log of the total hazard at their infection time. If an infected person
(other than one the likelihood conditions on) faces zero total hazard at their
infection time, because no possible infector can infect them then and there is
no infection from outside, the infection times are impossible under the model
and the result is `-Inf`.

`kernel` is the contact interval, in days from the start of the infector's
infectious period. It can be one distribution shared by every pair, a function
`(infector, susceptible) -> Distribution` of the two people's numbers for
covariates, a [`PairKernel`](@ref) that also uses the infector's infection time
(and, with a `state`, each person's record), or a vector of vectors parallel to
a contact list (`kernel[i][k]` for person `i`'s `k`-th listed contact).

`external_hazard` is infection from outside the households or network: a
constant rate per person per day, or a distribution of the time of infection
from outside. Such infections happen only in the first `data.obs_end` days.
With an external hazard, index cases are explained like any other case;
without one, the likelihood conditions on them. Each person is exposed to the
external hazard until the earlier of their infection and `data.obs_end`, so a
person infected after `obs_end` must have been infected by a possible infector.
Spread within households or the network continues after `obs_end`: a person
never infected is exposed over each possible infector's whole infectious
period.

`susceptibility` is an effect on people's own risk of infection, such as a
candidate [`VaccineEffect`](@ref) or a model's interventions; `nothing` (the
default) leaves every hazard as the kernel gives it. Through
[`susceptibility_components`](@ref EpiBranch.susceptibility_components) the
effect says how it scales every hazard a person faces, from possible infectors
and from outside alike. A `VaccineEffect` reads each person's immunity time
from the record's `host_times`, which [`household_infections`](@ref
EpiBranch.household_infections) and [`network_infections`](@ref
EpiBranch.network_infections) record for a model with vaccination; its
`efficacy` must be a number. Under [`LeakyMode`](@ref), a vaccinated person's
hazards from their immunity time on are multiplied by `1 - efficacy`, or by
`1 - efficacy * waning(dt)` with `waning`. Under [`AllOrNothingMode`](@ref),
their contribution is the mixture `efficacy * Lᵖ + (1 - efficacy) * Lᵘ`: `Lᵖ`
the likelihood of their escapes and infection if fully protected from their
immunity time on (zero if they were infected after it), and `Lᵘ` if
unprotected. Whether the vaccine protects someone is decided once per person
and holds for all their exposures, so the escape from all their possible
infectors sits inside the mixture.

Everything is cut at [`followup_end(data)`](@ref EpiBranch.followup_end): a
person infected after it counts as uninfected until then, and no exposure
counts after it. This gives the same value as first cutting the data there:
later infections unobserved, and removal times and `obs_end` capped at it.

In inference, build the layout once with [`compile_contact_pairs`](@ref) and
reuse it while the imputed infection times change; its `external` setting must
agree with `external_hazard`. The two-argument form builds a layout on each
call.

# Example

Estimate the mean contact interval from a simulated household outbreak:

```julia
using EpiBranch, EpiHouseholds, Distributions, StableRNGs
truth = ModelSpec(HouseholdProcess(fill(4, 500), Exponential(4.0));
    progression = [Transition(:recovered; from = :infection, delay = 6.0, terminal = true)])
data = household_infections(simulate(truth; rng = StableRNG(3)), truth)
ll(scale) = pairwise_surv_loglik(Exponential(scale), data)
grid = 2.0:0.5:6.0
grid[argmax(ll.(grid))]   # close to the true mean of 4 days
```

!!! note "Gradients"
    The likelihood is differentiable in the kernel's parameters. A `Gamma`
    kernel or community hazard needs a reverse-mode automatic differentiation
    backend such as Mooncake, because ForwardDiff cannot differentiate its
    cumulative hazard. `Weibull` and `Exponential` work with either.

!!! warning "A vanishing community hazard is not the no-community case"
    The two condition on different things, and the log-likelihood jumps
    between them at `α = 0`. With `external_hazard = α > 0` an index case
    infected at time `t` contributes `log(α) − αt`, which falls to `-Inf` as
    `α → 0`, because a model that allows infection from outside has to explain
    the index cases it saw. At exactly `external_hazard = 0` index cases are
    conditioned on and contribute nothing, leaving a finite value. A likelihood
    ratio between "some community transmission" and "none" therefore cannot be
    read off by letting `α` approach zero: evaluate the two models separately.

    The jump is at that one point. Near it, the log-likelihood is
    `k log α − αT` up to terms free of `α`, where `k` counts the cases only the
    community can explain and `T` is the total time people are exposed to it.
    In `log α` this is a straight line of slope `k`.
"""
# Validation and type promotion shared by the layout-based forms of
# `pairwise_surv_loglik`: the community-hazard survival distribution, the time
# to truncate at, and the number type the reduction runs in, promoted against
# the kernel's parameter type so AD values in a fitted kernel survive it.
function _pairwise_setup(kernel, data, layout::ContactPairsLayout, external_hazard)
    external = _ext_active(external_hazard)
    external == layout.external ||
        throw(ArgumentError("layout.external = $(layout.external) but external_hazard = $external_hazard"))
    # The @inbounds passes index the time vectors by host id up to the population
    # the layout was compiled for; guard against a `data` with fewer individuals.
    min(
        length(data.infection_time), length(data.infectious_time),
        length(data.removal_time)
    ) >= layout.nhosts ||
        throw(
        DimensionMismatch(
            "data covers fewer individuals than the layout " *
                "was compiled for ($(layout.nhosts))"
        )
    )
    extdist = external ? _ext_survival(external_hazard) : kernel

    tfollow = followup_end(data)
    (!isnan(tfollow) && tfollow >= 0) || throw(
        ArgumentError(
            "followup_end must be a non-negative number (Inf allowed), got $tfollow"
        )
    )
    Tdata = promote_type(
        eltype(data.infection_time),
        eltype(data.infectious_time),
        eltype(data.removal_time),
        typeof(tfollow),
        Float64
    )
    Text = external ? Distributions.partype(extdist) : Union{}
    T = promote_type(Tdata, _kernel_partype(kernel, layout, data, Tdata), Text)
    return extdist, convert(Tdata, tfollow), T
end

function pairwise_surv_loglik(
        kernel, data::InfectionLayer, layout::ContactPairsLayout;
        external_hazard = 0.0, susceptibility = nothing
    )
    return pairwise_reduce(
        _TotalLogLik(), kernel, data, layout; external_hazard, susceptibility
    )
end

# ── Susceptible-level mixtures ───────────────────────────────────────
#
# A susceptible a `susceptibility` effect modifies is evaluated on its own: its
# rows are left out of the two flat passes, and its contribution is the
# log-sum-exp over its mixture components of the log weight plus the escape
# and event terms under that component's modifier.

# The mixture of each susceptible the effect modifies, by host id, or `nothing`
# when it modifies none, which evaluates the layer exactly as without an effect.
_host_mixtures(::Nothing, data, layout) = nothing
function _host_mixtures(effect, data, layout)
    mixtures = Dict{Int, Any}()
    for j in layout.sus_unique
        mixture = susceptibility_components(effect, _layer_host(data, j))
        mixture === nothing || (mixtures[j] = mixture)
    end
    return isempty(mixtures) ? nothing : mixtures
end

_is_mixed(::Nothing, j) = false
_is_mixed(mixtures::AbstractDict, j) = haskey(mixtures, j)

# The number type of the mixtures' weights and modifiers, so that an effect's
# fitted parameters (AD values) survive the reduction.
_mixtures_partype(::Nothing) = Union{}
function _mixtures_partype(mixtures::AbstractDict)
    return foldl(values(mixtures); init = Union{}) do S, mixture
        foldl(mixture; init = S) do S2, (weight, modifier)
            promote_type(S2, typeof(weight), _modifier_partype(modifier))
        end
    end
end
_modifier_partype(::Nothing) = Union{}
function _modifier_partype(m::HazardScaling)
    return promote_type(typeof(m.start), typeof(_scaling_at(m.factor, zero(m.start))))
end

# Cumulative hazard of `kernel` over row-relative time `[0, stop]` (calendar time
# `origin` plus that), under a modifier. A constant factor splits the integral
# exactly at `start`, since `cumhazard` is itself an integral of the hazard; a
# factor that varies is integrated numerically against the kernel's hazard.
_scaled_cumhazard(::Nothing, kernel, origin, stop) = cumhazard(kernel, stop)
function _scaled_cumhazard(m::HazardScaling{<:Real, <:Real}, kernel, origin, stop)
    boundary = clamp(m.start - origin, zero(stop), stop)
    before = cumhazard(kernel, boundary)
    tail = cumhazard(kernel, stop) - before
    # An infinite tail under a zero factor contributes nothing, where the
    # product would give NaN. A finite tail stays in the product so that the
    # derivative with respect to the factor survives a factor of exactly zero.
    iszero(m.factor) && !isfinite(tail) && return before
    return before + m.factor * tail
end
function _scaled_cumhazard(m::HazardScaling, kernel, origin, stop)
    boundary = clamp(m.start - origin, zero(stop), stop)
    before = cumhazard(kernel, boundary)
    # A bounded profile's survival reaches zero at the top of its support, past
    # which its hazard is undefined, so the cumulative hazard is infinite there,
    # as the kernel's own `cumhazard` has it.
    # A window that closes before the effect starts has nothing to integrate,
    # and evaluating the factor there would read it before its start.
    boundary < stop || return before
    stop < maximum(kernel) || return oftype(float(before), Inf)
    after, _ = quadgk(
        s -> hazard(kernel, s) * _scaling_at(m.factor, origin + s - m.start), boundary, stop
    )
    return before + after
end

# The log-hazard `lh` at calendar time `t` under a modifier.
_scaled_loghazard(::Nothing, lh, t) = lh
function _scaled_loghazard(m::HazardScaling, lh, t)
    t < m.start && return lh
    return lh + log(_scaling_at(m.factor, t - m.start))
end

# Log-likelihood of susceptible `tj`'s rows (the range `rg` of
# `layout.sus_row_order`) with every hazard it faces under `modifier`: the
# escape over each row's at-risk window, and the log of the summed hazard at its
# infection if that falls within follow-up. The windows are those of the flat
# passes.
function _component_loglik(
        modifier, kernel, extdist, data, layout, rg, tj, tfollow, ::Type{T},
        gaps = nothing
    ) where {T}
    infector = layout.infector
    is_ext = layout.is_ext
    tend = (isnan(tj) || tj > tfollow) ? tfollow : convert(typeof(tfollow), tj)
    ll = zero(T)
    @inbounds for k in rg
        r = layout.sus_row_order[k]
        if is_ext[r]
            stop = min(tend, data.obs_end)
            stop > 0 || continue
            ll -= _scaled_cumhazard(modifier, extdist, zero(stop), stop)
        else
            i = infector[r]
            oi = data.infectious_time[i]
            isfinite(oi) || continue
            oi < tend || continue
            stop = min(data.removal_time[i], tend) - oi
            stop > 0 || continue
            pk = _pair_kernel(kernel, layout, r, data)
            ll -= _gapped_cumhazard(
                t -> _scaled_cumhazard(modifier, pk, oi, t),
                _host_stretches(gaps, i), oi, stop
            )
        end
    end
    (isnan(tj) || tj > tfollow) && return ll
    acc = _LogSumExpAcc{T}()
    @inbounds for k in rg
        r = layout.sus_row_order[k]
        if is_ext[r]
            (tj >= 0 && tj <= data.obs_end) || continue
            _push!(acc, _scaled_loghazard(modifier, loghazard(extdist, tj), tj))
        else
            i = infector[r]
            oi = data.infectious_time[i]
            isfinite(oi) || continue
            if oi < tj && tj <= data.removal_time[i] && !_removed_at(gaps, i, tj)
                lh = loghazard(_pair_kernel(kernel, layout, r, data), tj - oi)
                _push!(acc, _scaled_loghazard(modifier, lh, tj))
            end
        end
    end
    v = _value(acc)
    _is_minus_inf(v) && return T(-Inf)
    return ll + v
end

# One modified susceptible's contribution: the log of its mixture.
function _mixture_loglik(
        mixture, kernel, extdist, data, layout, g, tfollow, ::Type{T}, gaps = nothing
    ) where {T}
    j = layout.sus_unique[g]
    rg = layout.sus_row_ranges[g]
    tj = data.infection_time[j]
    # The weights enter linearly, never through `log(weight)`: a weight of
    # exactly zero would otherwise drop its component, and with it the
    # derivative of the mixture with respect to that weight.
    terms = map(mixture) do (weight, modifier)
        weight => _component_loglik(
            modifier, kernel, extdist, data, layout, rg, tj, tfollow, T, gaps
        )
    end
    m = T(-Inf)
    for (_, l) in terms
        _is_minus_inf(l) && continue
        m = _is_minus_inf(m) ? T(l) : max(m, T(l))
    end
    _is_minus_inf(m) && return T(-Inf)
    total = zero(T)
    for (weight, l) in terms
        _is_minus_inf(l) || (total += weight * exp(l - m))
    end
    return m + log(total)
end

function pairwise_surv_loglik(
        kernel, data::InfectionLayer; external_hazard = 0.0,
        susceptibility = nothing
    )
    layout = compile_contact_pairs(data; external = _ext_active(external_hazard))
    return pairwise_surv_loglik(kernel, data, layout; external_hazard, susceptibility)
end

# ── Row-grouped reductions ───────────────────────────────────────────
#
# `pairwise_surv_loglik` and `pairwise_surv_loglik_by_component` score the
# same two passes over `layout`'s rows and differ only in which rows add into
# a shared number and which add into their own. A `PairwiseReduction` says
# that, so the maths underneath is written once.

"""
    PairwiseReduction

How the pairwise likelihood is split into groups, for example by household,
age stratum or spatial patch, so that each group's log-likelihood is reported
separately. A subtype defines [`ngroups`](@ref EpiBranch.ngroups) (how many
groups) and [`group`](@ref EpiBranch.group) (which group a person's terms add
into), and nothing else. [`pairwise_surv_loglik`](@ref) puts everything into
one group; [`pairwise_surv_loglik_by_component`](@ref) groups by household or
connected part of the network. A new grouping is written the same way from
outside the package and evaluated with [`pairwise_reduce`](@ref
EpiBranch.pairwise_reduce).
"""
abstract type PairwiseReduction end

"""
    ngroups(reduction::PairwiseReduction) -> Int

How many groups `reduction` splits the pairwise likelihood into. A
[`PairwiseReduction`](@ref) subtype defines this.
"""
function ngroups(reduction::PairwiseReduction)
    throw(
        ArgumentError(
            "$(nameof(typeof(reduction))) needs a method for " *
                "`EpiBranch.ngroups` giving how many groups it sums rows into"
        )
    )
end

"""
    group(reduction::PairwiseReduction, host::Int) -> Int

Which of `reduction`'s `1:ngroups(reduction)` groups the likelihood terms of
person `host` add into. A [`PairwiseReduction`](@ref) subtype defines this.
"""
function group(reduction::PairwiseReduction, host)
    throw(
        ArgumentError(
            "$(nameof(typeof(reduction))) needs a method for " *
                "`EpiBranch.group` naming which group a host's rows add into"
        )
    )
end

# `pairwise_surv_loglik`'s grouping: every row shares the one group its
# scalar result is returned as.
struct _TotalLogLik <: PairwiseReduction end
ngroups(::_TotalLogLik) = 1
group(::_TotalLogLik, host) = 1

# `pairwise_surv_loglik_by_component`'s grouping: a row's group is its
# susceptible's connected component (shared with its infector, since the
# contact structure is what makes them a possible pair), read off the layout
# it was compiled from.
struct _ByComponent <: PairwiseReduction
    component::Vector{Int}
    ncomponents::Int
end
ngroups(r::_ByComponent) = r.ncomponents
group(r::_ByComponent, host) = r.component[host]

# The running state of a reduction: one number per group, and which groups
# are already known impossible; their rows are then skipped and their number
# stays -Inf regardless of what else would be added. A group's number widens
# if a row adds a wider type than it currently holds: a covariate kernel may
# hold its fitted parameters on only some rows, and the type probe behind the
# initial `T` can miss them. `pairwise_surv_loglik`'s single running total
# used to be a bare local and widened the same way for free; a vector element
# cannot, which is why `_add!` below checks and widens explicitly instead.
struct _GroupTotals{T}
    ll::Vector{T}
    infeasible::Vector{Bool}
end
function _GroupTotals(reduction::PairwiseReduction, ::Type{T}) where {T}
    n = ngroups(reduction)
    return _GroupTotals{T}(zeros(T, n), falses(n))
end

function _add!(reduction::PairwiseReduction, totals::_GroupTotals{T}, host, Δ) where {T}
    g = group(reduction, host)
    totals.infeasible[g] && return totals
    S = promote_type(T, typeof(Δ))
    widened = S === T ? totals : _GroupTotals{S}(convert(Vector{S}, totals.ll), totals.infeasible)
    widened.ll[g] += Δ
    return widened
end

function _infeasible!(reduction::PairwiseReduction, totals::_GroupTotals{T}, host) where {T}
    g = group(reduction, host)
    totals.ll[g] = T(-Inf)
    totals.infeasible[g] = true
    return totals
end

_is_infeasible(reduction::PairwiseReduction, totals::_GroupTotals, host) =
    totals.infeasible[group(reduction, host)]

# A reduction's result is its vector of group totals, bar `_TotalLogLik`'s
# single group, which is returned as the bare scalar `pairwise_surv_loglik`
# promises.
_result(::PairwiseReduction, totals::_GroupTotals) = totals.ll
_result(::_TotalLogLik, totals::_GroupTotals) = totals.ll[1]

"""
    pairwise_reduce(reduction::PairwiseReduction, kernel, data::InfectionLayer,
                     layout::ContactPairsLayout; external_hazard = 0.0,
                     susceptibility = nothing) -> Vector{<:Real}

The pairwise log-likelihood split into the groups of `reduction`, a
[`PairwiseReduction`](@ref): returns one log-likelihood per group,
`ngroups(reduction)` in all. [`pairwise_surv_loglik`](@ref) and
[`pairwise_surv_loglik_by_component`](@ref) are this call with their own
groupings; a new grouping (by stratum, by spatial patch) calls it directly with
its own `PairwiseReduction` subtype. The other arguments are as in
`pairwise_surv_loglik`.
"""
function pairwise_reduce(
        reduction::PairwiseReduction, kernel, data::InfectionLayer,
        layout::ContactPairsLayout; external_hazard = 0.0, susceptibility = nothing
    )
    extdist, tfollow, T = _pairwise_setup(kernel, data, layout, external_hazard)
    mixtures = _host_mixtures(susceptibility, data, layout)
    # A per-edge or covariate kernel's parameter type is only known at run time;
    # the function barrier keeps the passes type-stable.
    return _pairwise_surv_loglik(
        kernel, extdist, data, layout, tfollow, reduction,
        promote_type(T, _mixtures_partype(mixtures)), mixtures
    )
end

function _pairwise_surv_loglik(
        kernel, extdist, data, layout, tfollow,
        reduction::PairwiseReduction, ::Type{T}, mixtures = nothing
    ) where {T}
    totals = _GroupTotals(reduction, T)
    # An infected host that is not conditioned on and has no possible infector
    # cannot have been infected, unless that infection falls after the end of
    # follow-up; mark its group before any pass runs.
    @inbounds for j in layout.no_rows
        tj = data.infection_time[j]
        (isnan(tj) || tj > tfollow) || (totals = _infeasible!(reduction, totals, j))
    end
    if !all(totals.infeasible)
        gaps = _removal_gaps(data)
        totals = _pairwise_cumhazard(
            reduction, kernel, extdist, data, layout, tfollow, totals, mixtures, gaps
        )
        totals = _pairwise_events(
            reduction, kernel, extdist, data, layout, tfollow, totals, mixtures, gaps
        )
        totals = _pairwise_mixtures(
            reduction, kernel, extdist, data, layout, tfollow, totals, mixtures, gaps
        )
    end
    return _result(reduction, totals)
end

# Pass 1: cumulative-hazard contribution per row, each at risk from 0, added
# into its susceptible's group. A susceptible is exposed to its possible
# infectors until it is infected, and to the community hazard until the
# earlier of that and `obs_end`, after which there are no more introductions.
# Nothing is at risk after the end of follow-up, and a host infected after it
# has escaped until then as far as the data show.
function _pairwise_cumhazard(reduction, kernel, extdist, data, layout, tfollow, totals, mixtures = nothing, gaps = nothing)
    sus = layout.sus
    infector = layout.infector
    is_ext = layout.is_ext

    @inbounds for row in eachindex(sus)
        j = sus[row]
        _is_infeasible(reduction, totals, j) && continue
        _is_mixed(mixtures, j) && continue
        tj = data.infection_time[j]
        tend = (isnan(tj) || tj > tfollow) ? tfollow : convert(typeof(tfollow), tj)
        if is_ext[row]
            stop = min(tend, data.obs_end)
            stop > 0 || continue
            totals = _add!(reduction, totals, j, -cumhazard(extdist, stop))
        else
            i = infector[row]
            oi = data.infectious_time[i]
            isfinite(oi) || continue
            oi < tend || continue
            stop = min(data.removal_time[i], tend) - oi
            stop > 0 || continue
            pk = _pair_kernel(kernel, layout, row, data)
            h = _gapped_cumhazard(
                t -> cumhazard(pk, t), _host_stretches(gaps, i), oi, stop
            )
            totals = _add!(reduction, totals, j, -h)
        end
    end
    return totals
end

# Pass 2: per-susceptible log-sum-exp over event rows, added into (or dooming)
# its susceptible's group. A single accumulator is reused across susceptibles
# (reset per susceptible), keeping the reduction allocation-free on the AD
# tape. Every host in the layout is explained: an infected one with no
# positive hazard at its infection time has density zero, and its group is
# impossible: mark it rather than adding it, so that the derivative is zero
# too. Adding it would leave the derivatives of the other hosts' finite terms
# sitting alongside an infinite value, whereas the log-density is -Inf
# throughout a neighbourhood of the parameters, because impossibility is a
# discrete fact of the fixed times.
function _pairwise_events(reduction, kernel, extdist, data, layout, tfollow, totals, mixtures = nothing, gaps = nothing)
    sus = layout.sus
    infector = layout.infector
    is_ext = layout.is_ext

    T = eltype(totals.ll)
    acc = _LogSumExpAcc{T}()
    @inbounds for g in eachindex(layout.sus_unique)
        j = layout.sus_unique[g]
        _is_infeasible(reduction, totals, j) && continue
        _is_mixed(mixtures, j) && continue
        tj = data.infection_time[j]
        (isnan(tj) || tj > tfollow) && continue
        acc.m = T(-Inf)
        acc.s = zero(T)
        acc.nseen = 0
        for k in layout.sus_row_ranges[g]
            row = layout.sus_row_order[k]
            if is_ext[row]
                # a host infected after `obs_end` was infected along a contact;
                # one infected at 0 is a community case like any other
                (tj >= 0 && tj <= data.obs_end) || continue
                _push!(acc, loghazard(extdist, tj))
            else
                i = infector[row]
                oi = data.infectious_time[i]
                isfinite(oi) || continue
                if oi < tj && tj <= data.removal_time[i] && !_removed_at(gaps, i, tj)
                    _push!(acc, loghazard(_pair_kernel(kernel, layout, row, data), tj - oi))
                end
            end
        end
        v = _value(acc)
        totals = _is_minus_inf(v) ? _infeasible!(reduction, totals, j) : _add!(reduction, totals, j, v)
    end

    return totals
end

# Pass 3: a susceptible whose susceptibility an effect modifies is left out of
# both flat passes and evaluated on its own here, as the log-sum-exp over its
# mixture components of the log weight plus the escape and event terms under
# that component's modifier. As in the event pass, a mixture of density zero
# makes its group impossible.
_pairwise_mixtures(
    reduction, kernel, extdist, data, layout, tfollow, totals, ::Nothing, gaps = nothing
) = totals
function _pairwise_mixtures(
        reduction, kernel, extdist, data, layout, tfollow, totals,
        mixtures::AbstractDict, gaps = nothing
    )
    for g in eachindex(layout.sus_unique)
        j = layout.sus_unique[g]
        _is_infeasible(reduction, totals, j) && continue
        mixture = get(mixtures, j, nothing)
        mixture === nothing && continue
        T = eltype(totals.ll)
        v = _mixture_loglik(mixture, kernel, extdist, data, layout, g, tfollow, T, gaps)
        totals = _is_minus_inf(v) ? _infeasible!(reduction, totals, j) : _add!(reduction, totals, j, v)
    end
    return totals
end

# ── Per-component contributions ──────────────────────────────────────
#
# A sampler that updates the latent infection layer one connected component at
# a time (a household, say) accepts or rejects the move on that component's
# own share of the log-likelihood. Every row's susceptible and infector share a
# component, since the contact structure is what makes them a possible pair, so
# the total is exactly the sum of the per-component contributions below.

"""
    pairwise_surv_loglik_by_component(kernel, data::InfectionLayer, layout::ContactPairsLayout;
                                      external_hazard = 0.0,
                                      susceptibility = nothing) -> Vector{<:Real}
    pairwise_surv_loglik_by_component(kernel, data::InfectionLayer;
                                      external_hazard = 0.0,
                                      susceptibility = nothing) -> Vector{<:Real}

[`pairwise_surv_loglik`](@ref) split by household (or by connected part of a
contact network): entry `c` is the log-likelihood of group `c` of `layout`,
and `sum(pairwise_surv_loglik_by_component(...)) == pairwise_surv_loglik(...)`.
`layout.component` gives each person's group and `layout.ncomponents` their
number.

A group containing an infected person whom no possible infector can explain
gets `-Inf`, as does the total, but the other groups keep their finite values.
A sampler that imputes infection times one household at a time can therefore
accept or reject each proposal on its own entry without rebuilding the layout.

The arguments are as in [`pairwise_surv_loglik`](@ref); see also
[`PairwiseReduction`](@ref) for other groupings.
"""
function pairwise_surv_loglik_by_component(
        kernel, data::InfectionLayer, layout::ContactPairsLayout;
        external_hazard = 0.0, susceptibility = nothing
    )
    reduction = _ByComponent(layout.component, layout.ncomponents)
    return pairwise_reduce(
        reduction, kernel, data, layout; external_hazard, susceptibility
    )
end

function pairwise_surv_loglik_by_component(
        kernel, data::InfectionLayer; external_hazard = 0.0,
        susceptibility = nothing
    )
    layout = compile_contact_pairs(data; external = _ext_active(external_hazard))
    return pairwise_surv_loglik_by_component(kernel, data, layout; external_hazard, susceptibility)
end
