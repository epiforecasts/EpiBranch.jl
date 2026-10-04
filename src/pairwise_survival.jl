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

Counting-process rows for [`pairwise_surv_loglik`](@ref). Row `r` is an ordered
at-risk interval `(start[r], stop[r]]` for susceptible `sus[r]`, with `event[r]`
true if an infectious contact occurred at `stop[r]`. A susceptible has one row
per possible infector. The rows record only at-risk intervals and events, with
no spatial structure and no infection order.
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

Marginal pairwise survival log-likelihood on counting-process rows. Each
susceptible contributes the log of its summed hazard over its event rows (its
possible infectors), minus the cumulative hazard every row accrues over its
at-risk interval:

    ll = Σ_susceptible log Σ_{event rows} hazard(stop)
         − Σ_rows [cumhazard(stop) − cumhazard(start)]

Right-censoring is built in: a susceptible that never had an event contributes
only the escaped cumulative hazard. `kernel` is a `Distributions.jl`
distribution shared by every row, or a callable `r -> Distribution` for
covariates. The result is differentiable in the kernel's parameters and can be
optimised with Optim or added to a Turing model with `@addlogprob!`.
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

Supertype for an outbreak's infection layer together with the contact structure
it spread over. The pairwise likelihood is a density over this data. A subtype
holds, per host `i` (numbered `1:n`):

- `infection_time[i]`: the infection time, `NaN` if never infected;
- `infectious_time[i]`: when the infectious window opens;
- `removal_time[i]`: when it closes;
- `is_index[i]`: whether the host was introduced from outside the structure;

and a scalar `obs_end`, the time community introductions stop (only read when
there is a community hazard). Spread along the contact structure continues after
it. A subtype may also hold `host_times`, a named tuple of further per-host time
vectors such as `onset_time`, which a live [`PairKernel`](@ref) or a
susceptibility effect reads in the
likelihood as it reads host state in simulation. `missing` marks a host without
that time; a `NaN` entry is a recorded value, as simulation stores the onset of
an asymptomatic case.

Observed data stop at the end of follow-up, which
[`followup_end`](@ref EpiBranch.followup_end) gives: a `followup_end` field when
the subtype has one, and `Inf` otherwise. The likelihood ignores everything after
it. A host infected later counts as escaped until then, and exposure to a
possible infector stops there. A case still infectious at the end of follow-up
can therefore keep a removal time of `Inf`. Simulated outbreaks run to completion
and need no end of follow-up. A subtype also defines
[`contact_structure`](@ref EpiBranch.contact_structure), which says who could
have infected whom. [`compile_contact_pairs`](@ref) and
[`pairwise_surv_loglik`](@ref) then work on it with no further methods.
`HouseholdInfections` (in `EpiHouseholds`) and `NetworkInfections` (in
`EpiNetwork`) are the worked examples.

The infection layer is latent: it is known exactly after a simulation and
augmented in inference. Observables such as onsets and tests are outputs of the
progression and are conditioned on separately — [`progression_loglik`](@ref)
evaluates that term, so the sum of the two is the full log-likelihood of the
augmented data. There is no likelihood of the onsets alone, since the latent
infections cannot be marginalised in closed form.

A companion package reads a simulated outbreak back into its layer
(`household_infections`, `network_infections`). Each infected host's window
opens at the process's `from` state and closes at the earliest of its `until`
states and the time the model's interventions take the host out of transmission,
such as by isolation or quarantine after tracing. These are the windows the
simulation used. Exact evaluation also requires the kernel to include every
other hazard modification, apart from changes to a host's own susceptibility,
which the composed components declare through
[`susceptibility_components`](@ref EpiBranch.susceptibility_components) from the
host times they read. Structured `loglikelihood(data, spec)` methods check the
composed components using [`infection_likelihood_compatible`](@ref). Use
`pairwise_surv_loglik` with an explicit effective kernel when additional effects
must be represented. Passing a reader the `followup_end` keyword evaluates the
outbreak as if observation had stopped at that time.
"""
abstract type InfectionLayer end

"""
    contact_structure(data::InfectionLayer)

Who could have infected whom in `data`, in a form
[`compile_contact_pairs`](@ref) accepts: a membership vector (hosts sharing a
label can all infect one another, as in a household partition) or an adjacency
list (`contacts[i]` lists the hosts `i` can infect, as in a directed contact
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

The end of follow-up of `data`, the time its observation stops. The pairwise
likelihood covers the infection layer up to it and ignores infections and
exposure after it. The default reads a `followup_end` field when the
[`InfectionLayer`](@ref) subtype has one, and is `Inf` otherwise; a subtype that
stores it elsewhere defines a method.
"""
function followup_end(data::InfectionLayer)
    return hasproperty(data, :followup_end) ?
        data.followup_end : Inf
end

# The per-host times of an infection layer beyond its infectious windows, as a
# named tuple of vectors; empty when the subtype holds none.
_host_times(data) = hasproperty(data, :host_times) ? data.host_times : (;)

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
        map(v -> nonmissingtype(eltype(v)), values(host_times))..., Float64
    )
    return (
        Vector{T}(infection_time), Vector{T}(infectious_time),
        Vector{T}(removal_time), Vector{Bool}(is_index), T(obs_end), T(followup_end),
        map(v -> Vector{Missing <: eltype(v) ? Union{Missing, T} : T}(v), host_times),
    )
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

Declare that a composed component's effects on infection hazards are fully
represented by the infectious opening and removal times in an [`InfectionLayer`](@ref).
The default is `false`. External components may opt in when they change only
these times, change a host's susceptibility only through
[`susceptibility_components`](@ref EpiBranch.susceptibility_components), or have
no effect on infection hazards. Partial blocking, infectiousness multipliers, and
altered contact kernels require an explicitly effective kernel instead.

This declaration is a modelling contract. It does not evaluate callbacks or
verify their side effects. The infection likelihood conditions on the supplied
infection layer; it excludes the probability of clinical outcomes, intervention
assignment, attribute draws and observations.
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

A modifier of one susceptible's infection hazard, from every possible infector
and the community alike: from calendar time `start` on, each hazard is
multiplied by `factor`. `factor` is a non-negative `Real`, or a function
`dt -> Real` of the time since `start` for a multiplier that changes over time,
such as protection that wanes. Before `start` the hazard is unchanged.

A component of a [`susceptibility_components`](@ref
EpiBranch.susceptibility_components) mixture.
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

How `effect` modifies the infection hazard of one susceptible `host` of an
[`InfectionLayer`](@ref), for [`pairwise_surv_loglik`](@ref)'s `susceptibility`
keyword. `host` is a [`LayerHost`](@ref): its `id`, `infection_time`, and the
layer's `host_times` under `host.state`, read as a live [`PairKernel`](@ref)
projection reads them.

The return value is `nothing`, the default, when `effect` leaves the host's
hazard as it is. Otherwise it is a collection of `weight => modifier` pairs: the
host's contribution to the likelihood is the mixture over them, each weight
times the likelihood of the host's escapes and infection with every hazard it
faces modified by that component's [`HazardScaling`](@ref EpiBranch.HazardScaling)
(or `nothing` for no modification). The weights are probabilities summing to
one. One component describes an effect every exposure shares; several describe
a host-level state that is drawn once and is not observed, which then governs
all of that host's exposures together.

A vaccination's `VaccineEffect` gives one component under `LeakyMode`,
`1 => HazardScaling(τ, 1 - efficacy)` from the host's immunity time `τ` (with
`waning`, the factor is `dt -> 1 - efficacy * waning(dt)`). Under
`AllOrNothingMode` it gives two: `efficacy => HazardScaling(τ, 0.0)` for a
responder and `1 - efficacy => nothing` for a non-responder. The immunity time
is read from the host time `:immunity_time` (`:immunity_time_<label>` for a
labelled dose), and a host without one is unmodified. Every
`AbstractVaccination` answers with its `VaccineEffect`, and an
`InterventionWrapper` with the intervention it wraps.

A collection of components, such as a model's interventions, gives the one
non-`nothing` answer among them, and raises an `ArgumentError` if more than one
component modifies the same host. Define a method for a new effect type, and
[`susceptibility_host_times`](@ref EpiBranch.susceptibility_host_times) for the
host times it reads.
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

The per-host times `component`'s [`susceptibility_components`](@ref
EpiBranch.susceptibility_components) reads, which
`household_infections` and `network_infections` record in the infection layer's
`host_times` whenever the model composes `component`. The default is `()`. An
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
    return keys
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

The static row structure the pairwise likelihood is evaluated on. Each row is
one ordered (susceptible, possible infector) pair, plus, when a community hazard
is modelled, one row per susceptible for the community hazard. Rows whose times
do not overlap are kept and skipped at evaluation. One layout then works for
every configuration of latent times with the same structure and infected set.

`component` gives each host's connected component of the contact structure (a
household on a household partition, or a connected component of a contact
network), numbered `1:ncomponents`. [`pairwise_surv_loglik_by_component`](@ref)
reads it to attribute the likelihood to the groups a sampler updating the
infection layer group by group accepts or rejects separately.

Build it with [`compile_contact_pairs`](@ref).
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

Enumerate the (susceptible, possible infector) rows of the pairwise likelihood
once, as a [`ContactPairsLayout`](@ref) to reuse across evaluations.

The contact structure is either a membership vector, where hosts sharing a label
can all infect one another (a partition into cliques, such as households), or an
adjacency list, where `contacts[i]` lists the hosts `i` can infect (a directed
network; list each edge both ways for an undirected one). A host's possible
infectors are then its group-mates or its in-neighbours. An edge listed twice is
two contact processes and contributes two rows.

`infected` is the static at-risk mask: true for a host that is infected in every
configuration the layout will evaluate (its infection time may still be augmented).
Only infected hosts can be infectors. `is_index` marks hosts introduced from
outside; without a community hazard they are conditioned on and appear only as
infectors. With `external = true` every host is explained, and each gets an
extra row for the community hazard. The single-argument form reads the structure
off `data` with [`contact_structure`](@ref EpiBranch.contact_structure) and the
mask as `.!isnan.(data.infection_time)`.
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

The contact-process log-density of the infection layer `data` under a
contact-interval `kernel`, marginal over who infected whom. Each susceptible
accrues cumulative hazard from every possible infector over the overlap of that
infector's infectious window with its own time at risk, and each infected one
adds the log of the summed hazard at its infection time. An infected host that
is not conditioned on and has no positive hazard at its infection time, such as
one infected when none of its possible infectors is infectious, makes the whole
configuration impossible, and the density is `-Inf` with a zero gradient.

`kernel` is a `Distributions.jl` distribution shared by every pair, a callable
`(infector, susceptible) -> Distribution` for covariates, a [`PairKernel`](@ref)
that also receives the infector's infection time (and, with host state, each
host's record), or a per-edge vector parallel to an adjacency list
(`kernel[i][k]` for host `i`'s `k`-th listed contact). `external_hazard` is a community hazard (a positive rate or a
calendar-time distribution) that introduces cases over `[0, data.obs_end]`. With
one, index cases are explained like any other case; without one they are
conditioned on. Each host accrues the community hazard until the earlier of its
infection and `data.obs_end`. A host infected after `obs_end` can only have
been infected by a possible infector. Spread along the contact structure
continues after `obs_end`: a host that is never infected accrues hazard over
each possible infector's whole infectious window.

`susceptibility` is an effect on the susceptibles' own hazards, such as a
candidate [`VaccineEffect`](@ref) or a model's interventions, and `nothing`
(the default) leaves every hazard as the kernel gives it. The effect says, per
host and through [`susceptibility_components`](@ref
EpiBranch.susceptibility_components), how it scales every hazard that host
faces, from its possible infectors and the community alike. A `VaccineEffect`
reads each host's immunity time from the layer's `host_times`, which
[`household_infections`](@ref EpiBranch.household_infections) and
[`network_infections`](@ref EpiBranch.network_infections) record for a model
that composes a vaccination; its `efficacy` must be a `Real`. Under
[`LeakyMode`](@ref), a vaccinated host's hazards from its immunity time on are
multiplied by `1 - efficacy`, or by `1 - efficacy * waning(dt)` with `waning`.
Under [`AllOrNothingMode`](@ref), its contribution is the mixture `efficacy *
Lᵖ + (1 - efficacy) * Lᵘ` over responder status: `Lᵖ` the likelihood of its
escapes and infection while fully protected from its immunity time on (zero if
it was infected after that time), and `Lᵘ` the unprotected one. Responder status
is drawn once per host and governs every exposure it faces, so the escape from
all of its possible infectors sits inside the mixture. Both are differentiable
in `efficacy`.

Everything is cut at [`followup_end(data)`](@ref EpiBranch.followup_end): a host
infected after it is treated as escaped until then, and no exposure accrues past
it. Evaluating data with an end of follow-up gives the same value as first
truncating the data there: later infections unobserved, and removal times and
`obs_end` capped at it.

Use the layout form in inference: compile the layout once with
[`compile_contact_pairs`](@ref) and reuse it while the latent times move. Its
`external` setting must agree with `external_hazard`. The two-argument form
compiles a layout on each call. Both are generic in the number type: the
kernel's parameters can be ForwardDiff or reverse-mode AD values. A `Gamma` is
the exception, whether it is the kernel or the community hazard: its cumulative
hazard calls `SpecialFunctions._gamma_inc`, which has no `ForwardDiff.Dual`
method. Fit a `Gamma` with a reverse-mode backend such as Mooncake. `Weibull`
and `Exponential` differentiate under either mode.

!!! warning "A vanishing community hazard is not the no-community case"
    The two are different conditionings, and the density jumps between them at
    `α = 0`. With `external_hazard = α > 0` an index case infected at time `t`
    contributes `log(α) − αt`, which falls to `-Inf` as `α → 0`, because a model
    that admits community introductions has to explain the ones it saw. At exactly `external_hazard = 0` index cases are instead
    conditioned on and contribute nothing, leaving a finite value. A likelihood
    ratio between "some community transmission" and "none" therefore cannot be
    read off by letting `α` approach zero: evaluate the two models separately.

    The discontinuity is at that one point. Approaching it, the log-density is
    `k log α − αT` up to terms free of `α`, where `k` counts the cases the
    community alone can explain and `T` is the total time hosts are exposed to
    it. In `log α` this is a straight line of slope `k`.
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
        modifier, kernel, extdist, data, layout, rg, tj, tfollow, ::Type{T}
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
            ll -= _scaled_cumhazard(modifier, _pair_kernel(kernel, layout, r, data), oi, stop)
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
            if oi < tj && tj <= data.removal_time[i]
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
        mixture, kernel, extdist, data, layout, g, tfollow, ::Type{T}
    ) where {T}
    j = layout.sus_unique[g]
    rg = layout.sus_row_ranges[g]
    tj = data.infection_time[j]
    # The weights enter linearly, never through `log(weight)`: a weight of
    # exactly zero would otherwise drop its component, and with it the
    # derivative of the mixture with respect to that weight.
    terms = map(mixture) do (weight, modifier)
        weight => _component_loglik(
            modifier, kernel, extdist, data, layout, rg, tj, tfollow, T
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

Supertype for how the two accumulation passes behind [`pairwise_surv_loglik`](@ref)
group rows into a result. [`ngroups`](@ref EpiBranch.ngroups) gives how many
groups a subtype has and [`group`](@ref EpiBranch.group) which group a host's
rows belong to; a subtype needs only these two methods; the passes themselves
do not change. [`pairwise_surv_loglik`](@ref) puts every row into the one
group its scalar result is; [`pairwise_surv_loglik_by_component`](@ref) groups
by the contact structure's connected components. A grouping by stratum or by
spatial patch is written the same way, from outside the package, and run with
[`pairwise_reduce`](@ref EpiBranch.pairwise_reduce).
"""
abstract type PairwiseReduction end

"""
    ngroups(reduction::PairwiseReduction) -> Int

How many groups `reduction` sums rows into. A [`PairwiseReduction`](@ref)
subtype defines this.
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

Which of `reduction`'s `1:ngroups(reduction)` groups `host`'s rows add into.
A [`PairwiseReduction`](@ref) subtype defines this.
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

Run the two accumulation passes behind [`pairwise_surv_loglik`](@ref) and
[`pairwise_surv_loglik_by_component`](@ref) under `reduction`, a
[`PairwiseReduction`](@ref), returning its `ngroups(reduction)` group
log-likelihoods. A new grouping (by stratum, by spatial patch) calls this
directly with its own `PairwiseReduction` subtype; `pairwise_surv_loglik` and
`pairwise_surv_loglik_by_component` are this call under their own built-in
groupings. Arguments are otherwise as in `pairwise_surv_loglik`.
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
        totals = _pairwise_cumhazard(reduction, kernel, extdist, data, layout, tfollow, totals, mixtures)
        totals = _pairwise_events(reduction, kernel, extdist, data, layout, tfollow, totals, mixtures)
        totals = _pairwise_mixtures(reduction, kernel, extdist, data, layout, tfollow, totals, mixtures)
    end
    return _result(reduction, totals)
end

# Pass 1: cumulative-hazard contribution per row, each at risk from 0, added
# into its susceptible's group. A susceptible is exposed to its possible
# infectors until it is infected, and to the community hazard until the
# earlier of that and `obs_end`, after which there are no more introductions.
# Nothing is at risk after the end of follow-up, and a host infected after it
# has escaped until then as far as the data show.
function _pairwise_cumhazard(reduction, kernel, extdist, data, layout, tfollow, totals, mixtures = nothing)
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
            totals = _add!(reduction, totals, j, -cumhazard(_pair_kernel(kernel, layout, row, data), stop))
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
function _pairwise_events(reduction, kernel, extdist, data, layout, tfollow, totals, mixtures = nothing)
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
                if oi < tj && tj <= data.removal_time[i]
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
_pairwise_mixtures(reduction, kernel, extdist, data, layout, tfollow, totals, ::Nothing) = totals
function _pairwise_mixtures(
        reduction, kernel, extdist, data, layout, tfollow, totals, mixtures::AbstractDict
    )
    for g in eachindex(layout.sus_unique)
        j = layout.sus_unique[g]
        _is_infeasible(reduction, totals, j) && continue
        mixture = get(mixtures, j, nothing)
        mixture === nothing && continue
        T = eltype(totals.ll)
        v = _mixture_loglik(mixture, kernel, extdist, data, layout, g, tfollow, T)
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

The per-component breakdown of [`pairwise_surv_loglik`](@ref): entry `c` sums
every term whose susceptible and infector lie in component `c` of `layout` (a
household on a household partition, or a connected component of a contact
network), with `sum(pairwise_surv_loglik_by_component(...)) ==
pairwise_surv_loglik(...)`. `layout.component` gives each host's component and
`layout.ncomponents` their count.

A component with an infected host that no possible infector can explain gets
`-Inf`, as does the total; unlike the total, the other components keep their
finite value, so a sampler that updates the infection layer component by
component can accept or reject each move on its own entry without recompiling
the layout.

Arguments, community-hazard handling and `susceptibility` are otherwise as in
[`pairwise_surv_loglik`](@ref), which this shares the [`PairwiseReduction`](@ref)
machinery with: the only difference is grouping by component instead of into
one total.
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
