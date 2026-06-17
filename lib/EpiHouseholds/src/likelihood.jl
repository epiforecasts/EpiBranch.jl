# ── Pairwise survival likelihood ─────────────────────────────────────
#
# The contact-process density: the log-likelihood of a household's *infection
# layer* (who is infected, and the latent infection time and infectious window
# of each) under a contact-interval kernel. It is the marginal pairwise
# likelihood of Kenah (2011): who-infected-whom and the order of infections are
# both unobserved, so each infected susceptible's contribution sums the
# contact-interval hazard over every possible infector — no ordering assumed.
#
# This scores the infection layer, which is *latent*: observed in a `simulate`
# round-trip, augmented in inference (where the progression then links each
# infection to its observed onset/test). It never takes onsets as the event.

"""
    PairwiseSurvivalData(sus, start, stop, event)

Counting-process rows for [`pairwise_surv_loglik`](@ref). Row `r` is an ordered
at-risk interval `(start[r], stop[r]]` for susceptible `sus[r]`, with `event[r]`
true if an infectious contact occurred at `stop[r]`. Several rows share a
susceptible — its possible infectors. The form is space- and order-agnostic: it
knows only at-risk intervals and events, never an infection order.
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

# Promote the time-type so callers don't have to spell it out; widens to Float64
# when the inputs are integers.
function PairwiseSurvivalData(sus, start, stop, event)
    T = promote_type(eltype(start), eltype(stop), Float64)
    return PairwiseSurvivalData{T}(sus, start, stop, event)
end

Base.length(d::PairwiseSurvivalData) = length(d.sus)

# Row r's contact-interval distribution: a shared distribution, or a per-row
# callable `r -> Distribution` (the seam where covariates enter).
_rowkernel(k::ContinuousUnivariateDistribution, r) = k
_rowkernel(k, r) = k(r)

_logsumexp(xs) = (m = maximum(xs); m + log(sum(x -> exp(x - m), xs)))

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
covariates. The result is differentiable in the kernel's parameters, so it drops
into Optim or Turing's `@addlogprob!`.
"""
function pairwise_surv_loglik(kernel, data::PairwiseSurvivalData)
    groups = Dict{Int, Vector{Int}}()
    for r in eachindex(data.event)
        data.event[r] && push!(get!(groups, data.sus[r], Int[]), r)
    end

    ll = 0.0
    for (_, g) in groups
        ll += _logsumexp([loghazard(_rowkernel(kernel, r), data.stop[r]) for r in g])
    end
    for r in eachindex(data.stop)
        kr = _rowkernel(kernel, r)
        ll -= cumhazard(kr, data.stop[r])
        data.start[r] > 0 && (ll += cumhazard(kr, data.start[r]))
    end
    return ll
end

# ── The household infection layer ────────────────────────────────────

"""
    HouseholdInfections(household_of, infection_time, infectious_time, removal_time, is_index)

The infection layer of a household outbreak: per individual, their household,
infection time (`NaN` if never infected), infectiousness onset and removal (the
infectious-window endpoints; `removal_time = Inf` when not yet removed, i.e.
right-censored), and whether they were introduced from outside the household.

These are the latent quantities the contact process is a density over — read out
of a `simulate` round-trip with [`household_infections`](@ref), or augmented in
inference. Onsets, tests and other observables are *not* here: they are the
progression's outputs, conditioned separately.
"""
struct HouseholdInfections{T <: Real}
    household_of::Vector{Int}
    infection_time::Vector{T}
    infectious_time::Vector{T}
    removal_time::Vector{T}
    is_index::Vector{Bool}
    obs_end::T
end

# `obs_end` is the end of follow-up — the window the community hazard acts over;
# only used when the model carries an external hazard, and may be left `Inf`.
function HouseholdInfections(household_of, infection_time, infectious_time,
        removal_time, is_index; obs_end = Inf)
    T = promote_type(eltype(infection_time), eltype(infectious_time),
        eltype(removal_time), typeof(obs_end), Float64)
    return HouseholdInfections{T}(collect(Int, household_of),
        Vector{T}(infection_time),
        Vector{T}(infectious_time),
        Vector{T}(removal_time),
        Vector{Bool}(is_index),
        T(obs_end))
end

Base.length(d::HouseholdInfections) = length(d.household_of)

"""
    household_infections(state, model) -> HouseholdInfections

Read the infection layer out of a simulated `state`: each member's household,
infection time, infectiousness onset (the model's `from` state) and removal (the
earliest of its `until` states), and index status. The exact `simulate →
loglikelihood` round trip goes through this.
"""
function household_infections(state::SimulationState, model::HouseholdProcess;
        obs_end = Inf)
    inds = state.individuals
    n = length(inds)
    hh = Vector{Int}(undef, n)
    infection = fill(NaN, n)
    infectious = fill(NaN, n)
    removal = fill(Inf, n)
    index = falses(n)
    for (k, ind) in enumerate(inds)
        hh[k] = ind.state[:household]::Int
        if get(ind.state, :infected, false)
            infection[k] = ind.infection_time
            infectious[k] = _window_open(ind, model.from)
            removal[k] = _window_close(ind, model.until)
            index[k] = get(ind.state, :index, false)
        end
    end
    return HouseholdInfections(hh, infection, infectious, removal, index; obs_end)
end

# Counting-process rows for one household's pairs, with — per row — the global
# infector id (so a covariate kernel can be routed) and whether the row is the
# community term (`infector = 0`, calendar-time at-risk over `[0, tend]`). Without
# a community term index cases are conditioned on (they appear only as infectors);
# with one they are explained like any other case, so they get rows too.
function _survival_rows(d::HouseholdInfections{T}; external::Bool = false,
        obs_end::T = T(Inf)) where {T <: Real}
    # Bucket hosts by household into a `Vector{Vector{Int}}` indexed by id
    # offset (no hashing) and walk the buckets in two passes — first to
    # count rows, then to fill preallocated output arrays. Avoiding the
    # `Dict` and the per-row `push!` matters under reverse-mode AD, which
    # tracks every allocation. Offsetting by `lo` tolerates any integer
    # ids (sparse ids leave empty buckets, which are skipped).
    isempty(d.household_of) &&
        return PairwiseSurvivalData{T}(Int[], T[], T[], Bool[]), Int[], Bool[]
    lo, hi = extrema(d.household_of)
    n_buckets = hi - lo + 1
    households = [Int[] for _ in 1:n_buckets]
    for i in eachindex(d.household_of)
        push!(households[d.household_of[i] - lo + 1], i)
    end

    # ── Pass 1: count rows ──────────────────────────────────────────────
    n_rows = 0
    for h in 1:n_buckets
        mem = households[h]
        isempty(mem) && continue
        for j in mem
            (!external && d.is_index[j]) && continue
            tj = d.infection_time[j]
            infected_j = !isnan(tj)
            tend = infected_j ? tj : (external ? obs_end : T(Inf))
            external && (n_rows += 1)
            for i in mem
                isfinite(d.infectious_time[i]) || continue
                i == j && continue
                oi = d.infectious_time[i]
                oi < tend || continue
                A = min(d.removal_time[i], tend) - oi
                A > 0 || continue
                n_rows += 1
            end
        end
    end

    # ── Pass 2: fill ────────────────────────────────────────────────────
    sus = Vector{Int}(undef, n_rows)
    start = Vector{T}(undef, n_rows)
    stop = Vector{T}(undef, n_rows)
    event = Vector{Bool}(undef, n_rows)
    infector = Vector{Int}(undef, n_rows)
    is_ext = Vector{Bool}(undef, n_rows)
    r = 0
    for h in 1:n_buckets
        mem = households[h]
        isempty(mem) && continue
        for j in mem
            (!external && d.is_index[j]) && continue
            tj = d.infection_time[j]
            infected_j = !isnan(tj)
            tend = infected_j ? tj : (external ? obs_end : T(Inf))
            if external
                r += 1
                sus[r] = j
                start[r] = zero(T)
                stop[r] = tend
                event[r] = infected_j
                infector[r] = 0
                is_ext[r] = true
            end
            for i in mem
                isfinite(d.infectious_time[i]) || continue
                i == j && continue
                oi = d.infectious_time[i]
                oi < tend || continue
                A = min(d.removal_time[i], tend) - oi
                A > 0 || continue
                r += 1
                sus[r] = j
                start[r] = zero(T)
                stop[r] = A
                event[r] = infected_j && oi < tj <= d.removal_time[i]
                infector[r] = i
                is_ext[r] = false
            end
        end
    end
    return PairwiseSurvivalData{T}(sus, start, stop, event), infector, is_ext
end

"""
    pairwise_surv_loglik(kernel, data::HouseholdInfections; external_hazard = 0.0) -> Float64

The household contact-process log-density for a `kernel` and (latent) infection
layer `data` — the **inference-friendly** form. In a household `@model` the
`kernel` and `external_hazard` carry the parameters being fit while `data` is
the augmented infection layer, so neither a `HouseholdProcess` nor the household
structure is rebuilt per evaluation. Differentiable in the kernel's parameters,
so it drops into Turing's `@addlogprob!`. With `external_hazard` set a community
term explains index cases over `data.obs_end`; otherwise they are conditioned on.
"""
function pairwise_surv_loglik(kernel, data::HouseholdInfections; external_hazard = 0.0)
    external = _ext_active(external_hazard)
    rows, infector, is_ext = _survival_rows(data; external, obs_end = data.obs_end)
    if !external && kernel isa ContinuousUnivariateDistribution
        return pairwise_surv_loglik(kernel, rows)   # shared-kernel fast path
    end
    extdist = external ? _ext_survival(external_hazard) : kernel
    rowkernel = r -> is_ext[r] ? extdist : _pair(kernel, infector[r], rows.sus[r])
    return pairwise_surv_loglik(rowkernel, rows)
end

"""
    loglikelihood(data::HouseholdInfections, model::HouseholdProcess) -> Float64

The contact-process log-density of `model`'s kernel given the infection layer
`data` — sugar for `pairwise_surv_loglik(model.kernel, data; external_hazard =
model.external_hazard)`, and the exact `simulate → loglikelihood` round trip. The
observed onsets/tests are conditioned separately through the progression — there
is deliberately no `loglikelihood(onsets, model)`, since the latent infections
cannot be marginalised in closed form.
"""
function Distributions.loglikelihood(data::HouseholdInfections, model::HouseholdProcess)
    pairwise_surv_loglik(model.kernel, data; external_hazard = model.external_hazard)
end

# Resolve a covariate pair-kernel; a shared distribution ignores the pair.
_pair(k::ContinuousUnivariateDistribution, i, j) = k
_pair(k, i, j) = k(i, j)

# The community hazard as a calendar-time survival distribution: a constant rate
# α is `Exponential(1/α)` (hazard α, cumulative α·t); a distribution is itself.
_ext_survival(α::Real) = Exponential(1 / α)
_ext_survival(d::ContinuousUnivariateDistribution) = d
