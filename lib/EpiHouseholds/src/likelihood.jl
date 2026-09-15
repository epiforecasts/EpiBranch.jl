# ── Household pairwise survival likelihood ───────────────────────────
#
# Household methods for EpiBranch's pairwise survival likelihood. The density,
# its compiled pair layout and the external hazard term work for any contact
# structure and live in EpiBranch. A household population supplies its partition
# as the contact structure, so household-mates are each other's possible
# infectors.

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
struct HouseholdInfections{T <: Real} <: InfectionLayer
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

# Household-mates are each other's possible infectors.
EpiBranch.contact_structure(d::HouseholdInfections) = d.household_of

"""
    household_infections(state, model::ModelSpec) -> HouseholdInfections

Read the infection layer out of a simulated `state`: each member's household,
infection time, infectiousness onset (the infectious-window `from` state) and
removal (the earliest of its `until` states), and index status. The infectious
window is read from the same composed progression the simulation used, so the
`simulate → loglikelihood` round trip is exact. A bare `HouseholdProcess` is
accepted too (its window opens at `:infection`).
"""
function household_infections(state::SimulationState,
        model::ModelSpec{<:HouseholdProcess}; obs_end = model.process.obs_end)
    process = model.process
    from = _resolve_infectious_from(process.from, model.progression)
    until = process.until
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
            infectious[k] = _window_open(ind, from)
            removal[k] = _window_close(ind, until)
            index[k] = get(ind.state, :index, false)
        end
    end
    return HouseholdInfections(hh, infection, infectious, removal, index; obs_end)
end

function household_infections(state::SimulationState, process::HouseholdProcess;
        kwargs...)
    return household_infections(state, ModelSpec(process); kwargs...)
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

function Distributions.loglikelihood(data::HouseholdInfections,
        model::ModelSpec{<:HouseholdProcess})
    loglikelihood(data, model.process)
end

# Resolve a covariate pair-kernel; a shared distribution ignores the pair.
_pair(k::ContinuousUnivariateDistribution, i, j) = k
_pair(k, i, j) = k(i, j)

# ── Compiled pair layout ─────────────────────────────────────────────

"""
    HouseholdPairsLayout

The compiled pair layout for a household population. It is another name for
EpiBranch's [`ContactPairsLayout`](@ref), used when the layout is built from a
household partition. Each row is one ordered (susceptible, household-mate) pair
that the contact process scores.

Build it with [`compile_household_pairs`](@ref).
"""
const HouseholdPairsLayout = ContactPairsLayout

"""
    compile_household_pairs(household_of, is_index, infected; external=false)
    compile_household_pairs(data::HouseholdInfections; external=false)

Pre-compute the structural pair list for the inference fast path. `infected`
is the static at-risk mask — true iff the host appears in the posterior as
an infected case (its `infection_time` will be augmented). The single-arg
form reads the mask off `data` as `.!isnan.(data.infection_time)`.

With `external=true` each susceptible gets an additional row for the community
hazard term; otherwise index cases are conditioned on and contribute only as
infectors. This is [`compile_contact_pairs`](@ref) on the household partition.
Evaluate the result with
`pairwise_surv_loglik(kernel, data, layout; external_hazard)`.
"""
function compile_household_pairs(household_of::AbstractVector{<:Integer},
        is_index::AbstractVector{Bool},
        infected::AbstractVector{Bool};
        external::Bool = false)
    return compile_contact_pairs(household_of, is_index, infected; external)
end

function compile_household_pairs(d::HouseholdInfections; external::Bool = false)
    return compile_contact_pairs(d; external)
end
