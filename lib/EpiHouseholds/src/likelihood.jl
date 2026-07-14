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

# The community hazard as a calendar-time survival distribution: a constant rate
# α is `Exponential(1/α)` (hazard α, cumulative α·t); a distribution is itself.
_ext_survival(α::Real) = Exponential(1 / α)
_ext_survival(d::ContinuousUnivariateDistribution) = d

# ── Compiled pair layout (fast path for inference) ───────────────────
#
# In inference the household structure (`household_of`, `is_index`) and the
# set of ever-infected hosts are fixed across gradient evaluations — only the
# augmented `infection_time`/`infectious_time` move. The default
# `pairwise_surv_loglik(kernel, ::HouseholdInfections)` rebuilds the entire
# row structure (a `Dict`, six fresh `Vector{T}`, and per-row allocations)
# every call, which reverse-mode AD tracks and pays for repeatedly.
#
# `HouseholdPairsLayout` captures everything that doesn't depend on the
# augmented times: row → (susceptible, infector, is_ext) and a susceptible-
# grouped index for the log-sum-exp. With a layout in hand,
# `pairwise_surv_loglik(kernel, infections, layout)` runs in a single pass
# over rows with no `Dict` and a streaming logsumexp — the AD tape is much
# shorter.

"""
    HouseholdPairsLayout

Static row structure for [`pairwise_surv_loglik`](@ref) under a fixed
household population. Each row is one ordered (susceptible, infector) pair
that the contact process scores; the constructor enumerates the same rows
the dynamic `_survival_rows` would, but only once.

Build it with [`compile_household_pairs`](@ref).
"""
struct HouseholdPairsLayout
    sus::Vector{Int}                       # row r → susceptible host id
    infector::Vector{Int}                  # row r → infector host id (0 = external)
    is_ext::Vector{Bool}
    sus_unique::Vector{Int}                # susceptibles that have ≥1 row
    sus_row_ranges::Vector{UnitRange{Int}} # row indices in `sus_row_order`
    sus_row_order::Vector{Int}             # row indices, susceptible-grouped
    external::Bool
    nhosts::Int                            # population the layout was compiled for
end

Base.length(L::HouseholdPairsLayout) = length(L.sus)

"""
    compile_household_pairs(household_of, is_index, infected; external=false)
    compile_household_pairs(data::HouseholdInfections; external=false)

Pre-compute the structural pair list for the inference fast path. `infected`
is the static at-risk mask — true iff the host appears in the posterior as
an infected case (its `infection_time` will be augmented). The single-arg
form reads the mask off `data` as `.!isnan.(data.infection_time)`.

With `external=true` an additional row per susceptible carries the
community hazard term; otherwise index cases are conditioned on and
contribute only as infectors.
"""
function compile_household_pairs(household_of::AbstractVector{<:Integer},
        is_index::AbstractVector{Bool},
        infected::AbstractVector{Bool};
        external::Bool = false)
    n = length(household_of)
    (length(is_index) == n && length(infected) == n) ||
        throw(ArgumentError("household_of, is_index and infected must be same length"))
    isempty(household_of) && return HouseholdPairsLayout(
        Int[], Int[], Bool[], Int[], UnitRange{Int}[], Int[], external, 0)

    lo, hi = extrema(household_of)
    n_buckets = hi - lo + 1
    buckets = [Int[] for _ in 1:n_buckets]
    for i in 1:n
        push!(buckets[household_of[i] - lo + 1], i)
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

    HouseholdPairsLayout(sus, infector, is_ext, sus_unique, sus_row_ranges,
        sus_row_order, external, n)
end

function compile_household_pairs(d::HouseholdInfections; external::Bool = false)
    infected = .!isnan.(d.infection_time)
    compile_household_pairs(d.household_of, d.is_index, infected; external)
end

# Streaming logsumexp — replaces the `_logsumexp([...])` allocator from the
# dynamic path. Mooncake/ReverseDiff track each `exp`/`log`, but not the
# intermediate Vector{T}, which dominates AD cost on small groups.
mutable struct _LogSumExpAcc{T}
    m::T
    s::T
    nseen::Int
end
_LogSumExpAcc{T}() where {T} = _LogSumExpAcc{T}(T(-Inf), zero(T), 0)
function _push!(acc::_LogSumExpAcc{T}, x) where {T}
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
_value(acc::_LogSumExpAcc{T}) where {T} = acc.nseen == 0 ? T(-Inf) : acc.m + log(acc.s)

# The parameter float type the kernel contributes to the accumulator. In
# inference the fitted parameters ride the kernel (as AD duals), not the data,
# so the streaming accumulator must be typed to hold them: a distribution
# exposes its parameter type via `partype`, and a covariate callable is probed
# on the first internal pair. Falls back to `T` when there is no internal row.
function _kernel_partype(
        kernel::ContinuousUnivariateDistribution, layout, ::Type{T}) where {T}
    Distributions.partype(kernel)
end
function _kernel_partype(kernel, layout, ::Type{T}) where {T}
    for r in eachindex(layout.is_ext)
        layout.is_ext[r] && continue
        return Distributions.partype(_pair(kernel, layout.infector[r], layout.sus[r]))
    end
    return T
end

"""
    pairwise_surv_loglik(kernel, data::HouseholdInfections, layout::HouseholdPairsLayout;
                         external_hazard = 0.0) -> Real

Inference fast path: evaluate the contact-process log-density on the rows
captured in `layout`. The augmented `data.infection_time` /
`data.infectious_time` / `data.removal_time` are read on the fly, so the
caller can re-use the same `layout` across gradient evaluations.

Matches the dynamic-path result up to row order. `external` mode must be
consistent with `layout.external`; mismatching the two raises.
"""
function pairwise_surv_loglik(kernel, data::HouseholdInfections,
        layout::HouseholdPairsLayout;
        external_hazard = 0.0)
    external = _ext_active(external_hazard)
    external == layout.external ||
        throw(ArgumentError("layout.external = $(layout.external) but external_hazard = $external_hazard"))
    # The @inbounds passes index the time vectors by host id up to the population
    # the layout was compiled for; guard against a `data` with fewer individuals.
    min(length(data.infection_time), length(data.infectious_time),
        length(data.removal_time)) >= layout.nhosts ||
        throw(DimensionMismatch("data covers fewer individuals than the layout " *
                                "was compiled for ($(layout.nhosts))"))
    extdist = external ? _ext_survival(external_hazard) : kernel
    _shared_kernel = !external && kernel isa ContinuousUnivariateDistribution

    sus = layout.sus
    infector = layout.infector
    is_ext = layout.is_ext

    Tdata = promote_type(eltype(data.infection_time),
        eltype(data.infectious_time),
        eltype(data.removal_time),
        Float64)
    # promote against the kernel's parameter type so AD duals carried by the
    # fitted kernel (not the data) survive the reduction — the accumulator and
    # running total below are typed to hold whatever `loghazard`/`cumhazard`
    # returns, not just the data float type.
    Text = external ? Distributions.partype(extdist) : Union{}
    T = promote_type(Tdata, _kernel_partype(kernel, layout, Tdata), Text)
    ll = zero(T)

    # Pass 1: cumulative-hazard contribution per row (start always 0 by
    # construction of `_survival_rows`; we preserve that here).
    @inbounds for r in eachindex(sus)
        j = sus[r]
        tj = data.infection_time[j]
        infected_j = !isnan(tj)
        tend = infected_j ? tj : (external ? data.obs_end : T(Inf))
        if is_ext[r]
            stop = tend
            stop > 0 || continue
            ll -= cumhazard(extdist, stop)
        else
            i = infector[r]
            oi = data.infectious_time[i]
            isfinite(oi) || continue
            oi < tend || continue
            stop = min(data.removal_time[i], tend) - oi
            stop > 0 || continue
            kr = _shared_kernel ? kernel : _pair(kernel, i, j)
            ll -= cumhazard(kr, stop)
        end
    end

    # Pass 2: per-susceptible log-sum-exp over event rows. A single accumulator
    # is reused across groups (reset per group) so the reduction stays
    # allocation-free on the AD tape.
    acc = _LogSumExpAcc{T}()
    @inbounds for g in eachindex(layout.sus_unique)
        rng = layout.sus_row_ranges[g]
        acc.m = T(-Inf)
        acc.s = zero(T)
        acc.nseen = 0
        had_event = false
        for k in rng
            r = layout.sus_row_order[k]
            j = sus[r]
            tj = data.infection_time[j]
            infected_j = !isnan(tj)
            infected_j || continue
            if is_ext[r]
                stop = tj
                stop > 0 || continue
                _push!(acc, loghazard(extdist, stop))
                had_event = true
            else
                i = infector[r]
                oi = data.infectious_time[i]
                isfinite(oi) || continue
                if oi < tj && tj <= data.removal_time[i]
                    stop = tj - oi
                    kr = _shared_kernel ? kernel : _pair(kernel, i, j)
                    _push!(acc, loghazard(kr, stop))
                    had_event = true
                end
            end
        end
        had_event && (ll += _value(acc))
    end

    return ll
end
