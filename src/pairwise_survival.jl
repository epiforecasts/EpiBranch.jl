# ── Pairwise survival likelihood over a contact structure ────────────
#
# The contact-process density: the log-likelihood of an outbreak's *infection
# layer* (who is infected, and the latent infection time and infectious window
# of each) under a contact-interval kernel. It is the marginal pairwise
# likelihood of Kenah (2011): who-infected-whom and the order of infections are
# both unobserved, so each infected susceptible's contribution sums the
# contact-interval hazard over every possible infector, with no ordering assumed.
#
# Nothing in it depends on what the contact structure is. It is a product over
# (susceptible, possible infector) pairs, so a household partition, a contact
# network and any other "who could have infected whom" relation share it; only
# the enumeration of those pairs differs.
#
# This scores the infection layer, which is latent: observed in a `simulate`
# round trip, augmented in inference (where the progression then links each
# infection to its observed onset or test). It never takes onsets as the event.

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

# ── The infection layer ──────────────────────────────────────────────

"""
    InfectionLayer

Supertype of an outbreak's infection layer paired with the contact structure it
spread over — the data the pairwise likelihood is a density of. A subtype holds,
per host `i` (numbered `1:n`):

- `infection_time[i]` — the infection time, `NaN` if never infected;
- `infectious_time[i]` — when the infectious window opens;
- `removal_time[i]` — when it closes (`Inf` when right-censored);
- `is_index[i]` — whether the host was introduced from outside the structure;

and a scalar `obs_end`, the end of follow-up over which a community hazard acts
(only read when there is one). It also defines
[`contact_structure`](@ref EpiBranch.contact_structure), which names who could
have infected whom. With those in place,
[`compile_contact_pairs`](@ref) and [`pairwise_surv_loglik`](@ref) work on it
unchanged. `HouseholdInfections` (in `EpiHouseholds`) and `NetworkInfections`
(in `EpiNetwork`) are the worked examples.
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
    throw(ArgumentError("$(nameof(typeof(data))) needs a method for " *
                        "`EpiBranch.contact_structure` naming who could have infected whom"))
end

# ── Compiled pair layout ─────────────────────────────────────────────
#
# In inference the contact structure, the index cases and the set of
# ever-infected hosts are fixed across gradient evaluations; only the augmented
# infection and infectious times move. `ContactPairsLayout` captures everything
# that does not depend on those times — row → (susceptible, infector, the edge
# it travels along) and a susceptible-grouped index for the log-sum-exp — so an
# evaluation is a single pass over rows with no `Dict` and a streaming
# log-sum-exp, which keeps a reverse-mode AD tape short.

"""
    ContactPairsLayout

The static row structure the pairwise likelihood is evaluated on. Each row is
one ordered (susceptible, possible infector) pair, plus, when a community hazard
is modelled, one row per susceptible for that external source. Rows whose timing
turns out not to overlap are kept and pruned at evaluation, so one layout serves
every configuration of latent times with the same structure and infected set.

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
    external::Bool
    nhosts::Int                            # population the layout was compiled for
end

Base.length(L::ContactPairsLayout) = length(L.sus)

function _check_host_masks(n, is_index, infected)
    (length(is_index) == n && length(infected) == n) || throw(ArgumentError(
        "the contact structure, is_index and infected must cover the same hosts"))
    return nothing
end

# Group rows by susceptible for the per-susceptible log-sum-exp, and wrap up.
function _contact_pairs_layout(sus, infector, contact_index, is_ext, external, n)
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
    return ContactPairsLayout(sus, infector, contact_index, is_ext, sus_unique,
        sus_row_ranges, sus_row_order, external, n)
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
configuration the layout will score (its infection time may still be augmented).
Only infected hosts can be infectors. `is_index` marks hosts introduced from
outside; without a community hazard they are conditioned on and appear only as
infectors. With `external = true` every host is explained, and each gets an
extra row for the community hazard. The single-argument form reads the structure
off `data` with [`contact_structure`](@ref EpiBranch.contact_structure) and the
mask as `.!isnan.(data.infection_time)`.
"""
function compile_contact_pairs(membership::AbstractVector{<:Integer},
        is_index::AbstractVector{Bool}, infected::AbstractVector{Bool};
        external::Bool = false)
    n = length(membership)
    _check_host_masks(n, is_index, infected)
    isempty(membership) && return _contact_pairs_layout(
        Int[], Int[], Int[], Bool[], external, 0)

    # Bucket hosts by label into a `Vector{Vector{Int}}` indexed by label offset
    # (no hashing); offsetting by `lo` tolerates any integer labels.
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
    return _contact_pairs_layout(sus, infector, contact_index, is_ext, external, n)
end

function compile_contact_pairs(contacts::AbstractVector{<:AbstractVector{<:Integer}},
        is_index::AbstractVector{Bool}, infected::AbstractVector{Bool};
        external::Bool = false)
    n = length(contacts)
    _check_host_masks(n, is_index, infected)

    # Invert the out-lists of infected hosts into in-lists (compressed rows), so
    # each susceptible's possible infectors come out together, in infector order.
    indeg = zeros(Int, n + 1)
    for i in 1:n
        infected[i] || continue
        for j in contacts[i]
            1 <= j <= n || throw(ArgumentError(
                "host $i lists contact $j, outside the $n hosts in the structure"))
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
    return _contact_pairs_layout(sus, infector, contact_index, is_ext, external, n)
end

function compile_contact_pairs(data::InfectionLayer; external::Bool = false)
    infected = .!isnan.(data.infection_time)
    return compile_contact_pairs(contact_structure(data), data.is_index, infected;
        external)
end

# ── Evaluation ───────────────────────────────────────────────────────

# The community hazard is off at a zero rate; a distribution is always on.
_ext_active(α::Real) = α > 0
_ext_active(::ContinuousUnivariateDistribution) = true

# The community hazard as a calendar-time survival distribution: a constant rate
# α is `Exponential(1/α)` (hazard α, cumulative α·t); a distribution is itself.
_ext_survival(α::Real) = Exponential(1 / α)
_ext_survival(d::ContinuousUnivariateDistribution) = d

# Row r's contact-interval distribution: a shared distribution, a per-edge
# vector parallel to the adjacency the layout was compiled from, or a callable
# `(infector, susceptible) -> Distribution` for covariates.
_pair_kernel(k::ContinuousUnivariateDistribution, layout, r) = k
function _pair_kernel(k::AbstractVector{<:AbstractVector}, layout, r)
    c = layout.contact_index[r]
    c > 0 || throw(ArgumentError(
        "a per-edge kernel needs a layout compiled from an adjacency list"))
    return k[layout.infector[r]][c]
end
_pair_kernel(k, layout, r) = k(layout.infector[r], layout.sus[r])

# Streaming logsumexp, so the per-susceptible reduction allocates no
# intermediate vector for reverse-mode AD to track.
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
# exposes its parameter type via `partype`, and a per-edge or covariate kernel
# is probed on the first internal pair. Falls back to `T` when there is none.
function _kernel_partype(
        kernel::ContinuousUnivariateDistribution, layout, ::Type{T}) where {T}
    Distributions.partype(kernel)
end
function _kernel_partype(kernel, layout, ::Type{T}) where {T}
    for r in eachindex(layout.is_ext)
        layout.is_ext[r] && continue
        return Distributions.partype(_pair_kernel(kernel, layout, r))
    end
    return T
end

"""
    pairwise_surv_loglik(kernel, data::InfectionLayer, layout::ContactPairsLayout;
                         external_hazard = 0.0) -> Real
    pairwise_surv_loglik(kernel, data::InfectionLayer; external_hazard = 0.0) -> Real

The contact-process log-density of the infection layer `data` under a
contact-interval `kernel`, marginal over who infected whom. Each susceptible
accrues cumulative hazard from every possible infector over the overlap of that
infector's infectious window with its own time at risk, and each infected one
adds the log of the summed hazard at its infection time.

`kernel` is a `Distributions.jl` distribution shared by every pair, a callable
`(infector, susceptible) -> Distribution` for covariates, or a per-edge vector
parallel to an adjacency list (`kernel[i][k]` for host `i`'s `k`-th listed
contact). `external_hazard` is a community hazard — a positive rate, or a
calendar-time distribution — acting over `[0, data.obs_end]`; with one, index
cases are explained like any other case, and without one they are conditioned
on.

The layout form is the inference fast path: compile the layout once with
[`compile_contact_pairs`](@ref) and reuse it while the latent times move; its
`external` setting must agree with `external_hazard`. The two-argument form
compiles a layout on each call. Both are generic in the number type, so the
kernel's parameters can carry ForwardDiff or reverse-mode AD values.
"""
function pairwise_surv_loglik(kernel, data::InfectionLayer, layout::ContactPairsLayout;
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

    Tdata = promote_type(eltype(data.infection_time),
        eltype(data.infectious_time),
        eltype(data.removal_time),
        Float64)
    # Promote against the kernel's parameter type so AD values carried by the
    # fitted kernel (rather than the data) survive the reduction.
    Text = external ? Distributions.partype(extdist) : Union{}
    T = promote_type(Tdata, _kernel_partype(kernel, layout, Tdata), Text)
    # A per-edge or covariate kernel's parameter type is only known at run time,
    # so pass it through a function barrier to keep the passes type-stable.
    return _pairwise_surv_loglik(kernel, extdist, data, layout, external, T)
end

function _pairwise_surv_loglik(kernel, extdist, data, layout, external,
        ::Type{T}) where {T}
    sus = layout.sus
    infector = layout.infector
    is_ext = layout.is_ext
    ll = zero(T)

    # Pass 1: cumulative-hazard contribution per row, each at risk from 0.
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
            ll -= cumhazard(_pair_kernel(kernel, layout, r), stop)
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
                    _push!(acc, loghazard(_pair_kernel(kernel, layout, r), stop))
                    had_event = true
                end
            end
        end
        had_event && (ll += _value(acc))
    end

    return ll
end

function pairwise_surv_loglik(kernel, data::InfectionLayer; external_hazard = 0.0)
    layout = compile_contact_pairs(data; external = _ext_active(external_hazard))
    return pairwise_surv_loglik(kernel, data, layout; external_hazard)
end
