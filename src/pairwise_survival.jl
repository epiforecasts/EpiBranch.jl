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
it.

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
progression and are conditioned on separately. There is no likelihood of the
onsets alone, since the latent infections cannot be marginalised in closed form.

A companion package reads a simulated outbreak back into its layer
(`household_infections`, `network_infections`). Each infected host's window
opens at the process's `from` state and closes at the earliest of its `until`
states and the time the model's interventions take the host out of transmission,
such as by isolation or quarantine after tracing. These are the windows the
simulation used, which makes scoring the layer under the kernel that simulated it
an exact `simulate → loglikelihood` round trip. Passing a reader the `followup_end`
keyword scores the outbreak as if observation had stopped at that time.
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

"""
    followup_end(data::InfectionLayer)

The end of follow-up of `data`, the time its observation stops. The pairwise
likelihood scores the infection layer up to it and ignores infections and
exposure after it. The default reads a `followup_end` field when the
[`InfectionLayer`](@ref) subtype has one, and is `Inf` otherwise; a subtype that
stores it elsewhere defines a method.
"""
function followup_end(data::InfectionLayer)
    hasproperty(data, :followup_end) ?
    data.followup_end : Inf
end

# The per-host fields of an `InfectionLayer` subtype over `n` hosts, in field
# order after the contact structure: the three time vectors, `is_index`,
# `obs_end` and `followup_end`. Every time shares one number type, at least
# `Float64`, which lets a constructor take integers or AD values.
function _infection_layer_fields(n, infection_time, infectious_time, removal_time,
        is_index; obs_end, followup_end)
    all(length(v) == n
    for v in (infection_time, infectious_time, removal_time, is_index)) ||
        throw(ArgumentError("the contact structure and the per-host vectors must " *
                            "cover the same hosts"))
    T = promote_type(eltype(infection_time), eltype(infectious_time),
        eltype(removal_time), typeof(obs_end), typeof(followup_end), Float64)
    return (Vector{T}(infection_time), Vector{T}(infectious_time),
        Vector{T}(removal_time), Vector{Bool}(is_index), T(obs_end), T(followup_end))
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
end

Base.length(L::ContactPairsLayout) = length(L.sus)

function _check_host_masks(n, is_index, infected)
    (length(is_index) == n && length(infected) == n) || throw(ArgumentError(
        "the contact structure, is_index and infected must cover the same hosts"))
    return nothing
end

# Group rows by susceptible for the per-susceptible log-sum-exp, and wrap up.
# Hosts that are explained (all of them with a community hazard, all but the
# index cases without one) and have no possible infector are listed apart, since
# an infection of one has zero density.
function _contact_pairs_layout(sus, infector, contact_index, is_ext, external,
        is_index, n)
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
    return ContactPairsLayout(sus, infector, contact_index, is_ext, sus_unique,
        sus_row_ranges, sus_row_order, no_rows, external, n)
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
        Int[], Int[], Int[], Bool[], external, is_index, 0)

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
    return _contact_pairs_layout(sus, infector, contact_index, is_ext, external,
        is_index, n)
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
    return _contact_pairs_layout(sus, infector, contact_index, is_ext, external,
        is_index, n)
end

function compile_contact_pairs(data::InfectionLayer; external::Bool = false)
    infected = .!isnan.(data.infection_time)
    return compile_contact_pairs(contact_structure(data), data.is_index, infected;
        external)
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

# ── Evaluation ───────────────────────────────────────────────────────

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
# intermediate vector for reverse-mode AD to track. A -Inf term (a zero hazard)
# adds nothing to the sum and is skipped. An accumulator that saw only zero
# hazards then gives -Inf without taking -Inf - (-Inf).
mutable struct _LogSumExpAcc{T}
    m::T
    s::T
    nseen::Int
end
_LogSumExpAcc{T}() where {T} = _LogSumExpAcc{T}(T(-Inf), zero(T), 0)
function _push!(acc::_LogSumExpAcc{T}, x) where {T}
    x == -Inf && return acc
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

# The parameter float type the kernel adds to the accumulator. In inference the
# fitted parameters are AD duals inside the kernel. The data's float type alone
# cannot hold them, and the streaming accumulator is typed to include them.
# A distribution gives its parameter type through `partype`; a per-edge or
# covariate kernel is probed on the first internal pair. With no internal pair
# the type falls back to `T`.
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
adds the log of the summed hazard at its infection time. An infected host that
is not conditioned on and has no positive hazard at its infection time, such as
one infected when none of its possible infectors is infectious, makes the whole
configuration impossible, and the density is `-Inf` with a zero gradient.

`kernel` is a `Distributions.jl` distribution shared by every pair, a callable
`(infector, susceptible) -> Distribution` for covariates, or a per-edge vector
parallel to an adjacency list (`kernel[i][k]` for host `i`'s `k`-th listed
contact). `external_hazard` is a community hazard (a positive rate or a
calendar-time distribution) that introduces cases over `[0, data.obs_end]`. With
one, index cases are explained like any other case; without one they are
conditioned on. Each host accrues the community hazard until the earlier of its
infection and `data.obs_end`. A host infected after `obs_end` can only have
been infected by a possible infector. Spread along the contact structure
continues after `obs_end`: a host that is never infected accrues hazard over
each possible infector's whole infectious window.

Everything is cut at [`followup_end(data)`](@ref EpiBranch.followup_end): a host
infected after it is scored as escaped until then, and no exposure accrues past
it. Scoring data with an end of follow-up gives the same value as first
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
    read off by letting `α` approach zero: score the two models separately.

    The discontinuity is at that one point. Approaching it, the log-density is
    `k log α − αT` up to terms free of `α`, where `k` counts the cases the
    community alone can explain and `T` is the total time hosts are exposed to
    it. In `log α` this is a straight line of slope `k`.
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

    tfollow = followup_end(data)
    (!isnan(tfollow) && tfollow >= 0) || throw(ArgumentError(
        "followup_end must be a non-negative number (Inf allowed), got $tfollow"))
    Tdata = promote_type(eltype(data.infection_time),
        eltype(data.infectious_time),
        eltype(data.removal_time),
        typeof(tfollow),
        Float64)
    # Promote against the kernel's parameter type so AD values in the fitted
    # kernel survive the reduction.
    Text = external ? Distributions.partype(extdist) : Union{}
    T = promote_type(Tdata, _kernel_partype(kernel, layout, Tdata), Text)
    # A per-edge or covariate kernel's parameter type is only known at run time;
    # the function barrier keeps the passes type-stable.
    return _pairwise_surv_loglik(kernel, extdist, data, layout,
        convert(Tdata, tfollow), T)
end

function _pairwise_surv_loglik(kernel, extdist, data, layout, tfollow,
        ::Type{T}) where {T}
    # An infected host that is not conditioned on and has no possible infector
    # cannot have been infected, unless that infection falls after the end of
    # follow-up.
    @inbounds for j in layout.no_rows
        tj = data.infection_time[j]
        (isnan(tj) || tj > tfollow) || return T(-Inf)
    end

    # A covariate or per-edge kernel may hold the fitted parameters on only some
    # pairs, and the probe behind `T` can miss them. Every row pass 2 scores has a
    # positive at-risk time in pass 1. Pass 1's sum has therefore seen every
    # kernel pass 2 will use, and its type sets pass 2's accumulator.
    ll = _pairwise_cumhazard(kernel, extdist, data, layout, tfollow, T)
    return _pairwise_events(kernel, extdist, data, layout, tfollow, ll,
        promote_type(T, typeof(ll)))
end

function _pairwise_cumhazard(kernel, extdist, data, layout, tfollow,
        ::Type{T}) where {T}
    sus = layout.sus
    infector = layout.infector
    is_ext = layout.is_ext
    ll = zero(T)

    # Pass 1: cumulative-hazard contribution per row, each at risk from 0. A
    # susceptible is exposed to its possible infectors until it is infected, and
    # to the community hazard until the earlier of that and `obs_end`, after
    # which there are no more introductions. Nothing is at risk after the end of
    # follow-up, and a host infected after it has escaped until then as far as
    # the data show.
    @inbounds for r in eachindex(sus)
        j = sus[r]
        tj = data.infection_time[j]
        tend = (isnan(tj) || tj > tfollow) ? tfollow : convert(typeof(tfollow), tj)
        if is_ext[r]
            stop = min(tend, data.obs_end)
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
    return ll
end

function _pairwise_events(kernel, extdist, data, layout, tfollow, ll0,
        ::Type{T}) where {T}
    sus = layout.sus
    infector = layout.infector
    is_ext = layout.is_ext
    ll = T(ll0)

    # Pass 2: per-susceptible log-sum-exp over event rows. A single accumulator
    # is reused across groups (reset per group) so the reduction stays
    # allocation-free on the AD tape. Every host in the layout is explained: an
    # infected one with no positive hazard at its infection time has density
    # zero, and the whole configuration is impossible: return -Inf there rather
    # than adding it, so that the derivative is zero too. Adding it would leave
    # the derivatives of the other hosts' finite terms sitting alongside an
    # infinite value, whereas the log-density is -Inf throughout a neighbourhood
    # of the parameters, because impossibility is a discrete fact of the fixed
    # times.
    acc = _LogSumExpAcc{T}()
    @inbounds for g in eachindex(layout.sus_unique)
        tj = data.infection_time[layout.sus_unique[g]]
        (isnan(tj) || tj > tfollow) && continue
        acc.m = T(-Inf)
        acc.s = zero(T)
        acc.nseen = 0
        for k in layout.sus_row_ranges[g]
            r = layout.sus_row_order[k]
            if is_ext[r]
                # a host infected after `obs_end` was infected along a contact;
                # one infected at 0 is a community case like any other
                (tj >= 0 && tj <= data.obs_end) || continue
                _push!(acc, loghazard(extdist, tj))
            else
                i = infector[r]
                oi = data.infectious_time[i]
                isfinite(oi) || continue
                if oi < tj && tj <= data.removal_time[i]
                    _push!(acc, loghazard(_pair_kernel(kernel, layout, r), tj - oi))
                end
            end
        end
        v = _value(acc)
        v == -Inf && return T(-Inf)
        ll += v
    end

    return ll
end

function pairwise_surv_loglik(kernel, data::InfectionLayer; external_hazard = 0.0)
    layout = compile_contact_pairs(data; external = _ext_active(external_hazard))
    return pairwise_surv_loglik(kernel, data, layout; external_hazard)
end
