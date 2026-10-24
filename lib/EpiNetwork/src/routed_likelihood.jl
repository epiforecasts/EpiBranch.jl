# ── RoutedNetwork pairwise survival likelihood ───────────────────────
#
# A `RoutedNetwork` outbreak is still one Sellke race, but with several
# competing routes open over each case rather than one: the race proposes a
# contact along every route at once and the earliest proposal wins, so each
# susceptible faces the *sum* of every route's hazard from every infector that
# can reach it. The likelihood below scores that exactly: survival sums each
# route's cumulative hazard (independent competing risks), and the event term
# at an infected person's own infection time is a log-sum-exp over every
# route and infector that could have produced it, exactly as `logsumexp` over
# `data.contacts` terms would be for a single route. A route not listing
# `EpiBranch.INTERVENTION_REMOVAL` keeps exposing its susceptibles through a
# stretch that removes the infector on another route, which is the entire
# point of separating them.

"""
    RoutedNetworkInfections(route_names, contacts, infectious_time, removal_time,
                            infection_time, is_index; obs_end = Inf,
                            followup_end = Inf, host_times = (;))

Routed-network outbreak data for the pairwise likelihood: an infection record
per person, with a separate infectious window and contact list for each
route, in the order `route_names` gives. `contacts[r][i]` lists the people `i`
can infect on route `r`, as [`RouteWindow`](@ref)'s `reach`; `infectious_time[r]`
and `removal_time[r]` are when route `r` opened and closed for each person,
the same windows [`RoutedNetwork`](@ref)'s race used. The remaining arguments
are as for [`NetworkInfections`](@ref): `infection_time`, `is_index`,
`obs_end`, `followup_end` and `host_times` are shared across routes, since
they describe the person rather than any one route. Read one from a
simulation with [`routed_network_infections`](@ref).
"""
struct RoutedNetworkInfections{T <: Real, H <: NamedTuple} <: InfectionLayer
    route_names::Vector{Symbol}
    contacts::Vector{Vector{Vector{Int}}}
    infectious_time::Vector{Vector{T}}
    removal_time::Vector{Vector{T}}
    infection_time::Vector{T}
    is_index::Vector{Bool}
    obs_end::T
    followup_end::T
    host_times::H
end

function RoutedNetworkInfections(
        route_names::AbstractVector{Symbol},
        contacts::AbstractVector{<:AbstractVector{<:AbstractVector{<:Integer}}},
        infectious_time::AbstractVector{<:AbstractVector{<:Real}},
        removal_time::AbstractVector{<:AbstractVector{<:Real}},
        infection_time::AbstractVector{<:Real}, is_index::AbstractVector{Bool};
        obs_end = Inf, followup_end = Inf, host_times = (;)
    )
    n_routes = length(route_names)
    (
        length(contacts) == n_routes && length(infectious_time) == n_routes &&
            length(removal_time) == n_routes
    ) || throw(
        ArgumentError(
            "route_names, contacts, infectious_time and removal_time must each " *
                "have one entry per route"
        )
    )
    n = length(infection_time)
    length(is_index) == n || throw(
        ArgumentError("infection_time and is_index must cover the same hosts")
    )
    for r in 1:n_routes
        (
            length(contacts[r]) == n && length(infectious_time[r]) == n &&
                length(removal_time[r]) == n
        ) || throw(
            ArgumentError(
                "route :$(route_names[r]) must cover the same $n hosts as infection_time"
            )
        )
        for (i, nb) in enumerate(contacts[r]), j in nb
            1 <= j <= n || throw(
                ArgumentError(
                    "route :$(route_names[r]) host $i lists contact $j, outside the " *
                        "$n hosts in the structure"
                )
            )
        end
    end
    host_times isa NamedTuple ||
        throw(ArgumentError("host_times must be a named tuple of per-host vectors"))
    all(length(v) == n for v in values(host_times)) || throw(
        ArgumentError("host_times columns must cover the same $n hosts as infection_time")
    )

    T = promote_type(
        eltype(infection_time), typeof(obs_end), typeof(followup_end), Float64,
        (eltype(v) for v in infectious_time)..., (eltype(v) for v in removal_time)...
    )
    adj = [Vector{Int}[Int.(nb) for nb in route] for route in contacts]
    return RoutedNetworkInfections{T, typeof(host_times)}(
        Vector{Symbol}(route_names), adj,
        [Vector{T}(v) for v in infectious_time], [Vector{T}(v) for v in removal_time],
        Vector{T}(infection_time), Vector{Bool}(is_index),
        T(obs_end), T(followup_end), host_times
    )
end

Base.length(d::RoutedNetworkInfections) = length(d.infection_time)

# There is no one contact structure to hand `compile_contact_pairs`: each
# route has its own, and they are combined inside `pairwise_surv_loglik`
# rather than compiled into a shared layout.
function EpiBranch.contact_structure(data::RoutedNetworkInfections)
    throw(
        ArgumentError(
            "RoutedNetworkInfections has a separate contact structure per route; " *
                "evaluate it with `pairwise_surv_loglik(windows, data)` or " *
                "`loglikelihood(data, model::RoutedNetwork)` rather than " *
                "`compile_contact_pairs`"
        )
    )
end

"""
    routed_network_infections(state, model::ModelSpec{<:RoutedNetwork};
                              obs_end = model.process.obs_end, followup_end = Inf)
        -> RoutedNetworkInfections
    routed_network_infections(state, process::RoutedNetwork) -> RoutedNetworkInfections

Collect who was infected when from a routed-network outbreak simulated from
`model`, with every route's own infectious window read back exactly as the
race closed it, including a lapsing removal (such as an `Isolation` with a
finite `duration`) stored in the record's `host_times` the same way
[`network_infections`](@ref) stores it, for
[`pairwise_surv_loglik`](@ref) to read back out on whichever routes list
[`EpiBranch.INTERVENTION_REMOVAL`](@ref).
"""
function routed_network_infections(
        state::SimulationState,
        model::ModelSpec{<:RoutedNetwork}; obs_end = model.process.obs_end,
        followup_end = Inf
    )
    process = model.process
    n = process.n
    length(state.individuals) == n || throw(
        ArgumentError(
            "the state has $(length(state.individuals)) individuals but the routed " *
                "network has $n nodes"
        )
    )
    derived = something(process.from, infectious_from(model.progression))
    windows = [_start_unset(w, derived) for w in process.windows]
    n_routes = length(windows)

    infection_time = fill(NaN, n)
    is_index = falses(n)
    infectious_time = [fill(NaN, n) for _ in 1:n_routes]
    removal_time = [fill(Inf, n) for _ in 1:n_routes]
    for (k, ind) in enumerate(state.individuals)
        get(ind.state, :infected, false) || continue
        infection_time[k] = ind.infection_time
        is_index[k] = get(ind.state, :index, false)
        for (r, w) in enumerate(windows)
            infectious_time[r][k] = window_open(ind, w)
            removal_time[r][k] = window_close(ind, w, model.interventions)
        end
    end

    contacts = [Vector{Int}[Int.(nb) for nb in w.reach] for w in windows]
    route_names = Symbol[w.name for w in windows]
    host_times = _layer_host_times(state, model, ())
    return RoutedNetworkInfections(
        route_names, contacts, infectious_time, removal_time,
        infection_time, is_index; obs_end, followup_end, host_times
    )
end

function routed_network_infections(state::SimulationState, process::RoutedNetwork; kwargs...)
    return routed_network_infections(state, ModelSpec(process); kwargs...)
end

# ── Evaluation ───────────────────────────────────────────────────────

# A route's contact-interval distribution for infector `i`'s `pos`-th listed
# contact, `j`: as `_resolve_kernel` resolves it for `NetworkProcess`, minus
# the `PairKernel` form, which needs each person's record projected the way
# the single-route likelihood projects it and is not yet wired up here.
_route_kernel(k::ContinuousUnivariateDistribution, contacts, i, pos, j) = k
_route_kernel(k::AbstractVector{<:AbstractVector}, contacts, i, pos, j) = k[i][pos]
function _route_kernel(k::PairKernel, contacts, i, pos, j)
    throw(
        ArgumentError(
            "the routed-network likelihood does not yet resolve a PairKernel route " *
                "kernel; fit this route with a shared distribution, a per-edge vector, " *
                "or a covariate function (infector, susceptible) -> Distribution"
        )
    )
end
_route_kernel(k, contacts, i, pos, j) = k(i, j)

# A route's own parameter type, found the same way `_kernel_partype` finds it
# for the single-route likelihood: a shared distribution exposes it directly
# through `partype`; a per-edge or covariate kernel is probed on an actual
# edge of that route. With no edge to probe, the route contributes nothing of
# its own and `T` falls back to whatever the rest of the likelihood needs.
_route_kernel_partype(k::ContinuousUnivariateDistribution, contacts, in_neighbours) =
    Distributions.partype(k)
function _route_kernel_partype(k, contacts, in_neighbours)
    for (j, row) in enumerate(in_neighbours), (i, pos) in row
        return Distributions.partype(_route_kernel(k, contacts, i, pos, j))
    end
    return Union{}
end

# Route `r`'s possible infectors of each susceptible: for every infected host
# `i`, the (host, position-in-`reach[i]`) pairs of its listed contacts, so a
# susceptible's row only ever enumerates infected hosts that can reach it.
function _route_in_neighbours(contacts::Vector{Vector{Int}}, infected::AbstractVector{Bool})
    n = length(contacts)
    rev = [Tuple{Int, Int}[] for _ in 1:n]
    for i in 1:n
        infected[i] || continue
        for (pos, j) in enumerate(contacts[i])
            j == i && continue
            push!(rev[j], (i, pos))
        end
    end
    return rev
end

# A susceptible's escape and event terms under one mixture component's
# modifier, over every route and the external hazard: the same sums the
# no-susceptibility loop in `pairwise_surv_loglik` below works out, with
# `_scaled_cumhazard` and `_scaled_loghazard` applying the modifier to each
# route's and the external hazard's own hazard.
function _routed_modifier_loglik(
        modifier, windows, extsrc, data, in_neighbours, gaps, gates, tfollow, j
    )
    n_routes = length(windows)
    tj = data.infection_time[j]
    tend = (isnan(tj) || tj > tfollow) ? tfollow : tj

    survival = 0.0
    extdist = _ext_survival_of(extsrc, j)
    if extdist !== nothing
        stop = min(tend, data.obs_end)
        stop > 0 && (survival += _scaled_cumhazard(modifier, extdist, zero(stop), stop))
    end
    for r in 1:n_routes
        for (i, pos) in in_neighbours[r][j]
            oi = data.infectious_time[r][i]
            isfinite(oi) || continue
            oi < tend || continue
            stop = min(data.removal_time[r][i], tend) - oi
            stop > 0 || continue
            pk = _route_kernel(windows[r].kernel, data.contacts[r], i, pos, j)
            survival += gates[r] ?
                _gapped_cumhazard(
                    t -> _scaled_cumhazard(modifier, pk, oi, t), _host_stretches(gaps, i), oi, stop
                ) :
                _scaled_cumhazard(modifier, pk, oi, stop)
        end
    end
    ll = -survival

    (isnan(tj) || tj > tfollow) && return ll
    route_terms = [
        _scaled_loghazard(
            modifier,
            loghazard(
                _route_kernel(windows[r].kernel, data.contacts[r], i, pos, j),
                tj - data.infectious_time[r][i]
            ),
            tj
        )
            for r in 1:n_routes for (i, pos) in in_neighbours[r][j]
            if isfinite(data.infectious_time[r][i]) &&
            data.infectious_time[r][i] < tj <= data.removal_time[r][i] &&
            (!gates[r] || !_removed_at(gaps, i, tj))
    ]
    ext_term = (extdist !== nothing && 0 <= tj <= data.obs_end) ?
        [_scaled_loghazard(modifier, loghazard(extdist, tj), tj)] : Float64[]
    terms = vcat(route_terms, ext_term)
    isempty(terms) && return -Inf
    return ll + logsumexp(terms)
end

# Every susceptible's mixture, by host id, or `nothing` when `effect` modifies
# none, as `_host_mixtures` finds for a household or network's layout.
_routed_host_mixtures(::Nothing, data, n) = nothing
function _routed_host_mixtures(effect, data, n)
    mixtures = Dict{Int, Any}()
    for j in 1:n
        mixture = susceptibility_components(effect, _layer_host(data, j))
        mixture === nothing || (mixtures[j] = mixture)
    end
    return isempty(mixtures) ? nothing : mixtures
end

# Host `j`'s contribution as the weighted mixture of `_routed_modifier_loglik`
# under each of `mixture`'s components, combined by the same weighted
# log-sum-exp `_mixture_loglik` uses for a household or network: the weights
# enter linearly, never through `log(weight)`, so a weight of exactly zero
# still has a derivative with respect to itself.
function _routed_mixture_loglik(
        mixture, windows, extsrc, data, in_neighbours, gaps, gates, tfollow, j, ::Type{T}
    ) where {T}
    terms = map(mixture) do (weight, modifier)
        weight => _routed_modifier_loglik(
            modifier, windows, extsrc, data, in_neighbours, gaps, gates, tfollow, j
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

"""
    pairwise_surv_loglik(windows, data::RoutedNetworkInfections; external_hazard = 0.0,
                         susceptibility = nothing) -> Real

Log-likelihood of a routed-network outbreak under `windows`, a collection of
[`RouteWindow`](@ref)s in the same order as `data`'s routes, giving each
route's kernel and which states (including
[`EpiBranch.INTERVENTION_REMOVAL`](@ref)) end it.

Each susceptible's survival is the sum of every route's cumulative hazard
from every possible infector on it (routes are independent competing risks),
and an infected person's event term is the log of the summed hazard, over
every route and every possible infector, at their infection time — the
marginal density of whichever route and infector's draw came first in the
race. A route whose `until` does not list `EpiBranch.INTERVENTION_REMOVAL`
keeps exposing its susceptibles through a stretch that an intervention such as
`Isolation` or `Quarantine` removed the infector for on another route; one
that does list it has that stretch, read from `data`'s `host_times` as the
single-route likelihood reads it, taken out of its own exposure.

`external_hazard` is community infection shared across every route, as for
[`NetworkProcess`](@ref)'s likelihood.

`susceptibility` is an effect on people's own risk of infection, such as a
candidate [`VaccineEffect`](@ref) or a model's interventions, exactly as for
[`NetworkProcess`](@ref)'s likelihood: through
[`susceptibility_components`](@ref EpiBranch.susceptibility_components) the
effect says how it scales every hazard a person faces, from every route and
the external hazard alike, and a susceptible it modifies is scored as the
weighted mixture of their escape and infection under each of the effect's
components. A susceptible it leaves unmodified keeps the plain sum above.

# Example

```julia
using EpiNetwork, EpiBranch, Distributions
household = [[2, 3], [1, 3], [1, 2], Int[]]
community = [[4], Int[], Int[], [1]]
windows = [
    RouteWindow(:household; until = (:recovered,), kernel = Exponential(4.0), reach = household),
    RouteWindow(:community; until = (:recovered,), kernel = Exponential(20.0), reach = community),
]
progression = [Transition(:recovered; from = :infection, delay = 6.0, terminal = true)]
state = simulate(ModelSpec(RoutedNetwork(windows); progression); n_initial = 2)
data = routed_network_infections(state, RoutedNetwork(windows))
pairwise_surv_loglik(windows, data)
```
"""
function EpiBranch.pairwise_surv_loglik(
        windows::AbstractVector{<:RouteWindow}, data::RoutedNetworkInfections;
        external_hazard = 0.0, susceptibility = nothing
    )
    n_routes = length(data.contacts)
    length(windows) == n_routes || throw(
        ArgumentError(
            "windows must have one entry per route ($n_routes), got $(length(windows))"
        )
    )
    Symbol[w.name for w in windows] == data.route_names || throw(
        ArgumentError("windows must be given in the same order as data's routes")
    )

    n = length(data)
    external = _ext_active(external_hazard)
    external && _check_external(external_hazard, n)
    extsrc = external ? _ext_prepare(external_hazard) : nothing
    tfollow = followup_end(data)
    (!isnan(tfollow) && tfollow >= 0) || throw(
        ArgumentError("followup_end must be a non-negative number (Inf allowed), got $tfollow")
    )
    gaps = _removal_gaps(data)
    gates = Bool[INTERVENTION_REMOVAL in w.until for w in windows]
    infected = .!isnan.(data.infection_time)
    in_neighbours = [_route_in_neighbours(data.contacts[r], infected) for r in 1:n_routes]
    mixtures = _routed_host_mixtures(susceptibility, data, n)
    Twindows = promote_type(
        (
            _route_kernel_partype(windows[r].kernel, data.contacts[r], in_neighbours[r])
                for r in 1:n_routes
        )...
    )
    Text = external ? _ext_partype(extsrc, n) : Union{}
    T = promote_type(Float64, Twindows, _mixtures_partype(mixtures), Text)

    ll = zero(T)
    for j in 1:n
        (!external && data.is_index[j]) && continue
        mixture = mixtures === nothing ? nothing : get(mixtures, j, nothing)
        if mixture !== nothing
            v = _routed_mixture_loglik(
                mixture, windows, extsrc, data, in_neighbours, gaps, gates, tfollow, j, T
            )
            # Returning `-Inf` directly, rather than adding it to `ll`, discards
            # whatever AD partials `ll` has accumulated from this host and every
            # earlier one. Once a host is structurally unexplainable the
            # likelihood is the constant -Inf at every parameter value, so its
            # gradient must be exactly zero, and those partials would otherwise
            # survive into a nonzero one. The flat passes below return -Inf the
            # same way.
            _is_minus_inf(v) && return T(-Inf)
            ll += v
            continue
        end
        tj = data.infection_time[j]
        tend = (isnan(tj) || tj > tfollow) ? tfollow : tj

        survival = 0.0
        extdist = _ext_survival_of(extsrc, j)
        if extdist !== nothing
            stop = min(tend, data.obs_end)
            stop > 0 && (survival += cumhazard(extdist, stop))
        end
        for r in 1:n_routes
            for (i, pos) in in_neighbours[r][j]
                oi = data.infectious_time[r][i]
                isfinite(oi) || continue
                oi < tend || continue
                stop = min(data.removal_time[r][i], tend) - oi
                stop > 0 || continue
                pk = _route_kernel(windows[r].kernel, data.contacts[r], i, pos, j)
                survival += gates[r] ?
                    _gapped_cumhazard(t -> cumhazard(pk, t), _host_stretches(gaps, i), oi, stop) :
                    cumhazard(pk, stop)
            end
        end
        ll -= survival

        (isnan(tj) || tj > tfollow) && continue
        route_terms = [
            loghazard(
                _route_kernel(windows[r].kernel, data.contacts[r], i, pos, j), tj - data.infectious_time[r][i]
            )
                for r in 1:n_routes for (i, pos) in in_neighbours[r][j]
                if isfinite(data.infectious_time[r][i]) &&
                data.infectious_time[r][i] < tj <= data.removal_time[r][i] &&
                (!gates[r] || !_removed_at(gaps, i, tj))
        ]
        ext_term = (extdist !== nothing && 0 <= tj <= data.obs_end) ?
            [loghazard(extdist, tj)] : Float64[]
        terms = vcat(route_terms, ext_term)
        isempty(terms) && return -Inf
        ll += logsumexp(terms)
    end
    return ll
end

"""
    loglikelihood(data::RoutedNetworkInfections, model::RoutedNetwork;
                  susceptibility = nothing) -> Real
    loglikelihood(data::RoutedNetworkInfections, model::ModelSpec{<:RoutedNetwork}) -> Real

Log-likelihood of routed-network outbreak data under `model`'s route windows
and community hazard: `pairwise_surv_loglik(model.windows, data; external_hazard
= model.external_hazard, susceptibility)`. For a `ModelSpec`, every intervention
must declare [`infection_likelihood_compatible`](@ref
EpiBranch.infection_likelihood_compatible), and `susceptibility` is the
model's interventions, so a vaccination in the model is evaluated from the
immunity times [`routed_network_infections`](@ref) recorded.
"""
function Distributions.loglikelihood(
        data::RoutedNetworkInfections, model::RoutedNetwork;
        susceptibility = nothing
    )
    return pairwise_surv_loglik(
        model.windows, data; external_hazard = model.external_hazard, susceptibility
    )
end

function Distributions.loglikelihood(
        data::RoutedNetworkInfections,
        model::ModelSpec{<:RoutedNetwork}
    )
    _validate_infection_likelihood(model)
    return loglikelihood(data, model.process; susceptibility = model.interventions)
end
