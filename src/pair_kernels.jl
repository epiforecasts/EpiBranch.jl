"""
    PairContext(infector, susceptible, infector_infection_time)

Information available to a pair kernel in both simulation and an infection-layer
likelihood. `infector` and `susceptible` are population IDs. The infector's infection
time retains its number type, including automatic-differentiation values.

The susceptible's eventual infection time is excluded: it is unknown when
simulation selects the contact-interval distribution. Fixed host covariates can
be indexed by either ID in tables captured by the kernel callback.
"""
struct PairContext{T <: Real}
    infector::Int
    susceptible::Int
    infector_infection_time::T
end

"""
    ContextualKernel(callback)

A pair kernel whose `callback(context::PairContext)` returns a contact-interval
distribution. Supported by `NetworkProcess`, `HouseholdProcess`, and the
`InfectionLayer` forms of [`pairwise_surv_loglik`](@ref).

```julia
kernel = ContextualKernel(context ->
    Exponential(exp(0.1 * context.infector_infection_time)))
```

The callback must be deterministic and describe the same distribution for a pair
throughout the infector's infectious window. It can use fixed covariates and the
infector's infection time. It does not receive live intervention state or the
current clock. Wrap it in `CalendarKernel` when the returned distribution
describes a calendar-time hazard. Unwrapped callables retain their `(infector_id, susceptible_id)` signature.
"""
struct ContextualKernel{F}
    callback::F
end

"""
    pair_kernel(kernel, infector, susceptible, infector_infection_time)
    pair_kernel(kernel, infector, susceptible, infector_infection_time, infectious_time)
    pair_kernel(kernel, infector, susceptible, infector_infection_time, infectious_time,
        state)

Resolve a contact-interval distribution for an ordered pair. A shared continuous
distribution is returned unchanged; an ordinary callable receives the two IDs;
a [`ContextualKernel`](@ref) receives a [`PairContext`](@ref). Structured-process
extensions can use this method to share kernel semantics with the likelihood.
The five-argument form supplies the infectious opening required by `CalendarKernel`.
The six-argument form also passes the `SimulationState`, from which a live
[`StatefulKernel`](@ref) reads both hosts' records; simulation must use it, since
a live kernel has no method for the shorter forms, which are for likelihoods
evaluated against a [`RecordedKernel`](@ref). Every other kernel returns what the
five-argument form does.
"""
pair_kernel(k::ContinuousUnivariateDistribution, i, j, infection_time) = k
pair_kernel(k, i, j, infection_time) = k(i, j)
function pair_kernel(k::ContextualKernel, i, j, infection_time)
    return k.callback(PairContext(i, j, infection_time))
end

"""
    CalendarKernel(kernel)

Interpret a shared distribution, pair callback, `ContextualKernel`, `StatefulKernel`
or `RecordedKernel` on the
simulation's calendar-time axis. In network and household models, condition the
returned distribution on surviving to the infector's infectious opening, then
subtract that opening to obtain a contact interval. Network per-edge distribution
vectors are supported too.

The distribution's hazard at calendar date `t` is the pair's contact rate at `t`.
The same conditioning is used by infection-layer likelihoods, including when
infectious opening times change during inference. The calendar law must have
positive survival at each opening. Supply parameters and covariates consistently to simulation and inference.
A live `StatefulKernel` may update its hazard through dated histories; other
kernels remain fixed throughout a simulation.
"""
struct CalendarKernel{K}
    kernel::K
end

# The five-argument interface also supplies the origin of the contact interval.
pair_kernel(k, i, j, infection_time, infectious_time) = pair_kernel(k, i, j, infection_time)
function pair_kernel(k::CalendarKernel, i, j, infection_time, infectious_time)
    return _calendar_interval(pair_kernel(k.kernel, i, j, infection_time), infectious_time)
end
function pair_kernel(::CalendarKernel, i, j, infection_time)
    throw(ArgumentError("CalendarKernel also requires the infectious opening time"))
end
function _calendar_interval(distribution, opening)
    return truncated(distribution; lower = opening) - opening
end

"""
    StatefulKernel(projection, callback)

A pair kernel whose hazard can change during simulation. `projection(individual)`
selects an immutable record (for example a named tuple of attributes and event
dates) from a live `Individual`. `callback(context::PairContext, source, target)`
returns the contact-interval law from those records. Wrap in [`CalendarKernel`](@ref)
for calendar-time laws.

A projection can also serve the likelihood directly when the infection layer
holds the per-host times it reads, such as an onset time an infector's
infectiousness is timed from. The likelihood then applies the projection to
each host as a [`LayerHost`](@ref), which has the `id` and `infection_time` of
an individual and a `state` holding the layer's `host_times` under their keys.
A projection that reads `ind.id`, `ind.infection_time` and the recorded times
through `ind.state` works for both a live `Individual` and a `LayerHost`, so the
same kernel scores an augmented infection layer, whose onsets change during
inference. Reading a key the layer did not record raises an error; record every
time the projection reads. For a likelihood evaluated against an ordinary vector
of records, build a [`RecordedKernel`](@ref) instead, or use [`record_kernel`](@ref)
to extract one after simulation.

Callbacks must describe a predictable hazard: adding an event at time `t` must
not change the hazard before `t`. A final vaccinated flag alone is insufficient;
retain its date and the earlier hazard. State changes occur through the existing
case-resolution and intervention hooks. The callback and projection must not
mutate state, and a kernel may read host state only through its projection,
since that record is all the likelihood is given.

Simulation keeps contacts consistent with the hazards in force as records
change, and a run whose records never change follows the same distribution as
an ordinary kernel. A race that can read several households or several routes at
once (see [`watched_records`](@ref)) races them together so that a policy can
read cases outside the one it is resolving, which draws the same outbreak from a
different random stream.
"""
struct StatefulKernel{S, F}
    state::S
    callback::F
end

"""
    RecordedKernel(records, callback)

A pair kernel whose host records are already known: `records` is a vector of
immutable per-host records indexed by population ID, and `callback` is as for
[`StatefulKernel`](@ref). Its hazard cannot change during a run, so it is the
likelihood counterpart of a live `StatefulKernel`: build one directly from
measured covariates, or call [`record_kernel`](@ref) on a `StatefulKernel` after
simulation to extract one from the records it read.
"""
struct RecordedKernel{S <: AbstractVector, F}
    state::S
    callback::F
end

"""
    LayerHost

One host of an [`InfectionLayer`](@ref) as a live [`StatefulKernel`](@ref)
projection sees it in a likelihood: its population `id`, its `infection_time`
(`NaN` if never infected), and a `state` holding the layer's per-host times
under their keys. `state` reads like an individual's: `state[key]` and
`get(state, key, default)` give the recorded time, including a recorded `NaN`,
and a host whose entry is `missing` has none, so `get` returns the default and
`state[key]` throws.
Reading a key the layer did not record throws an `ArgumentError`, so a
projection cannot silently fall back to a default for a time the likelihood
was never given.
"""
struct LayerHost{T, S}
    id::Int
    infection_time::T
    state::S
end

# The recorded times of one host of an infection layer, read from the layer's
# columns on demand so that a column holding `missing` stays type-stable to
# read. `missing` marks a time the host does not have, as an absent key does on
# an individual.
struct _LayerHostState{S <: NamedTuple}
    columns::S
    index::Int
end
@inline function _layer_time(s::_LayerHostState, key::Symbol)
    haskey(s.columns, key) || throw(
        ArgumentError(
            "the infection layer holds no host time `$key`; add it to `host_times`"
        )
    )
    return s.columns[key][s.index]
end
@inline function Base.get(s::_LayerHostState, key::Symbol, default)
    value = _layer_time(s, key)
    return ismissing(value) ? default : value
end
@inline function Base.getindex(s::_LayerHostState, key::Symbol)
    value = _layer_time(s, key)
    ismissing(value) && throw(KeyError(key))
    return value
end
Base.haskey(s::_LayerHostState, key::Symbol) = !ismissing(_layer_time(s, key))

function _layer_host(data, i)
    return LayerHost(i, data.infection_time[i], _LayerHostState(host_times(data), i))
end

_pair_state(project, individual) = project(individual)
_pair_state(records::AbstractVector, individual) = records[individual.id]
# The projection a live kernel reads host state through. Used only on a
# `StatefulKernel`, whose `state` field is always this projection; a
# `RecordedKernel`'s hazard cannot change during a run and so has none.
_kernel_projection(k::StatefulKernel) = k.state
_kernel_projection(k::CalendarKernel) = _kernel_projection(k.kernel)

"""
    watched_records(kernel) -> Tuple

The host records `kernel`'s hazard depends on, as the projections a race
compares to decide whether a pending contact needs redrawing. Defaults to `()`,
which marks a kernel whose hazard cannot change during a run — true of every
built-in kernel except a live [`StatefulKernel`](@ref), which reports its own
projection, and a [`CalendarKernel`](@ref), which delegates to the kernel it
wraps. A kernel may depend on host state only through a record this reports:
that restriction is what lets [`record_kernel`](@ref) reproduce a run's hazards
from recorded records alone.

A race holds the union of every route's kernel's watched records, keyed by
record rather than by route, so a changed record redraws only the proposals
whose route-kernel declared it; a route whose kernel reports `()` is never
redrawn.
"""
watched_records(k) = ()
watched_records(k::StatefulKernel) = (_kernel_projection(k),)
watched_records(k::CalendarKernel) = watched_records(k.kernel)
_live_kernel(k) = !isempty(watched_records(k))

# The extra argument is supplied only by simulation; ordinary kernels retain
# their existing extension methods and compiled likelihood fast paths.
function pair_kernel(k, i, j, infection_time, opening, state)
    return pair_kernel(k, i, j, infection_time, opening)
end
function pair_kernel(k::StatefulKernel, i, j, infection_time, opening, state)
    return k.callback(
        PairContext(i, j, infection_time),
        _pair_state(k.state, state.individuals[i]),
        _pair_state(k.state, state.individuals[j])
    )
end
function pair_kernel(k::RecordedKernel, i, j, infection_time)
    return k.callback(PairContext(i, j, infection_time), k.state[i], k.state[j])
end
function pair_kernel(k::CalendarKernel, i, j, infection_time, opening, state)
    return _calendar_interval(pair_kernel(k.kernel, i, j, infection_time, opening, state), opening)
end

"""
    record_kernel(kernel, state::SimulationState)

Return a [`RecordedKernel`](@ref) with the same callback and a vector of host
records extracted from a finished simulation. For a `StatefulKernel`, apply its
projection to each individual and copy the results so later simulation
mutations cannot change the record. A `CalendarKernel` preserves its
calendar-time conversion. Other kernels are returned unchanged.

The projection must retain event dates or full histories when hazards change.
This is extraction, not automatic history logging: overwritten past values cannot
be recovered. The resulting likelihood evaluates the transmission contribution along these
histories. Include separate attribute/intervention models when their probabilities
also belong in the joint likelihood. During inference, construct
`RecordedKernel(records, callback)` with the current latent records on each call.
"""
record_kernel(k, state::SimulationState) = k
function record_kernel(k::StatefulKernel, state::SimulationState)
    return RecordedKernel(
        [deepcopy(_pair_state(k.state, ind)) for ind in state.individuals],
        k.callback
    )
end
function record_kernel(k::CalendarKernel, state::SimulationState)
    return CalendarKernel(record_kernel(k.kernel, state))
end
