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

Resolve a contact-interval distribution for an ordered pair. A shared continuous
distribution is returned unchanged; an ordinary callable receives the two IDs;
a [`ContextualKernel`](@ref) receives a [`PairContext`](@ref). Structured-process
extensions can use this method to share kernel semantics with the likelihood.
The five-argument form supplies the infectious opening required by `CalendarKernel`.
"""
pair_kernel(k::ContinuousUnivariateDistribution, i, j, infection_time) = k
pair_kernel(k, i, j, infection_time) = k(i, j)
function pair_kernel(k::ContextualKernel, i, j, infection_time)
    k.callback(PairContext(i, j, infection_time))
end

"""
    CalendarKernel(kernel)

Interpret a shared distribution, pair callback, `ContextualKernel` or
`StatefulKernel` on the
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
    _calendar_interval(pair_kernel(k.kernel, i, j, infection_time), infectious_time)
end
function pair_kernel(::CalendarKernel, i, j, infection_time)
    throw(ArgumentError("CalendarKernel also requires the infectious opening time"))
end
function _calendar_interval(distribution, opening)
    truncated(distribution; lower = opening) - opening
end

"""
    StatefulKernel(state, callback)

A pair kernel with explicit host state. In simulation, `state(individual)` selects
an immutable record (for example a named tuple of attributes and event dates).
`callback(context::PairContext, source, target)` returns the contact-interval
law from those records. Wrap in [`CalendarKernel`](@ref) for calendar-time laws.

For likelihood evaluation, supply a vector of records indexed by population ID
as `state`, or use [`record_kernel`](@ref) to extract them after simulation.
The callback is identical in both paths. Records may contain typed covariates
and dated histories; the likelihood does not construct individuals.

Callbacks must describe a predictable hazard: adding an event at time `t` must
not change the hazard before `t`. A final vaccinated flag alone is insufficient;
retain its date and the earlier hazard. State changes occur through the existing
case-resolution and intervention hooks. The callback and projection must not
mutate state, and a kernel may read host state only through its projection,
since that record is all the likelihood is given.

Simulation keeps contacts consistent with the hazards in force as records
change; a run whose records never change matches an ordinary kernel exactly.
"""
struct StatefulKernel{S, F}
    state::S
    callback::F
end

_pair_state(project, individual) = project(individual)
_pair_state(records::AbstractVector, individual) = records[individual.id]
# The projection a live kernel reads host state through, or `nothing` for a
# kernel whose hazards cannot change during a run. A race compares successive
# projections to decide whether pending contacts need redrawing, so a kernel may
# depend on host state only through this record — the restriction `record_kernel`
# already relies on to reproduce a run's hazards from recorded records alone.
_kernel_projection(k) = nothing
_kernel_projection(k::StatefulKernel) = k.state
_kernel_projection(k::StatefulKernel{<:AbstractVector}) = nothing
_kernel_projection(k::CalendarKernel) = _kernel_projection(k.kernel)
_live_kernel(k) = _kernel_projection(k) !== nothing

# The projection a race has to watch for changes. Resolving a case writes only to
# that case's own record, and no contact already drawn depends on it: contacts to
# the case are settled, and its own contacts are drawn afterwards. So records that
# pending contacts depend on can move only through an intervention, and without
# one a live kernel races exactly as an ordinary kernel does.
function _watched_projection(kernel, interventions)
    isempty(interventions) ? nothing : _kernel_projection(kernel)
end

# The extra argument is supplied only by simulation; ordinary kernels retain
# their existing extension methods and compiled likelihood fast paths.
function pair_kernel(k, i, j, infection_time, opening, state)
    pair_kernel(k, i, j, infection_time, opening)
end
function pair_kernel(k::StatefulKernel, i, j, infection_time, opening, state)
    k.callback(PairContext(i, j, infection_time),
        _pair_state(k.state, state.individuals[i]),
        _pair_state(k.state, state.individuals[j]))
end
function pair_kernel(k::StatefulKernel{<:AbstractVector}, i, j, infection_time)
    k.callback(PairContext(i, j, infection_time), k.state[i], k.state[j])
end
function pair_kernel(::StatefulKernel, i, j, infection_time)
    throw(ArgumentError("a live StatefulKernel needs simulation state; " *
                        "supply recorded host states for likelihood evaluation with record_kernel"))
end
function pair_kernel(k::CalendarKernel, i, j, infection_time, opening, state)
    _calendar_interval(pair_kernel(k.kernel, i, j, infection_time, opening, state), opening)
end

"""
    record_kernel(kernel, state::SimulationState)

Return a kernel with the same callback and a vector of host records extracted
from a finished simulation. For a `StatefulKernel`, apply its projection to each
individual and copy the results so later simulation mutations cannot change the
record. A `CalendarKernel` preserves its calendar-time conversion. Other kernels
are returned unchanged.

The projection must retain event dates or full histories when hazards change.
This is extraction, not automatic history logging: overwritten past values cannot
be recovered. The resulting likelihood evaluates the transmission contribution along these
histories. Include separate attribute/intervention models when their probabilities
also belong in the joint likelihood. During inference, construct
`StatefulKernel(records, callback)` with the current latent records on each call.
"""
record_kernel(k, state::SimulationState) = k
function record_kernel(k::StatefulKernel, state::SimulationState)
    StatefulKernel([deepcopy(_pair_state(k.state, ind)) for ind in state.individuals],
        k.callback)
end
function record_kernel(k::CalendarKernel, state::SimulationState)
    CalendarKernel(record_kernel(k.kernel, state))
end
