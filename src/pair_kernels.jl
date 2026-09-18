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

Interpret a shared distribution, pair callback or `ContextualKernel` on the
simulation's calendar-time axis. In network and household models, condition the
returned distribution on surviving to the infector's infectious opening, then
subtract that opening to obtain a contact interval. Network per-edge distribution
vectors are supported too.

The distribution's hazard at calendar date `t` is the pair's contact rate at `t`.
The same conditioning is used by infection-layer likelihoods, including when
infectious opening times change during inference. The calendar law must have
positive survival at each opening. Its parameters and any covariate tables must
remain fixed throughout a simulation and be supplied consistently to inference.
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
