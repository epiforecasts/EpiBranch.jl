# ── Transmission-route windows ───────────────────────────────────────
#
# The simplest model gives a case one infectiousness profile and one set of
# people it can reach. Real transmission is often several routes at once, each
# open over a different stretch of the case's natural history and each cut by
# different things: a community route that ends when the case isolates, a
# household route that does not, a funeral route that only opens at death.
#
# A route window is the unit that makes those one mechanism. It names the state
# at which the route's infectiousness begins, the states that end it, and the
# survival kernel timing contacts within it. Who the route reaches is the
# model's business, since only the model knows its own structure, so the window
# carries a `reach` tag that the model resolves.
#
# Keeping the censoring per-window is the point. `R` stays the intrinsic
# reproduction number a case would achieve if never removed, and the realised
# one falls out of which windows were cut and when.

"""
    RouteWindow(name; from = nothing, until, kernel, reach = name,
                contacts_from = :infection)

One transmission route, open over part of a case's natural history.

- `name` labels the route (`:community`, `:household`, `:funeral`, …).
- `from` is the state at which this route's infectiousness begins. `:infection`
  opens it at the infection time itself; any other state opens it at that
  state's time, following the `<state>_time` convention that
  [`Transition`](@ref) writes. The default, `nothing`, takes the start the
  model derives from its progression, as the continuous-time processes do for
  their own `from`: `:infectious` when a transition writes it, otherwise the
  infection time.
- `until` is the tuple of states that end the route. The window closes at the
  earliest of their times. A state that no window lists never censors
  anything, and a state listed by one window and not another censors only the
  first — which is how a control measure cuts one route and leaves another.
  List [`EpiBranch.INTERVENTION_REMOVAL`](@ref) for a route that the composed
  interventions (isolation, quarantine on being traced) should end.
- `kernel` is the route's contact-interval distribution, measured from the
  window opening. A model reads it when it resolves `reach` into the route's
  contacts.
- `reach` tags who the route reaches, for the model to resolve. Defaults to
  `name`, which is usually what a model keys its structure on.
- `contacts_from` is the state from which the people the route reaches are the
  case's contacts, which is what contact tracing acts on. The default,
  `:infection`, suits a route over standing relationships such as a household,
  whose members are contacts however early the case is isolated. A route whose
  contacts come about through an event names that event's state, such as
  `:died` for a funeral: its contacts exist only if the event happens before the
  route is cut, and cannot be traced before it.

# Examples

A case that stops mixing in the community when it isolates but keeps infecting
the people it lives with:

```julia
community = RouteWindow(:community; from = :infectious,
    until = (:recovered, EpiBranch.INTERVENTION_REMOVAL), kernel = Exponential(4.0))
household = RouteWindow(:household; from = :infectious,
    until = (:recovered,), kernel = Weibull(1.5, 3.0))
```

Ebola transmission at funerals, a route that only opens once the case has died
and closes at burial:

```julia
funeral = RouteWindow(:funeral; from = :died, until = (:buried,),
    kernel = Exponential(1.0), contacts_from = :died)
```

Because the window contributes contacts only once its `from` state has
occurred, a case that recovers never opens the funeral route at all, so nothing
is created only to be censored. `contacts_from = :died` tells contact tracing
the same: the survivor had no funeral contacts to trace.
"""
struct RouteWindow{K, R}
    name::Symbol
    from::Union{Symbol, Nothing}
    until::Tuple
    kernel::K
    reach::R
    contacts_from::Symbol
end

function RouteWindow(
        name::Symbol; from::Union{Symbol, Nothing} = nothing, until::Tuple = (),
        kernel, reach = name, contacts_from::Symbol = :infection)
    return RouteWindow(name, from, until, kernel, reach, contacts_from)
end

function Base.show(io::IO, w::RouteWindow)
    print(io, "RouteWindow(:", w.name, ", from=", repr(w.from),
        ", until=", w.until, ", kernel=", nameof(typeof(w.kernel)))
    w.contacts_from === :infection || print(io, ", contacts_from=:", w.contacts_from)
    print(io, ")")
end

"""
    window_open(individual, window)

Time at which `window` opens for `individual`, or `Inf` if its `from` state has
not been reached. A route that never opened contributes no contacts.

A window with `from = nothing` opens at the individual's `:infectious_time` when
it has one and at its infection time otherwise, which is the start a model
derives from a progression with or without an `:infectious` transition.
"""
window_open(ind::Individual, w::RouteWindow) = _window_open(ind, _open_state(ind, w.from))

_open_state(ind::Individual, from::Symbol) = from
function _open_state(ind::Individual, ::Nothing)
    haskey(ind.state, :infectious_time) ? :infectious : :infection
end

"""
    window_close(individual, window, interventions = ())

Time at which `window` closes for `individual`: the earliest of its `until`
states' times, or `Inf` if none has been reached. Censoring is per window, so
the same removal can end one route and leave another running.

A window listing [`EpiBranch.INTERVENTION_REMOVAL`](@ref) also closes when the
composed `interventions` remove the case. Pass the interventions the simulation
used to get the same closing time as the simulation.
"""
function window_close(ind::Individual, w::RouteWindow, interventions = ())
    return _route_close(ind, w, interventions)
end
