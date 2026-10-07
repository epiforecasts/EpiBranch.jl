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
                contacts_from = :infection, traceable = 1.0)

A route of transmission, such as community, household or funeral, with its
own start, end and timing over a case's natural history. Used with
[`RoutedNetwork`](@ref), where each route reaches a different set of people
in the network.

- `name` labels the route (`:community`, `:household`, `:funeral`, …).
- `from` is the state at which this route's infectiousness begins. `:infection`
  opens it at the infection time itself; any other state opens it at that
  state's time, following the `<state>_time` convention that
  [`Transition`](@ref) writes. The default, `nothing`, opens it at
  `:infectious` when the natural history has a step to it, otherwise at
  infection.
- `until` is the tuple of states that end the route; it closes at the
  earliest of them. A state ends only the routes that list it, so a control
  measure can end one route and leave another open. List
  [`EpiBranch.INTERVENTION_REMOVAL`](@ref) for a route that the model's
  interventions (isolation, or quarantine after being traced) should end.
- `kernel` is the contact interval on this route: the time in days from the
  route opening to a contact that would infect if the route were still open.
- `reach` names the set of people the route reaches in the model's contact
  structure, such as one kind of tie in a network. Defaults to `name`.
- `contacts_from` is the state from which the people the route reaches are the
  case's contacts, which is what contact tracing acts on. The default,
  `:infection`, suits a route over standing relationships such as a household,
  whose members are contacts however early the case is isolated. A route whose
  contacts come about through an event names that event's state, such as
  `:died` for a funeral: its contacts exist only if the event happens before the
  route is cut, and cannot be traced before it.
- `traceable` is the probability that a case can name a given contact made on
  this route, so that contact tracing can find it. People can name the people
  they live with but not the strangers they stood next to, so a household route
  might keep the default `1.0` and a community route of casual encounters take
  something much lower. `true` and `false` also work, as `1.0` and `0.0`.

  Naming comes before the tracing policy: a contact is traced only if the case
  names it and [`ContactTracing`](@ref) then traces it, so the overall
  probability is `traceable` times the tracing probability. Use `traceable` for
  what the relationship allows and the tracing probability for how well the
  programme performs, and count each limit in only one of them. Each model
  applies the probability to its contacts in its own way (see
  `EpiNetwork.RoutedNetwork`).

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

A case that recovers never opens the funeral route, and with
`contacts_from = :died` contact tracing also treats a survivor as having no
funeral contacts.

A community route whose contacts are mostly strangers, of whom a case can name
one in five:

```julia
community = RouteWindow(:community;
    until = (:recovered, EpiBranch.INTERVENTION_REMOVAL),
    kernel = Exponential(4.0), traceable = 0.2)
```
"""
struct RouteWindow{K, R}
    name::Symbol
    from::Union{Symbol, Nothing}
    until::Tuple
    kernel::K
    reach::R
    contacts_from::Symbol
    traceable::Float64

    function RouteWindow(
            name::Symbol, from::Union{Symbol, Nothing}, until::Tuple,
            kernel::K, reach::R, contacts_from::Symbol,
            traceable::Real
        ) where {K, R}
        0 <= traceable <= 1 || throw(
            ArgumentError(
                "route :$name has traceable = $traceable; it is a probability and " *
                    "must lie in [0, 1]"
            )
        )
        return new{K, R}(
            name, from, until, kernel, reach, contacts_from,
            Float64(traceable)
        )
    end
end

function RouteWindow(
        name::Symbol; from::Union{Symbol, Nothing} = nothing, until::Tuple = (),
        kernel, reach = name, contacts_from::Symbol = :infection,
        traceable::Real = 1.0
    )
    return RouteWindow(name, from, until, kernel, reach, contacts_from, traceable)
end

function Base.show(io::IO, w::RouteWindow)
    print(
        io, "RouteWindow(:", w.name, ", from=", repr(w.from),
        ", until=", w.until, ", kernel=", nameof(typeof(w.kernel))
    )
    w.contacts_from === :infection || print(io, ", contacts_from=:", w.contacts_from)
    w.traceable == 1 || print(io, ", traceable=", w.traceable)
    return print(io, ")")
end

"""
    window_open(individual, window)

Time (days) at which transmission by route `window` starts for a case, or
`Inf` if the case never reaches its `from` state, in which case the route
transmits nothing. A route with `from = nothing` opens when the case becomes
infectious if the natural history has that step, otherwise at infection.
"""
window_open(ind::Individual, w::RouteWindow) = _window_open(ind, _open_state(ind, w.from))

_open_state(ind::Individual, from::Symbol) = from
function _open_state(ind::Individual, ::Nothing)
    return haskey(ind.state, :infectious_time) ? :infectious : :infection
end

"""
    window_close(individual, window, interventions = ())

Time (days) at which transmission by route `window` stops for a case: the
earliest of its `until` states, or `Inf` if none has been reached. Each route
is ended separately, so the same event can end one route and leave another
open.

A route listing [`EpiBranch.INTERVENTION_REMOVAL`](@ref) also closes when the
`interventions` isolate or quarantine the case. Pass the interventions the
simulation used to get the same closing time as the simulation.
"""
function window_close(ind::Individual, w::RouteWindow, interventions = ())
    return _route_close(ind, w, interventions)
end
