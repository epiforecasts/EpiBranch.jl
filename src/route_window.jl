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
    RouteWindow(name; from, until, kernel, reach = name)

One transmission route, open over part of a case's natural history.

- `name` labels the route (`:community`, `:household`, `:funeral`, …).
- `from` is the state at which this route's infectiousness begins. `:infection`
  opens it at the infection time itself; any other state opens it at that
  state's time, following the `<state>_time` convention that
  [`Transition`](@ref) writes.
- `until` is the tuple of states that end the route. The window closes at the
  earliest of their times. A state that no window lists never censors
  anything, and a state listed by one window and not another censors only the
  first — which is how a control measure cuts one route and leaves another.
- `kernel` times contacts within the window, as a contact-interval
  distribution measured from the window opening.
- `reach` tags who the route reaches, for the model to resolve. Defaults to
  `name`, which is usually what a model keys its structure on.

# Examples

A case that stops mixing in the community when it isolates but keeps infecting
the people it lives with:

```julia
community = RouteWindow(:community; from = :infectious,
    until = (:recovered, :isolated), kernel = Exponential(4.0))
household = RouteWindow(:household; from = :infectious,
    until = (:recovered,), kernel = Weibull(1.5, 3.0))
```

Ebola transmission at funerals, a route that only opens once the case has died
and closes at burial:

```julia
funeral = RouteWindow(:funeral; from = :died, until = (:buried,),
    kernel = Exponential(1.0))
```

Because the window contributes contacts only once its `from` state has
occurred, a case that recovers never opens the funeral route at all, so nothing
is created only to be censored.
"""
struct RouteWindow{K, R}
    name::Symbol
    from::Symbol
    until::Tuple
    kernel::K
    reach::R
end

function RouteWindow(name::Symbol; from::Symbol = :infection, until::Tuple = (),
        kernel, reach = name)
    return RouteWindow(name, from, until, kernel, reach)
end

function Base.show(io::IO, w::RouteWindow)
    print(io, "RouteWindow(:", w.name, ", from=:", w.from,
        ", until=", w.until, ", kernel=", nameof(typeof(w.kernel)), ")")
end

"""
    window_open(individual, window)

Time at which `window` opens for `individual`, or `Inf` if its `from` state has
not been reached. A route that never opened contributes no contacts.
"""
window_open(ind::Individual, w::RouteWindow) = _window_open(ind, w.from)

"""
    window_close(individual, window)

Time at which `window` closes for `individual`: the earliest of its `until`
states' times, or `Inf` if none has been reached. Censoring is per window, so
the same removal can end one route and leave another running.
"""
window_close(ind::Individual, w::RouteWindow) = _window_close(ind, w.until)
