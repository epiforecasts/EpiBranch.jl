# ── Sellke/Dijkstra continuous-time race ─────────────────────────────
#
# The generic continuous-time competing-risks primitive shared by the
# structure-driven models whose contacts are a finite, depleting set of
# existing nodes (a household clique, say). Members are processed in
# increasing infection-time order; popping the earliest pending infection
# makes it final, because any not-yet-processed infector has a later
# infection time, becomes infectious no earlier, and so reaches it later.

# When the infector becomes infectious: the `from` state's time (the infection
# time itself when the kernel times from :infection, otherwise a state key).
function _window_open(ind::Individual{T}, from::Symbol) where {T}
    from === :infection ? ind.infection_time :
    convert(T, get(ind.state, Symbol(from, :_time), T(Inf)))
end

# Earliest of the `until` removal states' times (Inf if none reached).
function _window_close(ind::Individual{T}, until::Tuple) where {T}
    isempty(until) ? T(Inf) :
    minimum(convert(T, get(ind.state, Symbol(s, :_time), T(Inf))) for s in until)
end

# ── Interventions on the continuous-time (Sellke) models ─────────────
# These models run their own event loop rather than the generation engine, so
# the engine's per-generation hook passes never fire. The one intervention seam
# is the infectious window: an intervention that removes a case from onward
# transmission (isolation) shortens it. After a case's natural history is
# stamped, run each intervention's per-individual resolution (so `Isolation`
# writes its isolation time), then close the window at the earliest removal
# across interventions as well as the `until` states.

# Run each intervention's per-individual resolution on a freshly-stamped case.
function _resolve_interventions!(state::SimulationState, ind, interventions)
    for iv in interventions
        resolve_individual!(iv, ind, state)
    end
    return nothing
end

# Earliest time any intervention removes `ind` from onward transmission.
function _intervention_removal_time(ind, interventions)
    t = Inf
    for iv in interventions
        t = min(t, infectious_removal_time(iv, ind))
    end
    return t
end

# Whether a continuous-time model honours an intervention — i.e. can express it
# through the infectious window. Perfect isolation shortens the window; a leaky
# isolation (`post_isolation_transmission > 0`) only reduces transmission, which
# the window cannot express, so it is not honoured. Interventions whose effect
# is a per-contact competing risk (leaky vaccination, contact tracing) likewise
# have no window representation. `Scheduled` is not honoured either: its
# time/count gate reads population state (`max_infection_time`, `cumulative_cases`)
# that the Sellke loop only reconciles after the run, so a `start_time` would
# never activate during it. The model warns for the unhonoured ones rather than
# silently ignoring them.
_sellke_honours(::AbstractIntervention) = false
_sellke_honours(iso::Isolation) = iso.post_isolation_transmission == 0
_sellke_honours(::Scheduled) = false

# Warn once (per `simulate` call) when a continuous-time model is handed
# interventions it cannot honour, so the limitation is loud rather than silent.
# Gated on `_honours_termination_controls`, which is `false` for exactly the
# structure-driven models that run their own Sellke loop.
function _warn_unhonoured_interventions(model, interventions)
    _honours_termination_controls(model) && return nothing
    unhonoured = unique(String[string(nameof(typeof(iv)))
                               for iv in interventions if !_sellke_honours(iv)])
    isempty(unhonoured) && return nothing
    @warn "$(nameof(typeof(model))) is a continuous-time model that expresses " *
          "interventions only through the infectious window; it does not honour " *
          "these, which will have no effect: $(join(unhonoured, ", ")). Express " *
          "such control as a removal `Transition` in the progression instead."
    return nothing
end

"""
    _sellke_race!(state, members, rng; seed!, targets, from, until)

Run the Sellke/Dijkstra continuous-time competing-risks race over the individuals
`members` (global ids). `seed!(best, members, rng)` fills the candidate infection
times `best` (indexed `1:length(members)`) with exogenous introductions or a seed.
`targets(infective_id, state)` yields `(target_id, kernel)` for the contacts an
infective can reach, each with its pairwise contact-interval kernel. Members are
processed in increasing infection time (each pop is final); on processing, the
case's natural history is stamped and it exposes still-susceptible targets with a
`from`-timed contact interval accepted inside its infectious window. Each case's
`interventions` are resolved after its natural history, and any that remove it
from transmission (isolation) shorten that window.
"""
function _sellke_race!(state::SimulationState, members::AbstractVector{Int},
        rng::AbstractRNG; seed!, targets, from::Symbol, until::Tuple,
        interventions = ())
    m = length(members)
    best = fill(Inf, m)
    src = zeros(Int, m)
    processed = falses(m)
    pos = Dict{Int, Int}(id => k for (k, id) in enumerate(members))

    seed!(best, members, rng)

    while true
        j = 0
        bt = Inf
        for k in 1:m
            if !processed[k] && best[k] < bt
                bt = best[k]
                j = k
            end
        end
        j == 0 && break
        processed[j] = true

        ind = state.individuals[members[j]]
        ind.infection_time = best[j]
        ind.state[:infected] = true
        ind.state[:index] = src[j] == 0
        if src[j] != 0
            infector = state.individuals[src[j]]
            ind.parent_id = infector.id
            ind.generation = infector.generation + 1
            ind.chain_id = infector.chain_id
        end
        resolve_transitions!(state, ind)
        _resolve_interventions!(state, ind, interventions)

        open_t = _window_open(ind, from)
        isfinite(open_t) || continue
        close_t = min(_window_close(ind, until),
            _intervention_removal_time(ind, interventions))

        for (target_id, kernel) in targets(members[j], state)
            k = get(pos, target_id, 0)
            (k == 0 || processed[k]) && continue
            dt = rand(rng, kernel)
            cand = open_t + dt
            (cand <= close_t && cand < best[k]) || continue
            best[k] = cand
            src[k] = members[j]
        end
    end
    return nothing
end
