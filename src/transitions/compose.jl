"""
    clinical_default(; reporting_delay = nothing, prob_hospitalisation = 0.2,
                       admission_delay = nothing, prob_death = 0.05,
                       outcome_delay = nothing, age_specific_cfr = NoCFR())

Build the conventional clinical-transition stack — reporting, then
hospitalisation, then a competing recovery/death pair — from a flat
keyword interface that mirrors the legacy `DelayOpts` / `OutcomeOpts`
shape. Distributions left as `nothing` skip the corresponding transition.

This is the convenience factory; for non-standard timelines (e.g. ICU
admission given hospitalisation, treatment-conditional outcomes) build
the vector by hand using the individual transition types.
"""
function clinical_default(; reporting_delay::Union{Distribution, Nothing} = nothing,
        prob_hospitalisation::Real = 0.2,
        admission_delay::Union{Distribution, Nothing} = nothing,
        prob_death::Real = 0.05,
        outcome_delay::Union{Distribution, Nothing} = nothing,
        age_specific_cfr = NoCFR())
    transitions = AbstractClinicalTransition[]
    reporting_delay !== nothing && push!(transitions, Reporting(delay = reporting_delay))
    admission_delay !== nothing && push!(transitions,
        Hospitalisation(delay = admission_delay,
            probability = float(prob_hospitalisation)))
    if outcome_delay !== nothing
        push!(transitions,
            Death(delay = outcome_delay,
                probability = float(prob_death),
                age_specific_cfr = age_specific_cfr))
        push!(transitions, Recovery(delay = outcome_delay))
    end
    return transitions
end

"""
    _finalise_terminal!(individual, transitions)

After all `resolve_individual!`s have run, collect terminal candidates
across all terminal transitions and set `:outcome` and `:outcome_time`
to the earliest. If no terminal transition fires, neither key is set.
"""
function _finalise_terminal!(individual, transitions)
    best_time = Inf
    best_label = :none
    has_any = false
    for t in transitions
        is_terminal(t) || continue
        ev = terminal_event(t, individual)
        ev === nothing && continue
        time, label = ev
        if time < best_time
            best_time = time
            best_label = label
            has_any = true
        end
    end
    if has_any
        individual.state[:outcome_time] = best_time
        individual.state[:outcome] = best_label
    end
    return nothing
end
