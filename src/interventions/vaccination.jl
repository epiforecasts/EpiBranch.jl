"""
Vaccinate traced contacts, reducing their susceptibility to infection.

Applied to contacts that have been traced (`:traced == true`, set by
[`ContactTracing`](@ref)). Contacts not yet infected at the time of
vaccination have their susceptibility reduced.

For post-exposure prophylaxis (PEP, cf.
[pepbp](https://github.com/sophiemeakin/pepbp)), set `delay_to_immunity = 0.0`
(the default). For ring vaccination with a vaccine that takes time to
confer protection, set `delay_to_immunity` to the appropriate delay.

Requires `:traced` (set by [`ContactTracing`](@ref)).

Initialises: `:vaccinated`, `:vaccination_time`.
"""
Base.@kwdef struct RingVaccination <: AbstractIntervention
    efficacy::Float64
    delay_to_immunity::Float64 = 0.0
    mode::Symbol = :leaky
end

required_fields(::RingVaccination) = [:traced]

function initialise_individual!(rv::RingVaccination, individual, state)
    individual.state[:vaccinated] = false
    individual.state[:vaccination_time] = Inf
    return nothing
end

# For Scheduled start-time enforcement: the action time is the time the
# individual would be vaccinated, which equals their isolation_time
# (set upstream by ContactTracing). Reading from existing state means a
# contact whose trace time falls before policy start is filtered before
# any irreversible vaccination logic fires.
intervention_time(::RingVaccination, ind::Individual) = isolation_time(ind)

function reset!(::RingVaccination, ind::Individual)
    # Best-effort: clear the vaccination flag and time. The susceptibility
    # multiplier (in :leaky mode) and rand-driven flips of :infected (in
    # both modes) are not inverted here. This branch is only reached if
    # the population gate (is_active) lets a contact through but their
    # action time falls pre-policy, which is impossible when vaccination
    # uses pre-existing isolation_time as its intervention_time, so the
    # reset path is effectively unused for RingVaccination today.
    ind.state[:vaccinated] = false
    ind.state[:vaccination_time] = Inf
    return nothing
end

function apply_post_transmission!(rv::RingVaccination, state, new_contacts)
    for ind in new_contacts
        # Only vaccinate traced contacts
        is_traced(ind) || continue

        # Already vaccinated (by a previous intervention)
        is_vaccinated(ind) && continue

        # Vaccination time = when the contact was traced
        # (approximated by the trace time, which is stored in isolation_time
        # if quarantine_on_trace was set by ContactTracing)
        vacc_time = isolation_time(ind)
        vacc_time == Inf && continue

        ind.state[:vaccinated] = true
        ind.state[:vaccination_time] = vacc_time

        # Check if immunity has developed by the time of potential transmission
        immunity_time = vacc_time + rv.delay_to_immunity
        if ind.infection_time < immunity_time
            # Transmission happened before immunity developed — no protection
            continue
        end

        if rv.mode == :leaky
            # Leaky: every vaccinated individual has reduced susceptibility
            ind.susceptibility *= (1.0 - rv.efficacy)
            # Retroactively re-evaluate infection
            if is_infected(ind) && rand(state.rng) < rv.efficacy
                ind.state[:infected] = false
            end
        elseif rv.mode == :all_or_nothing
            # All-or-nothing: fraction efficacy fully protected, rest unaffected
            if rand(state.rng) < rv.efficacy
                ind.susceptibility = 0.0
                if is_infected(ind)
                    ind.state[:infected] = false
                end
            end
        end
    end
end
