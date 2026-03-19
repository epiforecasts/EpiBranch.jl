"""
    RingVaccination(; efficacy, delay_to_immunity=0.0, mode=:leaky)

Vaccinate traced contacts, reducing their susceptibility to infection.

Applied to contacts that have been traced (`:traced == true`, set by
[`ContactTracing`](@ref)). Contacts not yet infected at the time of
vaccination have their susceptibility reduced.

For post-exposure prophylaxis (PEP), set `delay_to_immunity = 0.0`
(the default). For ring vaccination with a vaccine that takes time to
confer protection, set `delay_to_immunity` to the appropriate delay.

Requires `:traced` (set by [`ContactTracing`](@ref)).

- `efficacy`: vaccine/PEP efficacy (0–1)
- `delay_to_immunity`: time from administration to protective immunity (days)
- `mode`: `:leaky` (susceptibility reduced by `efficacy` for everyone) or
  `:all_or_nothing` (fraction `efficacy` fully protected, rest unaffected)

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

        # Apply vaccine effect to susceptibility
        if rv.mode == :leaky
            ind.susceptibility *= (1.0 - rv.efficacy)
        elseif rv.mode == :all_or_nothing
            if rand(state.rng) < rv.efficacy
                ind.susceptibility = 0.0
            end
        end

        # Re-evaluate infection status given reduced susceptibility.
        # This is an approximation: infection was decided in _resolve_infection
        # before vaccination was applied (vaccination happens in
        # apply_post_transmission!). The re-sampling here is not exactly
        # equivalent to having vaccinated before the infection decision,
        # but is a reasonable approximation for the competing risk framework.
        if is_infected(ind) && ind.susceptibility < 1.0
            if rand(state.rng) > ind.susceptibility / 1.0
                ind.state[:infected] = false
            end
        end
    end
end

