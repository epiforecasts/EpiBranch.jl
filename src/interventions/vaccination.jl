"""
    RingVaccination(; efficacy, delay_to_immunity=0.0, mode=:leaky)

Vaccinate traced contacts, reducing their susceptibility to infection.

Ring vaccination is applied in `apply_post_transmission!` to contacts that
have been traced (`:traced == true`, set by [`ContactTracing`](@ref)).
Contacts not yet infected at the time of vaccination have their
susceptibility reduced.

Requires `:traced` to be set (by [`ContactTracing`](@ref)).

- `efficacy`: vaccine efficacy (0–1)
- `delay_to_immunity`: time from vaccination to protective immunity (days).
  If the contact's generation time is shorter than the delay, vaccination
  has no effect.
- `mode`: `:leaky` (everyone's susceptibility reduced by `efficacy`) or
  `:all_or_nothing` (fraction `efficacy` are fully protected, rest unaffected)

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

"""
    PEP(; efficacy, mode=:leaky)

Post-exposure prophylaxis for traced contacts. Similar to
[`RingVaccination`](@ref) but acts immediately (no delay to immunity).

- `efficacy`: reduction in probability of infection (0–1)
- `mode`: `:leaky` or `:all_or_nothing`

Initialises: `:pep_received`.
"""
Base.@kwdef struct PEP <: AbstractIntervention
    efficacy::Float64
    mode::Symbol = :leaky
end

required_fields(::PEP) = [:traced]

function initialise_individual!(pep::PEP, individual, state)
    individual.state[:pep_received] = false
    return nothing
end

function apply_post_transmission!(pep::PEP, state, new_contacts)
    for ind in new_contacts
        is_traced(ind) || continue

        ind.state[:pep_received] = true

        # PEP acts immediately — no delay
        if pep.mode == :leaky
            ind.susceptibility *= (1.0 - pep.efficacy)
        elseif pep.mode == :all_or_nothing
            if rand(state.rng) < pep.efficacy
                ind.susceptibility = 0.0
            end
        end

        # Re-evaluate infection (same approximation as RingVaccination)
        if is_infected(ind) && ind.susceptibility < 1.0
            if rand(state.rng) > ind.susceptibility
                ind.state[:infected] = false
            end
        end
    end
end
