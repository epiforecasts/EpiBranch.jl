"""
    ContactTracing(; probability, delay, quarantine_on_trace=true, start_time=0.0)

Trace contacts of isolated symptomatic cases with given probability and delay.

Requires fields set by [`Isolation`](@ref): `:isolated`, `:isolation_time`.
Also reads `:asymptomatic` and `:onset_time` (from `clinical_presentation()`).

Initialises: `:traced`, `:quarantined`.
"""
Base.@kwdef struct ContactTracing <: AbstractIntervention
    probability::Float64
    delay::Distribution
    quarantine_on_trace::Bool = true
    start_time::Float64 = 0.0
end

required_fields(::ContactTracing) = [:isolated, :asymptomatic]

function initialise_individual!(ct::ContactTracing, individual, state)
    individual.state[:traced] = false
    individual.state[:quarantined] = false
    return nothing
end

function apply_post_transmission!(ct::ContactTracing, state, new_contacts)
    for ind in new_contacts
        ind.infection_time < ct.start_time && continue

        # Find parent
        parent_idx = findfirst(i -> i.id == ind.parent_id, state.individuals)
        parent_idx === nothing && continue
        parent = state.individuals[parent_idx]

        # Contact tracing only triggers from symptomatic, isolated parents
        is_asymptomatic(parent) && continue
        !is_isolated(parent) && continue

        # Trace with given probability
        if rand(state.rng) < ct.probability
            ind.state[:traced] = true

            trace_delay = rand(state.rng, ct.delay)
            trace_time = isolation_time(parent) + trace_delay

            if ct.quarantine_on_trace
                ind.state[:quarantined] = true
                set_isolated!(ind, trace_time)
            else
                ind_onset = onset_time(ind)
                if !isnan(ind_onset)
                    set_isolated!(ind, max(ind_onset, trace_time))
                end
            end
        end
    end
end
