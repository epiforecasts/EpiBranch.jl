"""
    ContactTracing(; probability, delay, quarantine_on_trace=true, start_time=0.0)

Trace contacts of isolated symptomatic cases with given probability and delay.

Traced contacts can be isolated earlier than they would be through symptom-based
surveillance alone. The timing follows ringbp's logic:
- Without quarantine: traced contact isolated at min(own onset + own delay, infector isolation time)
- With quarantine: traced contact can be quarantined before symptom onset

- `probability`: probability that each contact is successfully traced
- `delay`: distribution of time from infector isolation to contact being traced
- `quarantine_on_trace`: whether traced contacts are quarantined (can be before onset)
- `start_time`: simulation time when contact tracing policy begins
"""
Base.@kwdef struct ContactTracing <: AbstractIntervention
    probability::Float64
    delay::Distribution
    quarantine_on_trace::Bool = true
    start_time::Float64 = 0.0
end

function apply_post_transmission!(ct::ContactTracing, state, new_individuals)
    for ind in new_individuals
        ind.infection_time < ct.start_time && continue

        # Find parent
        parent_idx = findfirst(i -> i.id == ind.parent_id, state.individuals)
        parent_idx === nothing && continue
        parent = state.individuals[parent_idx]

        # Contact tracing only triggers from symptomatic, isolated parents
        parent.asymptomatic && continue
        !parent.isolated && continue

        # Trace with given probability
        if rand(state.rng) < ct.probability
            ind.traced = true

            # Compute trace time
            trace_delay = rand(state.rng, ct.delay)
            trace_time = parent.isolation_time + trace_delay

            if ct.quarantine_on_trace
                # Can quarantine before onset — isolation at trace time
                ind.quarantined = true
                ind.isolated = true
                ind.isolation_time = trace_time
            else
                # Can only isolate at or after onset
                if !isnan(ind.onset_time)
                    ind.isolated = true
                    ind.isolation_time = max(ind.onset_time, trace_time)
                end
            end
        end
    end
end
