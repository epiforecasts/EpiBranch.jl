"""
    ContactTracing(; probability, delay, quarantine_on_trace=true, start_time=0.0)

Trace contacts of isolated symptomatic cases with given probability and delay.

Requires fields set by [`Isolation`](@ref): `:isolated`, `:isolation_time`.
Also requires `:asymptomatic` and `:onset_time` (from `clinical_presentation()`).

Initialises: `:traced`, `:quarantined`.
"""
Base.@kwdef struct ContactTracing <: AbstractIntervention
    probability::Float64
    delay::Distribution
    quarantine_on_trace::Bool = true
    start_time::Float64 = 0.0
end

required_fields(::ContactTracing) = [:isolated, :asymptomatic]
start_time(ct::ContactTracing) = ct.start_time
intervention_time(::ContactTracing, ind::Individual) = isolation_time(ind)

function reset!(::ContactTracing, ind::Individual)
    ind.state[:traced] = false
    ind.state[:quarantined] = false
    if is_isolated(ind)
        ind.state[:isolated] = false
        ind.state[:isolation_time] = Inf
    end
    return nothing
end

function initialise_individual!(ct::ContactTracing, individual, state)
    individual.state[:traced] = false
    individual.state[:quarantined] = false
    return nothing
end

function apply_post_transmission!(ct::ContactTracing, state, new_contacts)
    for ind in new_contacts
        # O(1) parent lookup: id == 1-based index into individuals
        ind.parent_id == 0 && continue
        ind.parent_id > length(state.individuals) && continue
        parent = state.individuals[ind.parent_id]

        # Contact tracing only triggers from symptomatic, isolated parents
        is_asymptomatic(parent) && continue
        !is_isolated(parent) && continue

        # Trace with given probability
        if rand(state.rng) < ct.probability
            trace_delay = rand(state.rng, ct.delay)
            trace_time = isolation_time(parent) + trace_delay

            ind.state[:traced] = true

            if ct.quarantine_on_trace
                ind.state[:quarantined] = true
                # Quarantine: isolate at trace time (can be before onset)
                if is_isolated(ind)
                    # Already isolated via self-reporting — keep earlier time
                    set_isolated!(ind, min(isolation_time(ind), trace_time))
                else
                    set_isolated!(ind, trace_time)
                end
            else
                ind_onset = onset_time(ind)
                if !isnan(ind_onset)
                    # No quarantine: isolated at onset or trace, whichever is later
                    # But take minimum with any existing isolation (self-reporting)
                    traced_iso = max(ind_onset, trace_time)
                    if is_isolated(ind)
                        set_isolated!(ind, min(isolation_time(ind), traced_iso))
                    else
                        set_isolated!(ind, traced_iso)
                    end
                end
            end
        end
    end
end
