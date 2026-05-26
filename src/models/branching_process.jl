"""
    step!(model::BranchingProcess, state::SimulationState)

Process one generation of the branching process. Produces the new
contacts (both eventual infections and contacts whose transmission
will be blocked) without deciding `:infected` — the engine resolves
that via [`competing_risk`](@ref) after `step!` returns. See the
[Design](@ref "Simulation, mutation, and automatic differentiation")
section for implications on automatic differentiation.
"""
function step!(model::BranchingProcess, state::SimulationState)
    new_contacts = Individual[]
    for idx in state.active_ids
        individual = state.individuals[idx]
        offspring_result = _draw_offspring(state.rng, model.offspring, individual, state)
        gt_dist = get_generation_time(model.generation_time, individual)
        _create_contacts!(new_contacts, offspring_result, individual, state, gt_dist)
    end
    return new_contacts
end

# ── Offspring drawing ────────────────────────────────────────────────

"""Single-type offspring draw."""
function _draw_offspring(rng::AbstractRNG, offspring::Distribution,
        individual, state::SimulationState)
    rand(rng, offspring)
end

"""Function-based offspring draw. The function may be called as
`(rng, individual)` or `(rng, individual, state)`; the latter form lets
the offspring rule read population-level state (e.g. cumulative cases
for time- or policy-dependent caps)."""
function _draw_offspring(rng::AbstractRNG, offspring::Function,
        individual, state::SimulationState)
    if applicable(offspring, rng, individual, state)
        return offspring(rng, individual, state)
    end
    return offspring(rng, individual)
end

# ── Contact creation ─────────────────────────────────────────────────

"""Compute a contact's infection time from the parent's infection time
and the generation-time draw. Dispatches on the generation-time spec:
[`NoGenerationTime`](@ref) (no timing) returns the parent's time;
otherwise samples and adds, clamped to `state.latent_period`."""
_infection_time(::NoGenerationTime, parent, state) = parent.infection_time
function _infection_time(gt_dist::Distribution, parent, state)
    gt = max(rand(state.rng, gt_dist), state.latent_period)
    return parent.infection_time + gt
end

"""Single-type contacts."""
function _create_contacts!(new_contacts,
        n_contacts::Int, parent, state, gt_dist)
    for _ in 1:n_contacts
        inf_time = _infection_time(gt_dist, parent, state)
        make_contact!(new_contacts, state, parent, inf_time)
    end
    return nothing
end

"""Multi-type contacts."""
function _create_contacts!(new_contacts,
        counts::Vector{Int}, parent, state, gt_dist)
    for (type_idx, n) in enumerate(counts)
        for _ in 1:n
            inf_time = _infection_time(gt_dist, parent, state)
            make_contact!(new_contacts, state, parent, inf_time; type_idx)
        end
    end
    return nothing
end
