using Test
using EpiBranch
# Non-exported chain-size types, reached in production via chain_size_distribution.
using EpiBranch: GammaBorel, PoissonGammaChainSize
using Distributions
using Random
using StableRNGs
# Activates the serial_interval(model, infector, infectee) extension method.
# Not loaded on Julia 1.10, where these packages pull in EpiAwareADTools 0.1.
# Its xlogy rule clashes with the Mooncake rule.
const RUN_SERIAL_INTERVAL = VERSION >= v"1.11"
if RUN_SERIAL_INTERVAL
    using CensoredDistributions
    using ConvolvedDistributions
end

include("testutils/sim_analytical_consistency.jl")
include("testutils/progression.jl")

# CI runs one job per group to keep each job well within its timeout.
# `EPIBRANCH_TEST_GROUP` selects a group; `Pkg.test()` without it runs
# every file. A new file goes in the group matching its subject. Keep the
# running times of the two groups close.
const TEST_GROUPS = (
    engine = [
        "test_quality.jl",
        "test_types.jl",
        "test_signals.jl",
        "test_streams.jl",
        "test_branching_process.jl",
        "test_attributes.jl",
        "test_interventions.jl",
        "test_settle_case.jl",
        "test_case_hooks.jl",
        "test_group_vaccination.jl",
        "test_vaccine_effect.jl",
        "test_capacity_constrained.jl",
        "test_held_capacity.jl",
        "test_intervention_actions.jl",
        "test_competing_risks.jl",
        "test_ct_seams.jl",
        "test_source_tracing.jl",
        "test_composable_eligibility.jl",
        "test_isolation_seams.jl",
        "test_pc_observation.jl",
        "test_stopping_rules.jl",
        "test_transitions.jl",
        "test_infectiousness.jl",
        "test_funeral.jl",
        "test_density_dependent.jl",
        "test_structure_driven.jl",
        "test_reinfection.jl",
        "test_abort_infection.jl",
        "test_homogeneous.jl",
        "test_mixing.jl",
        "test_contact_matrices.jl",
        "test_continuation.jl",
        "test_state_space.jl",
        "test_population_dynamics.jl",
        "test_mixing_population_dynamics.jl",
        "test_mixing_vaccination.jl",
        "test_hybrid.jl",
        "test_generation_counts.jl",
        "test_generation_counts_population.jl",
        "test_generation_campaigns.jl",
        "test_linelist.jl",
        "test_route_windows.jl",
        "test_event_queue.jl",
    ],
    inference = [
        "test_model_spec.jl",
        "test_chains.jl",
        "test_generation_intervals.jl",
        "test_analytical.jl",
        "test_incidence_likelihood.jl",
        "test_likelihood_dists.jl",
        "test_tabulated_loglik.jl",
        "test_pairwise_likelihood.jl",
        "test_progression_likelihood.jl",
        "test_serial_interval.jl",
        "test_vaccine_likelihood.jl",
        "test_likelihood_compatibility.jl",
        "test_multitype.jl",
        "test_multitype_analytical.jl",
        "test_integration.jl",
        "test_r_targets.jl",
        "test_ad.jl",
        "test_stochastic_ad.jl",
        "test_pair_kernels.jl",
        "test_seeding_helpers.jl",
        "test_examples.jl",
    ],
)

const TEST_GROUP = Symbol(get(ENV, "EPIBRANCH_TEST_GROUP", "all"))
TEST_GROUP == :all || haskey(TEST_GROUPS, TEST_GROUP) ||
    error("EPIBRANCH_TEST_GROUP must be one of $(join(keys(TEST_GROUPS), ", ")) or all")

@testset "EpiBranch" begin
    for group in keys(TEST_GROUPS)
        TEST_GROUP in (:all, group) || continue
        @testset "$group" begin
            for file in TEST_GROUPS[group]
                file == "test_serial_interval.jl" && !RUN_SERIAL_INTERVAL && continue
                include(file)
            end
        end
    end
end
