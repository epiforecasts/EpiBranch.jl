function test_contextual_simulation(make_process, extract)
    @testset "Contextual kernel simulation and likelihood" begin
        covariates = [0.5, 1.0, 1.5]
        interval(context) = Exponential(0.25 * exp(
            0.1 * context.infector_infection_time + 0.2 * covariates[context.susceptible]))
        seen = Tuple{Int, Int, Float64}[]
        kernel = ContextualKernel(context -> begin
            push!(seen, (
                context.infector, context.susceptible, context.infector_infection_time))
            interval(context)
        end)
        progression = [Transition(:infectious; delay = 0.75),
            Transition(:recovered; from = :infectious, delay = 10.0, terminal = true)]
        spec = ModelSpec(make_process(kernel); progression)
        state = simulate(spec; rng = StableRNG(233))
        @test state.cumulative_cases == 3
        @test any(record -> record[3] > 0, seen)
        @test all(record -> record[3] == state.individuals[record[1]].infection_time, seen)
        data = extract(state, spec)
        contextual = ModelSpec(make_process(ContextualKernel(interval)); progression)
        frozen(i, j) = interval(PairContext(i, j, data.infection_time[i]))
        legacy = ModelSpec(make_process(frozen); progression)
        @test isfinite(loglikelihood(data, contextual))
        @test loglikelihood(data, contextual) ≈ loglikelihood(data, legacy)
        layout = compile_contact_pairs(data)
        @test pairwise_surv_loglik(ContextualKernel(interval), data, layout) ≈
              loglikelihood(data, legacy)

        fixed_ids(i, j) = Exponential(0.25 * covariates[j])
        fixed_context = ContextualKernel(context -> fixed_ids(context.infector, context.susceptible))
        id_spec = ModelSpec(make_process(fixed_ids); progression)
        context_spec = ModelSpec(make_process(fixed_context); progression)
        id_state = simulate(id_spec; rng = StableRNG(41))
        context_state = simulate(context_spec; rng = StableRNG(41))
        @test [i.infection_time for i in id_state.individuals] ==
              [i.infection_time for i in context_state.individuals]
        @test loglikelihood(extract(id_state, id_spec), id_spec) ≈
              loglikelihood(extract(context_state, context_spec), context_spec)
    end
end
