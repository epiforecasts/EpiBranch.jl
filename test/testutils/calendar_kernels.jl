function test_calendar_simulation(make_process, extract)
    @testset "Calendar kernels with sampled infectious openings" begin
        progression = [Transition(:infectious; delay = Uniform(0.4, 0.8)),
            Transition(:recovered; from = :infectious, delay = 4.0, terminal = true)]
        calendar = Weibull(2.0, 3.0)
        spec = ModelSpec(make_process(CalendarKernel(calendar)); progression)
        state = simulate(spec; rng = StableRNG(234))
        data = extract(state, spec)
        @test any(isfinite, data.infectious_time)
        @test all(
            i -> !isfinite(data.infectious_time[i]) ||
                 0.4 <= data.infectious_time[i] - data.infection_time[i] <= 0.8,
            eachindex(data.infection_time))

        expected = 0.0
        for j in eachindex(data.infection_time)
            tj = isnan(data.infection_time[j]) ? Inf : data.infection_time[j]
            event_hazard = 0.0
            for i in eachindex(data.infection_time)
                i == j && continue
                opening = data.infectious_time[i]
                isfinite(opening) || continue
                stop = min(tj, data.removal_time[i])
                stop > opening && (expected -= (stop^2 - opening^2) / 9)
                opening < tj <= data.removal_time[i] && (event_hazard += 2tj / 9)
            end
            isfinite(tj) && !data.is_index[j] && (expected += log(event_hazard))
        end
        @test loglikelihood(data, spec) ≈ expected
        @test pairwise_surv_loglik(CalendarKernel(calendar), data,
            compile_contact_pairs(data)) ≈ expected

        # Freezing the observed openings reproduces the same interval kernels.
        frozen(i, j) = truncated(calendar; lower = data.infectious_time[i]) -
                       data.infectious_time[i]
        @test loglikelihood(data, ModelSpec(make_process(frozen); progression)) ≈ expected
        id_calendar = CalendarKernel((i, j) -> calendar)
        @test loglikelihood(data, ModelSpec(make_process(id_calendar); progression)) ≈
              expected
    end
end
