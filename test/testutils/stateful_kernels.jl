# The second case records a future intervention on a different, pending host.
struct RecordKernelPolicy <: AbstractIntervention end
function EpiBranch.resolve_individual!(::RecordKernelPolicy, ind, state)
    ind.id == 2 || return nothing
    state.individuals[3].state[:policy_time] = ind.infection_time + 0.5
    return nothing
end

function state_policy_law(before, after, date)
    isfinite(date) || return Exponential(inv(before))
    survival = exp(-before * date)
    MixtureModel(
        [truncated(Exponential(inv(before)); upper = date),
            date + Exponential(inv(after))],
        [1 - survival, survival])
end

struct WaitForKernelDay <: AbstractIntervention end
function EpiBranch.competing_risk(::WaitForKernelDay, parent, contact, state)
    Risk(block_probability = state.max_infection_time < 1.0 ? 1.0 : 0.0)
end

function test_stateful_simulation(make_process, extract)
    @testset "Simultaneous contacts survive refresh" begin
        kernel = StatefulKernel(_ -> nothing, (c, a, b) -> Dirac(1.0))
        model = ModelSpec(make_process(kernel);
            progression = [Transition(:recovered; delay = 5.0, terminal = true)])
        state = simulate(model; initial_cases = [1], rng = StableRNG(1))
        @test [i.infection_time for i in state.individuals] == [0.0, 1.0, 1.0]
    end
    @testset "Sampled attributes in pair kernels" begin
        project(ind) = (scale = ind.state[:sampled_scale]::Float64,)
        callback(c, a, b) = Exponential(a.scale + b.scale)
        live = StatefulKernel(project, callback)
        attributes = (rng, ind) -> (ind.state[:sampled_scale] = rand(rng, Uniform(0.5, 1.5)))
        progression = [Transition(:recovered; delay = 5.0, terminal = true)]
        spec = ModelSpec(make_process(live); attributes, progression)
        state = simulate(spec; initial_cases = [1], rng = StableRNG(235))
        records = record_kernel(live, state)
        data = extract(state, spec)
        @test all(r -> 0.5 <= r.scale <= 1.5, records.state)
        expected(i, j) = Exponential(records.state[i].scale + records.state[j].scale)
        @test pairwise_surv_loglik(records, data) ≈ pairwise_surv_loglik(expected, data)
        @test loglikelihood(data, make_process(records)) ≈
              pairwise_surv_loglik(expected, data)
        @test_throws ArgumentError loglikelihood(data, make_process(live))
        # A recorded vector can also drive simulation without live-state reads.
        a = simulate(ModelSpec(make_process(records); progression); initial_cases = [1], rng = StableRNG(9))
        b = simulate(ModelSpec(make_process(expected); progression); initial_cases = [1], rng = StableRNG(9))
        @test isequal([i.infection_time for i in a.individuals], [i.infection_time
                                                                  for i in b.individuals])
    end
    @testset "Recorded endogenous histories reproduce integrated hazards" begin
        project(ind) = (date = get(ind.state, :policy_time, Inf)::Float64,)
        kernel = CalendarKernel(StatefulKernel(project,
            (c, a, b) -> state_policy_law(0.4, 0.1, b.date)))
        model = ModelSpec(make_process(kernel);
            progression = [Transition(:infectious; delay = 0.3),
                Transition(:recovered; from = :infectious, delay = 5.0, terminal = true)],
            interventions = [RecordKernelPolicy()])
        observed_policy = false
        for seed in 1:20
            state = simulate(model; initial_cases = [1], rng = StableRNG(seed))
            data = extract(state, model)
            recorded = record_kernel(kernel, state)
            dates = [r.date for r in recorded.kernel.state]
            observed_policy |= isfinite(dates[3])
            expected = 0.0
            for j in eachindex(data.infection_time)
                tj = isnan(data.infection_time[j]) ? Inf : data.infection_time[j]
                hazard = 0.0
                for i in eachindex(data.infection_time)
                    i == j && continue
                    opening = data.infectious_time[i]
                    isfinite(opening) || continue
                    stop = min(tj, data.removal_time[i])
                    date = dates[j]
                    if stop > opening
                        expected -= 0.4 * (min(stop, date) - min(opening, date)) +
                                    0.1 * (max(stop - date, 0.0) - max(opening - date, 0.0))
                    end
                    opening < tj <= data.removal_time[i] &&
                        (hazard += tj < date ? 0.4 : 0.1)
                end
                isfinite(tj) && !data.is_index[j] && (expected += log(hazard))
            end
            @test pairwise_surv_loglik(recorded, data) ≈ expected
        end
        @test observed_policy
    end
    @testset "Refresh preserves blocked external introductions" begin
        live = StatefulKernel(_ -> nothing, (c, a, b) -> Dirac(20.0))
        process = make_process(live; external_hazard = 0.8, obs_end = 3.0)
        model = ModelSpec(process;
            progression = [Transition(:recovered; delay = 4.0, terminal = true)],
            interventions = [WaitForKernelDay()])
        infected = 0
        for seed in 1:500
            state = simulate(model; rng = StableRNG(seed))
            cases = filter(is_infected, state.individuals)
            @test all(i -> 1.0 <= i.infection_time <= 3.0, cases)
            infected += length(cases)
        end
        @test infected / 1500 ≈ 1 - exp(-0.8 * 2.0) atol = 0.04
    end
    @testset "Pending contacts reflect later recorded actions" begin
        project(ind) = (policy_time = get(ind.state, :policy_time, Inf)::Float64,)
        callback = function (c, a, b)
            (c.infector, c.susceptible) == (1, 2) && return Dirac(1.0)
            (c.infector, c.susceptible) == (1, 3) &&
                return state_policy_law(0.1, 1.0, b.policy_time)
            return Dirac(20.0)
        end
        kernel = StatefulKernel(project, callback)
        spec = ModelSpec(make_process(kernel);
            progression = [Transition(:recovered; delay = 5.0, terminal = true)],
            interventions = [RecordKernelPolicy()])
        n = 1500
        by_three = 0
        by_policy = 0
        for seed in 1:n
            state = simulate(spec; initial_cases = [1], rng = StableRNG(seed))
            @test state.individuals[2].infection_time == 1.0
            records = record_kernel(kernel, state)
            @test records.state[3].policy_time == 1.5
            t = state.individuals[3].infection_time
            by_three += is_infected(state.individuals[3]) && t <= 3.0
            by_policy += is_infected(state.individuals[3]) && t <= 1.5
        end
        @test by_three / n ≈ 1 - exp(-0.1 * 1.5 - 1.0 * 1.5) atol = 0.04
        @test by_policy / n ≈ 1 - exp(-0.1 * 1.5) atol = 0.03
    end
end
