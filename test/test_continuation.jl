using Distributions
using Statistics
using StableRNGs

# A tiny observation model that only counts how often it ran, to check that
# a continued run never applies it (see "no observation on continuation"
# below).
struct CountingObservation <: EpiBranch.ObservationModel
    counter::Vector{Int}
end
CountingObservation() = CountingObservation([0])
function EpiBranch.apply_observation!(o::CountingObservation, state, rng)
    o.counter[1] += 1
    return state
end

# A route that proposes its own contacts (`proposes_own_contacts`) and can
# leave some pending past a time limit (`pending_beyond`), exactly as
# `EpiBranch._MassAction` does, but without `EpiBranch._resume_route!`: a
# third-party route that implements the public seam without knowing it also
# needs this private one (see "a continuation whose route cannot resume is
# refused" below). It forwards every call to a real `_MassAction`.
struct _UnresumableMassAction
    inner::EpiBranch._MassAction
end
(r::_UnresumableMassAction)(infector_id, state) = r.inner(infector_id, state)
EpiBranch.proposes_own_contacts(::_UnresumableMassAction) = true
EpiBranch.route_pair_kernel(r::_UnresumableMassAction, infector_id, target_id, state) =
    EpiBranch.route_pair_kernel(r.inner, infector_id, target_id, state)
function EpiBranch.propose_along(
        r::_UnresumableMassAction, infector_id, state, open_t, close_t, capped, traits
    )
    return EpiBranch.propose_along(r.inner, infector_id, state, open_t, close_t, capped, traits)
end
EpiBranch.pending_beyond(r::_UnresumableMassAction, state, reinfects::Bool) =
    EpiBranch.pending_beyond(r.inner, state, reinfects)

# A homogeneous population whose infectious window never closes (`until =
# ()`), raced over `_UnresumableMassAction` instead of mass action directly.
struct NoResumeHomogeneous <: EpiBranch.TransmissionModel
    population_size::Int
    transmission_rate::Float64
end
EpiBranch.population_size(m::NoResumeHomogeneous) = m.population_size
function EpiBranch.simulate_once(
        model::NoResumeHomogeneous, sim_opts::EpiBranch.SimOpts;
        interventions, attributes, progression, observation, recorder, rng
    )
    N = model.population_size
    β = model.transmission_rate
    state = EpiBranch.new_state(model, progression, attributes, rng)
    EpiBranch.add_individuals!(state, N, interventions)
    members = collect(1:N)
    EpiBranch.sellke_race!(
        state, members, rng, sim_opts;
        seed! = (best, _, _) -> (best[1:(sim_opts.n_initial)] .= 0),
        targets = _UnresumableMassAction(
            EpiBranch._MassAction(state, members; rate = (_, _) -> β / N)
        ),
        from = EpiBranch.infectious_from(progression), until = (), interventions,
        risks = EpiBranch.transmission_risks(model), recorder
    )
    EpiBranch.apply_observation!(observation, state, rng)
    return state
end

@testset "simulate(model, state; max_time) continuation" begin
    @testset "continuing a closing window agrees bit for bit with one run" begin
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 2000);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)]
        )
        state = simulate(model; max_time = 5.0, n_initial = 5, rng = StableRNG(1))
        @test !state.extinct
        continued = simulate(model, state; max_time = 30.0)
        direct = simulate(model; max_time = 30.0, n_initial = 5, rng = StableRNG(1))
        @test continued === state
        @test continued.cumulative_cases == direct.cumulative_cases
        @test all(
            isequal(a.infection_time, b.infection_time) for
                (a, b) in zip(continued.individuals, direct.individuals)
        )
    end

    @testset "chained continuations agree with one run straight through" begin
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 2000);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)]
        )
        chained = simulate(model; max_time = 5.0, n_initial = 5, rng = StableRNG(7))
        chained = simulate(model, chained; max_time = 15.0)
        chained = simulate(model, chained; max_time = 30.0)
        direct = simulate(model; max_time = 30.0, n_initial = 5, rng = StableRNG(7))
        @test chained.cumulative_cases == direct.cumulative_cases
        @test all(
            isequal(a.infection_time, b.infection_time) for
                (a, b) in zip(chained.individuals, direct.individuals)
        )
    end

    @testset "a bare process continues the same way" begin
        # No progression at all, so (as for the never-closing window below)
        # mass action never finishes sampling a window in one call; agreement
        # is in distribution rather than bit for bit.
        process = HomogeneousProcess(transmission_rate = 0.4, population_size = 1500)
        final_size(seed; split) = begin
            if split === nothing
                simulate(process; max_time = 15.0, n_initial = 5, rng = StableRNG(seed)).cumulative_cases
            else
                state = simulate(process; max_time = split, n_initial = 5, rng = StableRNG(seed))
                simulate(process, state; max_time = 15.0).cumulative_cases
            end
        end
        direct = [final_size(s; split = nothing) for s in 1:150]
        continued = [final_size(s; split = 3.0) for s in 1:150]
        @test isapprox(mean(direct), mean(continued); rtol = 0.1)
    end

    @testset "mass action with a window that never closes agrees in distribution" begin
        # No `until` ever reached: the infectious window never closes, so
        # mass action samples contacts only up to the race's time limit and
        # must redraw the cut pairs on continuation.
        process = HomogeneousProcess(transmission_rate = 0.3, population_size = 1500)
        final_size(seed; split) = begin
            if split === nothing
                simulate(process; max_time = 20.0, n_initial = 5, rng = StableRNG(seed)).cumulative_cases
            else
                state = simulate(process; max_time = split, n_initial = 5, rng = StableRNG(seed))
                simulate(process, state; max_time = 20.0).cumulative_cases
            end
        end
        direct = [final_size(s; split = nothing) for s in 1:150]
        continued = [final_size(s; split = 4.0) for s in 1:150]
        @test isapprox(mean(direct), mean(continued); rtol = 0.1)
    end

    @testset "a branching process continues the same way" begin
        model = ModelSpec(BranchingProcess(NegBin(3.0, 0.15), Gamma(2.5, 2.0)))
        state = simulate(model; max_time = 5.0, max_cases = 10_000, rng = StableRNG(6))
        continued = simulate(model, state; max_time = 20.0)
        direct = simulate(model; max_time = 20.0, max_cases = 10_000, rng = StableRNG(6))
        @test length(continued.individuals) == length(direct.individuals)
        @test continued.cumulative_cases == direct.cumulative_cases
        @test all(
            isequal(a.infection_time, b.infection_time) for
                (a, b) in zip(continued.individuals, direct.individuals)
        )
    end

    @testset "continuing a state that already reached extinction is a no-op" begin
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 20);
            progression = [Transition(:recovered; rate = 5.0, terminal = true)]
        )
        state = simulate(model; max_time = 50.0, rng = StableRNG(3))
        @test state.extinct
        continued = simulate(model, state; max_time = 100.0)
        @test continued === state
    end

    @testset "a stop by a stopping rule or max_generation is not continued" begin
        # `MaxCases` settles its triggering case before it stops the race. The
        # leftover queue still has further events pending; popping them would
        # let more cases through the cap. Resuming such a state must leave it
        # alone, not pop its queue further.
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 2.0, population_size = 500);
            progression = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        )
        capped_cases = simulate(
            model; n_initial = 3,
            stopping_rules = [Extinction(), MaxCases(5)], rng = StableRNG(4)
        )
        @test capped_cases.cumulative_cases == 5
        @test !is_extinct(capped_cases)
        continued_cases = simulate(model, capped_cases; max_time = 1000.0)
        @test continued_cases === capped_cases
        @test continued_cases.cumulative_cases == 5

        capped_generations = simulate(
            model; n_initial = 3,
            stopping_rules = [Extinction(), MaxGenerations(3)], rng = StableRNG(4)
        )
        @test !is_extinct(capped_generations)
        before = capped_generations.cumulative_cases
        continued_generations = simulate(model, capped_generations; max_time = 1000.0)
        @test continued_generations === capped_generations
        @test continued_generations.cumulative_cases == before
    end

    @testset "a stopping-rule or max_generation stop still returns buffers to the pool" begin
        # A continuation built for such a stop can never be resumed (see
        # above). Keeping it would withhold its buffers from the task-local
        # pool forever, instead of returning them as an extinct run does.
        # Each check runs in its own task for a pool of its own.
        pool_size(T) = begin
            pool = get(task_local_storage(), T, nothing)
            pool === nothing ? 0 : length(pool)
        end
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 2.0, population_size = 500);
            progression = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        )
        extinct_buffers = fetch(
            Threads.@spawn begin
                state = simulate(model; n_initial = 3, rng = StableRNG(4))
                @assert is_extinct(state)
                pool_size(EpiBranch._RaceBuffers{Float64})
            end
        )
        capped_buffers = fetch(
            Threads.@spawn begin
                state = simulate(
                    model; n_initial = 3,
                    stopping_rules = [Extinction(), MaxCases(5)], rng = StableRNG(4)
                )
                @assert !is_extinct(state)
                pool_size(EpiBranch._RaceBuffers{Float64})
            end
        )
        @test capped_buffers == extinct_buffers

        branching_model = ModelSpec(BranchingProcess(NegBin(3.0, 0.15), Gamma(2.5, 2.0)))
        extinct_tree = fetch(
            Threads.@spawn begin
                state = simulate(
                    branching_model;
                    stopping_rules = [Extinction(), MaxCases(5)], rng = StableRNG(1)
                )
                @assert is_extinct(state)
                pool_size(EpiBranch._Tree{Float64})
            end
        )
        capped_tree = fetch(
            Threads.@spawn begin
                state = simulate(
                    branching_model;
                    stopping_rules = [Extinction(), MaxCases(5)], rng = StableRNG(3)
                )
                @assert !is_extinct(state)
                pool_size(EpiBranch._Tree{Float64})
            end
        )
        @test capped_tree == extinct_tree
    end

    @testset "max_time must move past the state's own" begin
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 500);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)]
        )
        state = simulate(model; max_time = 5.0, n_initial = 5, rng = StableRNG(4))
        @test !state.extinct
        @test_throws ArgumentError simulate(model, state; max_time = 5.0)
        @test_throws ArgumentError simulate(model, state; max_time = 2.0)
    end

    @testset "a continuation that changes may_block is refused" begin
        # The first call has no intervention that can block a proposal; the
        # second adds a leaky `Isolation`, which defines `competing_risk` and
        # turns on the blocking that sizes the race's buffers. Resuming with
        # it must be refused rather than index buffers sized for the first
        # call.
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 2000);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)]
        )
        blocking_model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 2000);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)],
            interventions = [
                Isolation(
                    onset_to_isolation_delay = 1.0, duration = Inf,
                    post_isolation_transmission = 0.5
                ),
            ]
        )
        state = simulate(model; max_time = 5.0, n_initial = 5, rng = StableRNG(1))
        @test !state.extinct
        @test_throws ArgumentError simulate(blocking_model, state; max_time = 30.0)
    end

    @testset "a continuation whose route cannot resume is refused" begin
        # `_UnresumableMassAction` reports a pending window just as mass
        # action does, but has no `_resume_route!` to redraw it: the run that
        # would otherwise need continuing is refused up front, rather than
        # silently dropping those contacts on a later `simulate(model, state;
        # max_time)` call.
        model = ModelSpec(NoResumeHomogeneous(500, 2.0))
        @test_throws ArgumentError simulate(
            model; max_time = 0.01, n_initial = 3, rng = StableRNG(1)
        )
    end

    @testset "no observation on continuation" begin
        counter = CountingObservation()
        model = ModelSpec(
            HomogeneousProcess(transmission_rate = 0.4, population_size = 500);
            progression = [Transition(:recovered; rate = 0.2, terminal = true)],
            observation = counter
        )
        state = simulate(model; max_time = 5.0, n_initial = 5, rng = StableRNG(5))
        @test !state.extinct
        @test counter.counter[1] == 1
        simulate(model, state; max_time = 30.0)
        @test counter.counter[1] == 1
    end
end
