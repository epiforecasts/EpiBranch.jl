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
    return MixtureModel(
        [
            truncated(Exponential(inv(before)); upper = date),
            date + Exponential(inv(after)),
        ],
        [1 - survival, survival]
    )
end

struct WaitForKernelDay <: AbstractIntervention end
function EpiBranch.competing_risk(::WaitForKernelDay, parent, contact, state)
    return Risk(block_probability = state.max_infection_time < 1.0 ? 1.0 : 0.0)
end

# Moves every host's record on every case, so the race redraws pending contacts
# at each step. A kernel whose records never move takes the ordinary path, which
# is the point of the equivalence test below, so a test about redrawing has to
# make something move.
struct TickEveryCase <: AbstractIntervention end
function EpiBranch.resolve_individual!(::TickEveryCase, ind, state)
    for person in state.individuals
        person.state[:tick] = state.cumulative_cases
    end
    return nothing
end
tick_state(ind) = (tick = get(ind.state, :tick, 0)::Int,)

# Gives every host a dose date at the first case, so a record that read
# `nothing` before then reads a number afterwards.
struct DoseEveryone <: AbstractIntervention end
function EpiBranch.resolve_individual!(::DoseEveryone, ind, state)
    for person in state.individuals
        get!(person.state, :dose_time, ind.infection_time + 0.5)
    end
    return nothing
end

# From the second case on, gives every host a dose dated at that case's
# infection time, and blocks every contact from host 1 to host 3.
struct DoseAtSecondCase <: AbstractIntervention end
function EpiBranch.resolve_individual!(::DoseAtSecondCase, ind, state)
    ind.id == 1 && return nothing
    for person in state.individuals
        get!(person.state, :dose_time, ind.infection_time)
    end
    return nothing
end
struct BlockOneToThree <: AbstractIntervention end
function EpiBranch.competing_risk(::BlockOneToThree, parent, contact, state)
    return Risk(block_probability = (parent.id, contact.id) == (1, 3) ? 1.0 : 0.0)
end

# Blocks half the contacts from host 1 to host 2.
struct HalfBlockOneToTwo <: AbstractIntervention end
function EpiBranch.competing_risk(::HalfBlockOneToTwo, parent, contact, state)
    return Risk(block_probability = (parent.id, contact.id) == (1, 2) ? 0.5 : 0.0)
end

# Stamps only the case being settled, which no pending contact reads.
struct StampOwnCase <: AbstractIntervention end
function EpiBranch.resolve_individual!(::StampOwnCase, ind, state)
    ind.state[:stamp] = ind.infection_time + 1.0
    return nothing
end

function test_stateful_simulation(make_process, extract)
    @testset "An unchanging live kernel races as an ordinary one" begin
        # Redrawing exists for hazards that move. A kernel whose records never
        # move must leave the race exactly where an ordinary kernel leaves it,
        # random stream included: a refresh that runs anyway is both wasted
        # and free to change the answer without any test noticing.
        project(ind) = (tag = get(ind.state, :tag, 0.0)::Float64,)
        progression = [Transition(:recovered; delay = 3.0, terminal = true)]
        for d in (Exponential(1.5), Weibull(2.0, 2.0), Gamma(3.0, 0.7))
            ordinary = ModelSpec(make_process(d); progression)
            stateful = ModelSpec(
                make_process(StatefulKernel(project, (c, a, b) -> d));
                progression
            )
            for seed in 1:25
                a = simulate(ordinary; initial_cases = [1], rng = StableRNG(seed))
                b = simulate(stateful; initial_cases = [1], rng = StableRNG(seed))
                @test isequal(
                    [i.infection_time for i in a.individuals],
                    [i.infection_time for i in b.individuals]
                )
            end
        end
    end
    @testset "A case moving its own record races as an ordinary kernel" begin
        project(ind) = (stamp = get(ind.state, :stamp, Inf)::Float64,)
        progression = [Transition(:recovered; delay = 3.0, terminal = true)]
        d = Exponential(1.5)
        ordinary = ModelSpec(
            make_process(d); progression,
            interventions = [StampOwnCase()]
        )
        stateful = ModelSpec(
            make_process(StatefulKernel(project, (c, a, b) -> d));
            progression, interventions = [StampOwnCase()]
        )
        for seed in 1:25
            a = simulate(ordinary; initial_cases = [1], rng = StableRNG(seed))
            b = simulate(stateful; initial_cases = [1], rng = StableRNG(seed))
            @test isequal(
                [i.infection_time for i in a.individuals],
                [i.infection_time for i in b.individuals]
            )
        end
    end
    @testset "A record may change type during a run" begin
        project(ind) = (dose = get(ind.state, :dose_time, nothing),)
        kernel = StatefulKernel(
            project,
            (c, a, b) -> Exponential(b.dose === nothing ? 1.0 : 2.0)
        )
        model = ModelSpec(
            make_process(kernel);
            progression = [Transition(:recovered; delay = 3.0, terminal = true)],
            interventions = [DoseEveryone()]
        )
        state = simulate(model; initial_cases = [1], rng = StableRNG(4))
        @test all(i -> i.state[:dose_time] isa Float64, state.individuals)
    end
    @testset "A record change governs a contact due at the same clock" begin
        # Host 3's dose at t = 1 moves its contacts from one day after an
        # infector's infection to two, including the contact from host 1 due at
        # t = 1 itself. That contact is drawn again at t = 2 and blocked; host
        # 2's contact under the new hazard infects host 3 at t = 3.
        project(ind) = (dose = get(ind.state, :dose_time, Inf)::Float64,)
        callback(c, a, b) = b.dose <= c.infector_infection_time + 1.0 ? Dirac(2.0) :
            Dirac(1.0)
        model = ModelSpec(
            make_process(StatefulKernel(project, callback));
            progression = [Transition(:recovered; delay = 4.0, terminal = true)],
            interventions = [DoseAtSecondCase(), BlockOneToThree()]
        )
        state = simulate(model; initial_cases = [1], rng = StableRNG(1))
        @test [i.infection_time for i in state.individuals] == [0.0, 1.0, 3.0]
        @test state.individuals[3].parent_id == 2
    end
    @testset "A moved record keeps an atom at the current clock" begin
        # Contacts at one and three days are equally likely. Stamping a date the
        # kernel ignores leaves the hazard unchanged, so a contact due at the
        # clock where a record moves must still be possible then.
        project(ind) = (dose = get(ind.state, :dose_time, Inf)::Float64,)
        delay = DiscreteNonParametric([1.0, 3.0], [0.5, 0.5])
        progression = [Transition(:recovered; delay = 5.0, terminal = true)]
        live = ModelSpec(
            make_process(StatefulKernel(project, (c, a, b) -> delay));
            progression, interventions = [DoseAtSecondCase()]
        )
        n = 4000
        early = count(1:n) do seed
            state = simulate(live; initial_cases = [1], rng = StableRNG(seed))
            state.individuals[3].infection_time == 1.0
        end
        @test isapprox(early / n, 0.5; atol = 0.03)
    end
    @testset "A moved record leaves a resolved atom resolved" begin
        # Host 2 comes before host 3 in the race, so by the time host 3 settles
        # at t = 1 every contact to host 2 at t = 1 has been resolved: a draw
        # that fell later, or a contact that was blocked. Moving records then
        # must not offer host 2 the atom at t = 1 again.
        progression = [Transition(:recovered; delay = 5.0, terminal = true)]
        n = 2000
        at_one(model) = count(1:n) do seed
            state = simulate(model; initial_cases = [1], rng = StableRNG(seed))
            state.individuals[2].infection_time == 1.0
        end / n
        two_atoms = DiscreteNonParametric([1.0, 2.0], [0.5, 0.5])
        kernel = StatefulKernel(
            tick_state,
            (c, a, b) -> c.susceptible == 2 ? two_atoms : Dirac(1.0)
        )
        drawn_later = ModelSpec(
            make_process(kernel); progression,
            interventions = [TickEveryCase()]
        )
        @test isapprox(at_one(drawn_later), 0.5; atol = 0.04)
        blocked = ModelSpec(
            make_process(StatefulKernel(tick_state, (c, a, b) -> Dirac(1.0)));
            progression, interventions = [HalfBlockOneToTwo(), TickEveryCase()]
        )
        @test isapprox(at_one(blocked), 0.5; atol = 0.04)
    end
    @testset "Simultaneous contacts survive refresh" begin
        kernel = StatefulKernel(tick_state, (c, a, b) -> Dirac(1.0))
        model = ModelSpec(
            make_process(kernel);
            progression = [Transition(:recovered; delay = 5.0, terminal = true)],
            interventions = [TickEveryCase()]
        )
        state = simulate(model; initial_cases = [1], rng = StableRNG(1))
        @test [i.infection_time for i in state.individuals] == [0.0, 1.0, 1.0]
    end
    @testset "A live kernel timed from onset serves simulation and likelihood" begin
        # Infectiousness starts at the infector's symptom onset. The same
        # projection reads the onset from an individual in simulation and from
        # the infection layer's host times in the likelihood.
        project(ind) = (onset = get(ind.state, :onset_time, NaN),)
        callback(c, a, b) = (a.onset - c.infector_infection_time) + Exponential(1.0)
        live = StatefulKernel(project, callback)
        model = ModelSpec(
            make_process(live);
            attributes = clinical_presentation(incubation_period = Gamma(2.0, 1.0)),
            progression = [Transition(:recovered; delay = 6.0, terminal = true)]
        )
        state = simulate(model; initial_cases = [1], rng = StableRNG(7))
        infected = filter(is_infected, state.individuals)
        @test length(infected) > 1
        @test all(infected) do ind
            ind.parent_id == 0 ||
                ind.infection_time >= state.individuals[ind.parent_id].state[:onset_time]
        end
        data = extract(state, model; host_times = (:onset_time,))
        onsets = data.host_times.onset_time
        expected(i, j) = (onsets[i] - data.infection_time[i]) + Exponential(1.0)
        @test loglikelihood(data, make_process(live)) ≈ pairwise_surv_loglik(expected, data)
        # Without the host times the likelihood cannot read the onsets, and it
        # refuses a projection reading a time the layer did not record.
        @test_throws ArgumentError loglikelihood(extract(state, model), make_process(live))
        wider(ind) = (
            onset = get(ind.state, :onset_time, NaN),
            traced = get(ind.state, :trace_time, Inf),
        )
        wider_kernel = StatefulKernel(wider, callback)
        @test_throws ArgumentError loglikelihood(data, make_process(wider_kernel))
        # A time no host holds, as when a policy never triggered, is recorded as
        # absent everywhere, and a projection reading it falls back to its default.
        unset = extract(state, model; host_times = (:onset_time, :policy_time))
        @test all(ismissing, unset.host_times.policy_time)
        policy(ind) = (
            onset = get(ind.state, :onset_time, NaN),
            date = get(ind.state, :policy_time, Inf),
        )
        policy_kernel = StatefulKernel(policy, callback)
        @test loglikelihood(unset, make_process(policy_kernel)) ≈
            loglikelihood(data, make_process(live))
        # A misspelt key leaves the key the projection reads unrecorded.
        misspelt = extract(state, model; host_times = (:onset,))
        @test_throws ArgumentError loglikelihood(misspelt, make_process(live))
        # A projection reading only the host's id needs no host times.
        scales = [1.0, 2.0, 0.5]
        by_id = StatefulKernel(ind -> (s = scales[ind.id],), (c, a, b) -> Exponential(a.s))
        plain = extract(state, model)
        @test loglikelihood(plain, make_process(by_id)) ≈
            pairwise_surv_loglik((i, j) -> Exponential(scales[i]), plain)
        # An asymptomatic case stores a NaN onset, which the layer holds as a value.
        # A projection reading it as such evaluates the same as recorded records.
        stored(ind) = (onset = ind.state[:onset_time]::Float64,)
        branch(c, a, b) = isnan(a.onset) ? Exponential(2.0) : callback(c, a, b)
        asym_kernel = StatefulKernel(stored, branch)
        asym_model = ModelSpec(
            make_process(asym_kernel);
            attributes = clinical_presentation(
                incubation_period = Gamma(2.0, 1.0),
                prob_asymptomatic = 1.0
            ),
            progression = [Transition(:recovered; delay = 6.0, terminal = true)]
        )
        asym_state = simulate(asym_model; initial_cases = [1], rng = StableRNG(7))
        @test any(
            ind -> is_infected(ind) && isnan(ind.state[:onset_time]),
            asym_state.individuals
        )
        asym_data = extract(asym_state, asym_model; host_times = (:onset_time,))
        @test loglikelihood(asym_data, make_process(asym_kernel)) ≈
            pairwise_surv_loglik(record_kernel(asym_kernel, asym_state), asym_data)
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
        @test isequal(
            [i.infection_time for i in a.individuals], [
                i.infection_time
                    for i in b.individuals
            ]
        )
    end
    @testset "Recorded endogenous histories reproduce integrated hazards" begin
        project(ind) = (date = get(ind.state, :policy_time, Inf)::Float64,)
        kernel = CalendarKernel(
            StatefulKernel(
                project,
                (c, a, b) -> state_policy_law(0.4, 0.1, b.date)
            )
        )
        model = ModelSpec(
            make_process(kernel);
            progression = [
                Transition(:infectious; delay = 0.3),
                Transition(:recovered; from = :infectious, delay = 5.0, terminal = true),
            ],
            interventions = [RecordKernelPolicy()]
        )
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
        live = StatefulKernel(tick_state, (c, a, b) -> Dirac(20.0))
        process = make_process(live; external_hazard = 0.8, obs_end = 3.0)
        model = ModelSpec(
            process;
            progression = [Transition(:recovered; delay = 4.0, terminal = true)],
            interventions = [WaitForKernelDay(), TickEveryCase()]
        )
        infected = 0
        for seed in 1:500
            state = simulate(model; rng = StableRNG(seed))
            cases = filter(is_infected, state.individuals)
            @test all(i -> 1.0 <= i.infection_time <= 3.0, cases)
            infected += length(cases)
        end
        @test infected / 1500 ≈ 1 - exp(-0.8 * 2.0) atol = 0.04
    end
    return @testset "Pending contacts reflect later recorded actions" begin
        project(ind) = (policy_time = get(ind.state, :policy_time, Inf)::Float64,)
        callback = function (c, a, b)
            (c.infector, c.susceptible) == (1, 2) && return Dirac(1.0)
            (c.infector, c.susceptible) == (1, 3) &&
                return state_policy_law(0.1, 1.0, b.policy_time)
            return Dirac(20.0)
        end
        kernel = StatefulKernel(project, callback)
        spec = ModelSpec(
            make_process(kernel);
            progression = [Transition(:recovered; delay = 5.0, terminal = true)],
            interventions = [RecordKernelPolicy()]
        )
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
