using DataFrames
using Dates

@testset "Line list output" begin
    clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

    @testset "linelist basic output" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = tsim(model; attributes = clinical,
            max_cases = 50, rng = rng)

        df = linelist(state)

        @test df isa DataFrame
        @test nrow(df) == state.cumulative_cases
        @test "id" in names(df)
        @test "parent_id" in names(df)
        @test "date_infection" in names(df)
        @test "date_onset" in names(df)
    end

    @testset "linelist without clinical presentation has no onset" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = tsim(model; max_cases = 50, rng = rng)

        df = linelist(state)
        @test "id" in names(df)
        @test !("date_onset" in names(df))
        @test !("age" in names(df))
    end

    @testset "linelist projects transition state" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        transitions = [
            Reporting(delay = Exponential(3.0)),
            Hospitalisation(delay = Exponential(5.0), probability = 0.3),
            Death(delay = Exponential(10.0), probability = 0.05),
            Recovery(delay = Exponential(10.0))
        ]
        state = tsim(model; attributes = clinical, transitions = transitions,
            max_cases = 100, rng = rng)

        df = linelist(state)
        @test "date_reporting" in names(df)
        @test "date_admission" in names(df)
        @test "outcome" in names(df)
        @test "date_outcome" in names(df)
        # Every symptomatic case has an outcome label from the terminal pair.
        for row in eachrow(df)
            !ismissing(row.date_onset) || continue
            @test !ismissing(row.outcome)
            @test row.outcome in ("died", "recovered")
        end
        # Admission and reporting times sit at or after onset (per transition spec).
        for row in eachrow(df)
            ismissing(row.date_onset) && continue
            ismissing(row.date_reporting) || @test row.date_reporting >= row.date_onset
            ismissing(row.date_admission) || @test row.date_admission >= row.date_onset
        end
    end

    @testset "linelist with demographics from attributes" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        attrs = [
            clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
            demographics(age_distribution = Normal(40, 15))
        ]
        state = tsim(model; attributes = attrs,
            max_cases = 50, rng = rng)

        df = linelist(state)
        @test "age" in names(df)
        @test "sex" in names(df)
        @test !ismissing(df.age[1])
    end

    @testset "linelist with age-conditional CFR via Death callable" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        attrs = [
            clinical_presentation(incubation_period = LogNormal(1.5, 0.5)),
            demographics(age_distribution = Uniform(0, 90))
        ]
        # CFR depends on age via a Death probability closure.
        death = Death(delay = Exponential(10.0),
            probability = (rng, ind) -> ind.state[:age] >= 65 ? 0.5 : 0.01)
        recovery = Recovery(delay = Exponential(10.0))
        state = tsim(model;
            condition = 50:500, attributes = attrs,
            transitions = [death, recovery],
            max_cases = 500, rng = rng)

        df = linelist(state)
        @test any(df.outcome .== "died")
        @test any(df.outcome .== "recovered")
        # Deaths are concentrated in the older band by construction.
        old_deaths = count(
            row -> !ismissing(row.outcome) && row.outcome == "died" &&
                   !ismissing(row.age) && row.age >= 65,
            eachrow(df))
        young_deaths = count(
            row -> !ismissing(row.outcome) && row.outcome == "died" &&
                   !ismissing(row.age) && row.age < 65,
            eachrow(df))
        @test old_deaths > young_deaths
    end

    @testset "linelist empty state" begin
        empty_state = SimulationState(Individual{Float64}[], Int[], 0, StableRNG(1),
            0, true, nothing, 0.0, nothing, AbstractClinicalTransition[])
        df = linelist(empty_state)
        @test nrow(df) == 0
    end

    @testset "linelist includes intervention state" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0))
        state = tsim(model;
            interventions = [iso], attributes = clinical,
            max_cases = 50, rng = rng)

        df = linelist(state)
        @test "isolated" in names(df)
    end

    @testset "contacts output" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(2.0), Exponential(5.0))
        state = tsim(model; max_cases = 50, rng = rng)

        df = contacts(state)
        @test df isa DataFrame
        @test "from" in names(df)
        @test "to" in names(df)
        @test "infected" in names(df)
    end

    @testset "index cases identifiable via parent_id" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        state = tsim(model; attributes = clinical,
            max_cases = 20, n_initial = 3, rng = rng)

        df = linelist(state)
        n_index = count(df.parent_id .== 0)
        @test n_index == 3
    end

    @testset "linelist infected_only=false includes the whole population" begin
        N = 200
        m = ModelSpec(
            HomogeneousProcess(; transmission_rate = 0.5, population_size = N);
            progression = [Transition(:recovered; from = :infection, delay = 1.0,
                terminal = true)])
        state = simulate(m; rng = StableRNG(1), n_initial = 1)
        @test state.cumulative_cases < N   # a sub-critical outbreak leaves survivors

        df = linelist(state; infected_only = false)
        @test nrow(df) == N
        @test "infected" in names(df)
        @test count(df.infected) == state.cumulative_cases

        # Never-infected individuals have no infection date.
        never_infected = filter(row -> !row.infected, df)
        @test all(ismissing, never_infected.date_infection)

        infected_rows = filter(row -> row.infected, df)
        @test all(!ismissing, infected_rows.date_infection)

        # The default returns infected cases only.
        @test nrow(linelist(state)) == state.cumulative_cases
    end

    @testset "linelist infected_only=false on a branching process with interventions" begin
        # Contacts exposed but not infected keep their exposure time and the
        # onset derived from it in state; the table must not report either.
        spec = ModelSpec(
            BranchingProcess(Poisson(2.5), Exponential(5.0); population_size = 400);
            attributes = clinical,
            interventions = [Isolation(onset_to_isolation_delay = Exponential(1.0)),
                ContactTracing(probability = 0.8,
                    isolation_to_trace_delay = Exponential(1.0)),
                RingVaccination(efficacy = 0.9)])
        state = simulate(spec; n_initial = 3, rng = StableRNG(1), max_cases = 200)
        df = linelist(state; infected_only = false)
        @test nrow(df) == length(state.individuals)

        uninfected = df[.!df.infected, :]
        @test nrow(uninfected) > 0
        @test all(ismissing, uninfected.date_infection)
        @test all(ismissing, uninfected.date_onset)

        # Tracing, vaccination and quarantine happen to uninfected contacts too.
        @test any(!ismissing, uninfected.date_trace)
        @test any(!ismissing, uninfected.date_vaccination)
        @test any(!ismissing, uninfected.date_isolation)
        quarantined = uninfected[coalesce.(uninfected.quarantined, false), :]
        @test all(!ismissing, quarantined.date_isolation)

        # Infected rows are exactly the default line list.
        cases = linelist(state)
        shared = intersect(names(df), names(cases))
        @test isequal(df[df.infected, shared], cases[:, shared])
    end

    @testset "linelist picks up custom state fields generically" begin
        rng = StableRNG(42)
        model = BranchingProcess(Poisson(1.5), Exponential(5.0))
        risk_group = (rng, ind) -> (ind.state[:risk_group] = rand(rng) < 0.3 ? :high : :low)
        custom_time = (
            rng, ind) -> (ind.state[:vaccination_time] = ind.infection_time + 7.0)
        attrs = [clinical, risk_group, custom_time]
        state = tsim(model; attributes = attrs,
            max_cases = 30, rng = rng)

        df = linelist(state)
        @test "risk_group" in names(df)             # pass-through, symbol → string
        @test "date_vaccination" in names(df)       # `_time` → `date_` convention
        @test eltype(df.risk_group) <: Union{Missing, AbstractString}
        @test eltype(df.date_vaccination) <: Union{Missing, Date}
        # The attribute reads the infection time at creation, which index cases
        # must already have, so every case is vaccinated a week after infection.
        @test any(==(0), df.parent_id)
        @test all(!ismissing, df.date_vaccination)
        @test all(df.date_vaccination .== df.date_infection .+ Day(7))
    end
end

function EpiBranch.event_time_metadata(::Val{:appointment_time})
    (column = :date_appointment, requires_infection = false)
end

@testset "External event dates" begin
    state = simulate(BranchingProcess(Poisson(0.0)); n_initial = 4, rng = StableRNG(42))
    reference = Date(2020, 1, 1)
    for (i, ind) in enumerate(state.individuals)
        ind.state[:infected] = i == 1
        ind.state[:appointment_time] = (2.0, 3.0, Inf, missing)[i]
        ind.state[:onset_time] = 4.0
        ind.state[:admission_time] = 5.0
        ind.state[:custom_disease_time] = 6.0
        ind.state[:vaccination_time_booster] = i == 3 ? NaN : 7.0
        ind.state[:immunity_time_booster] = i == 4 ? "unknown" : 8.0
        ind.state[:trace_time] = 1.0
    end
    df = linelist(state; infected_only = false, reference_date = reference)
    @test isequal(df.date_appointment, [
        reference + Day(2), reference + Day(3), missing, missing])
    for key in (:date_onset, :date_admission, :date_custom_disease)
        @test all(ismissing, df[2:4, key])
        @test !ismissing(df[1, key])
    end
    @test isequal(df.date_vaccination_booster,
        [reference + Day(7), reference + Day(7), missing, reference + Day(7)])
    @test isequal(df.date_immunity_booster,
        [reference + Day(8), reference + Day(8), reference + Day(8), missing])
    @test all(==(reference + Day(1)), df.date_trace)
    @test nrow(linelist(state)) == 1
    @test event_time_metadata(Val(:ordinary_value)) === nothing
    @test event_time_metadata(Val(:custom_disease_time)).requires_infection
end
