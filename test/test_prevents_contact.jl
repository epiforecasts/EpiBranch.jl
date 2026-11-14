using DataFrames

# A ward that blocks a share of every infector's contacts, standing in for a
# measure that either stops the meeting (`closed`) or only the infection.
struct Ward <: EpiBranch.AbstractIntervention
    closed::Bool
    probability::Float64
end
function EpiBranch.competing_risk(w::Ward, parent, contact, state)
    parent === contact && return nothing
    return Risk(block_probability = w.probability)
end
EpiBranch.prevents_contact(w::Ward) = w.closed

# A ward that says nothing about contacts, so it takes the default.
struct PlainWard <: EpiBranch.AbstractIntervention end
function EpiBranch.competing_risk(::PlainWard, parent, contact, state)
    parent === contact && return nothing
    return Risk(block_probability = 1.0)
end

const _CLINICAL = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))

# The infection time of each case's isolation, by id; `Inf` for those never
# isolated.
_iso_time(state, id) = get(state.individuals[id].state, :isolation_time, Inf)

# A model that names the contacts each case makes, creating them as it goes.
struct MintedContacts <: EpiBranch.TransmissionModel end
EpiBranch.population_size(::MintedContacts) = EpiBranch.NoPopulation()
function EpiBranch.contacts_of(::MintedContacts, parent, state::EpiBranch.SimulationState)
    return [
        (EpiBranch.make_contact!(state, parent, parent.infection_time + dt), parent.infection_time + dt)
            for dt in (1.0, 2.0)
    ]
end

@testset "A blocked contact that never happened" begin
    @testset "the trait and the built-in defaults" begin
        @test !EpiBranch.prevents_contact(PlainWard())
        @test EpiBranch.prevents_contact(Ward(true, 0.5))
        iso = Isolation(onset_to_isolation_delay = 1.0, duration = Inf)
        home = Isolation(onset_to_isolation_delay = 1.0, duration = Inf, prevents_contact = false)
        @test EpiBranch.prevents_contact(iso)
        @test !EpiBranch.prevents_contact(home)
        @test EpiBranch.prevents_contact(Quarantine(duration = 7.0))
        @test !EpiBranch.prevents_contact(Quarantine(duration = 7.0, prevents_contact = false))
        ct = ContactTracing(OnSymptomOnset(), 1.0, Exponential(1.0), Quarantine(duration = 7.0))
        @test EpiBranch.prevents_contact(ct)
        @test !EpiBranch.prevents_contact(
            ContactTracing(
                OnSymptomOnset(), 1.0, Exponential(1.0),
                Quarantine(duration = 7.0, prevents_contact = false)
            )
        )
        @test !EpiBranch.prevents_contact(
            ContactTracing(OnSymptomOnset(), 1.0, Exponential(1.0), FlagOnly())
        )
        @test EpiBranch.prevents_contact(Scheduled(iso; start_time = 3.0))
        @test !EpiBranch.prevents_contact(OnRoutes(home, (:household,)))
        @test !EpiBranch.prevents_contact(
            MassVaccination(efficacy = 0.5, eligibility_time = 0.0)
        )
    end

    @testset "a branching process drops the contacts a closing measure blocks" begin
        model = BranchingProcess(Poisson(3.0), Exponential(2.0))
        for (closed, expected) in ((true, 0), (false, nothing))
            spec = ModelSpec(model; interventions = [Ward(closed, 1.0)])
            state = simulate(spec; max_cases = 50, n_initial = 3, rng = StableRNG(1))
            rows = contacts(state)
            @test state.cumulative_cases == 3
            if closed
                @test nrow(rows) == 0
                @test all(isempty(ind.secondary_case_ids) for ind in state.individuals)
                @test nrow(linelist(state; infected_only = false)) == 3
            else
                @test nrow(rows) > 0
                @test !any(rows.infected)
                @test nrow(linelist(state; infected_only = false)) > 3
            end
        end
    end

    @testset "the default keeps the contact" begin
        spec = ModelSpec(
            BranchingProcess(Poisson(3.0), Exponential(2.0));
            interventions = [PlainWard()]
        )
        state = simulate(spec; max_cases = 50, n_initial = 3, rng = StableRNG(1))
        @test nrow(contacts(state)) > 0
    end

    @testset "a partial block leaves only the contacts that met" begin
        model = BranchingProcess(Poisson(4.0), Exponential(2.0))
        closed = simulate(
            ModelSpec(model; interventions = [Ward(true, 0.5)]);
            max_cases = 200, rng = StableRNG(2)
        )
        open_ = simulate(
            ModelSpec(model; interventions = [Ward(false, 0.5)]);
            max_cases = 200, rng = StableRNG(2)
        )
        # Every contact that remains after closing met and infected; open
        # contacts include those that were blocked.
        @test all(contacts(closed).infected)
        @test !all(contacts(open_).infected)
    end

    @testset "a later source that prevents the contact is still asked" begin
        # The open ward blocks first, but the closed one blocks the same
        # contact, so there was none.
        spec = ModelSpec(
            BranchingProcess(Poisson(3.0), Exponential(2.0));
            interventions = [Ward(false, 1.0), Ward(true, 1.0)]
        )
        state = simulate(spec; max_cases = 50, n_initial = 3, rng = StableRNG(1))
        @test nrow(contacts(state)) == 0
    end

    @testset "hospital isolation leaves no contact after it; home isolation does" begin
        model = BranchingProcess(Poisson(5.0), Gamma(3.0, 2.0))
        run(prevents) = simulate(
            ModelSpec(
                model;
                interventions = [
                    Isolation(
                        onset_to_isolation_delay = 0.0, duration = Inf,
                        prevents_contact = prevents
                    ),
                ],
                attributes = _CLINICAL
            );
            max_cases = 200, rng = StableRNG(3)
        )
        after_isolation(state) = let rows = contacts(state)
            count(
                r -> r.infection_time >= _iso_time(state, r.from), eachrow(rows)
            )
        end
        @test after_isolation(run(true)) == 0
        @test after_isolation(run(false)) > 0
    end

    @testset "tracing does not reach a contact that never happened" begin
        model = BranchingProcess(Poisson(5.0), Gamma(3.0, 2.0))
        run(prevents) = simulate(
            ModelSpec(
                model;
                interventions = [
                    Isolation(
                        onset_to_isolation_delay = 0.0, duration = Inf,
                        prevents_contact = prevents
                    ),
                    ContactTracing(
                        OnIsolation(), 1.0, Dirac(0.0),
                        Quarantine(duration = 14.0, prevents_contact = prevents)
                    ),
                ],
                attributes = _CLINICAL
            );
            max_cases = 200, rng = StableRNG(4)
        )
        hospital = run(true)
        home = run(false)
        listed(state) = Set(contacts(state).to)
        traced(state) = Set(ind.id for ind in state.individuals if is_traced(ind))
        @test traced(hospital) ⊆ listed(hospital)
        @test !any(EpiBranch._is_withdrawn, filter(is_traced, hospital.individuals))
        # Home isolation keeps contacts that came after it, and traces them.
        @test length(traced(home)) > length(traced(hospital))
    end

    @testset "a model that names its contacts drops those a closing measure blocks" begin
        run(closed) = simulate(
            ModelSpec(MintedContacts(); interventions = [Ward(closed, 1.0)]);
            n_initial = 1, rng = StableRNG(5),
            stopping_rules = [Extinction(), MaxGenerations(3)]
        )
        kept = run(false)
        dropped = run(true)
        @test nrow(contacts(kept)) == 2
        @test !any(contacts(kept).infected)
        @test nrow(contacts(dropped)) == 0
        @test isempty(dropped.individuals[1].secondary_case_ids)
    end

    @testset "a fixed population lists a blocked pair only if the contact happened" begin
        progression = [Transition(:recovered; from = :infection, delay = 3.0, terminal = true)]
        run(closed) = simulate(
            ModelSpec(
                HomogeneousProcess(; transmission_rate = 4.0, population_size = 6);
                progression, interventions = [Ward(closed, 1.0)],
                recorder = EpiBranch.ExposedContactRecorder()
            );
            n_initial = 1, rng = StableRNG(3)
        )
        @test nrow(contacts(run(false))) > 0
        @test nrow(contacts(run(true))) == 0
    end
end
