# A leaky vaccine written from outside the package: the whole population is
# dosed at `from_time`, and every exposure from then on is blocked with
# probability `efficacy`. The entire intervention is one competing risk, which
# is the shape #232's seam has to carry.
struct LeakyVaccine <: EpiBranch.AbstractIntervention
    efficacy::Float64
    from_time::Float64
end
function EpiBranch.competing_risk(v::LeakyVaccine, parent, contact, state)
    Risk(event_time = v.from_time, block_probability = v.efficacy)
end

@testset "HomogeneousProcess (Sellke fixed pool)" begin
    @testset "deterministic final size (major outbreaks)" begin
        # With β = 2 and mean infectious period 1, R0 = β·E[T] = 2; the
        # deterministic attack rate solves z = 1 - exp(-R0·z), z ≈ 0.7968.
        # Conditioning on major outbreaks, the mean should match.
        N = 3000
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = [Transition(:recovered; from = :infection,
                delay = Exponential(1.0), terminal = true)])
        finals = [simulate(m; rng = StableRNG(s), n_initial = 5).cumulative_cases
                  for s in 1:40]
        z = 0.7968
        major = filter(x -> x > 0.3 * N, finals)
        @test length(major) > 20                 # most seeds take off at R0 = 2
        @test all(x -> x <= N, finals)            # final size never exceeds the pool
        @test isapprox(mean(major) / N, z; atol = 0.03)
    end

    @testset "sub-critical outbreaks stay small" begin
        # β = 0.5 with mean infectious period 1 gives R0 = 0.5 < 1.
        N = 2000
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 0.5, population_size = N);
            progression = [Transition(:recovered; from = :infection, delay = 1.0,
                terminal = true)])
        finals = [simulate(m; rng = StableRNG(s), n_initial = 1).cumulative_cases
                  for s in 1:100]
        @test mean(finals) < 0.1 * N
    end

    @testset "saturation infects the whole pool" begin
        N = 200
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 100.0, population_size = N);
            progression = [Transition(:recovered; from = :infection, delay = 1.0,
                terminal = true)])
        finals = [simulate(m; rng = StableRNG(s), n_initial = 1).cumulative_cases
                  for s in 1:20]
        @test mean(finals) > 0.98 * N
        @test all(x -> x <= N, finals)
    end

    @testset "depletion is real (final size ≤ N)" begin
        N = 300
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 5.0, population_size = N);
            progression = [Transition(:recovered; from = :infection, delay = 1.0,
                terminal = true)])
        for s in 1:20
            @test simulate(m; rng = StableRNG(s), n_initial = 2).cumulative_cases <= N
        end
    end

    @testset "isolation shortens the outbreak" begin
        N = 1000
        base = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = [Transition(:recovered; from = :infection,
                delay = Exponential(1.0), terminal = true)])
        # An early isolation transition closes the infectious window (`:isolated`
        # is in the default `until`), curtailing spread.
        iso = ModelSpec(HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = [
                Transition(:recovered; from = :infection,
                    delay = Exponential(1.0), terminal = true),
                Transition(:isolated; from = :infection, delay = (rng, ind) -> 0.1)
            ])
        base_mean = mean(simulate(base; rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:30)
        iso_mean = mean(simulate(iso; rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:30)
        @test iso_mean < base_mean
    end

    @testset "Isolation intervention shortens the outbreak" begin
        # The Isolation *intervention* (not a Transition) must act on the pool:
        # its resolve_individual! runs in the Sellke loop and its isolation time
        # closes the infectious window. Onset comes from a progression transition
        # so it is anchored on the real infection time.
        N = 1000
        prog = [
            Transition(:onset; from = :infection, delay = 0.1),
            Transition(:recovered; from = :infection,
                delay = Exponential(1.0), terminal = true)
        ]
        base = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = prog)
        iso = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = prog,
            interventions = [Isolation(onset_to_isolation_delay = Exponential(0.1))])

        base_mean = mean(simulate(base; rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:20)
        iso_mean = mean(simulate(iso; rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:20)
        # Fast isolation pushes R below 1: the outbreak is curtailed.
        @test iso_mean < 0.1 * base_mean

        # A case's isolation time is actually written (the hook ran).
        state = simulate(iso; rng = StableRNG(1), n_initial = 3)
        @test any(isfinite(EpiBranch.isolation_time(ind)) for ind in state.individuals)
    end

    @testset "Unhonoured intervention warns rather than silently ignoring" begin
        prog = [Transition(:recovered; from = :infection,
            delay = Exponential(1.0), terminal = true)]
        # A mass-action pool has no pairwise contact structure for tracing to act
        # along, so tracing stays unhonoured there.
        ct = ModelSpec(
            HomogeneousProcess(; transmission_rate = 1.5, population_size = 200);
            progression = prog,
            interventions = [ContactTracing(probability = 0.5,
                isolation_to_trace_delay = Exponential(1.0))])
        @test_logs (:warn, r"does not honour"i) match_mode=:any simulate(
            ct; rng = StableRNG(1), n_initial = 2)

        # A rollout that doses each newly created contact has nobody to dose on a
        # path that creates none.
        mass = ModelSpec(
            HomogeneousProcess(; transmission_rate = 1.5, population_size = 200);
            progression = prog,
            interventions = [MassVaccination(efficacy = 0.8, eligibility_time = 0.0)])
        @test_logs (:warn, r"does not honour"i) match_mode=:any simulate(
            mass; rng = StableRNG(1), n_initial = 2)

        # Leaky isolation is honoured: the residual transmission it leaves is a
        # per-contact block, which the pool resolves on each contact it delivers.
        # It warns about nothing, and it bites in proportion to the residual.
        pool = HomogeneousProcess(; transmission_rate = 2.0, population_size = 500)
        onsets = clinical_presentation(incubation_period = LogNormal(-1.0, 0.3),
            prob_asymptomatic = 0.0)
        leaky(residual) = [Isolation(onset_to_isolation_delay = Exponential(0.5),
            post_isolation_transmission = residual)]
        mean_size(ivs) = sum(
            simulate(
                ModelSpec(pool; progression = prog, interventions = ivs,
                    attributes = onsets);
                rng = StableRNG(s), n_initial = 5).cumulative_cases
        for s in 1:20) / 20
        base_mean = mean_size(AbstractIntervention[])
        @test EpiBranch._sellke_honours(pool, leaky(0.9)[1])
        # Residual 1.0 is no isolation at all; 0.9 barely reduces transmission;
        # 0.2 cuts most of it. (Isolation draws its own delays either way, so
        # residual 1.0 gives the same process off a different rng stream.)
        @test isapprox(mean_size(leaky(1.0)), base_mean; rtol = 0.1)
        @test 0.8 * base_mean < mean_size(leaky(0.9)) < base_mean
        @test mean_size(leaky(0.2)) < 0.5 * base_mean
    end

    @testset "per-individual susceptibility and infectiousness apply on the pool" begin
        # A threshold crossing is one arriving contact, and the pool puts it to
        # the same competing risks the generation engine uses: both multipliers
        # mean what they mean there, a per-contact block.
        N = 500
        prog = [Transition(:recovered; from = :infection,
            delay = Exponential(1.0), terminal = true)]
        pool = HomogeneousProcess(; transmission_rate = 2.0, population_size = N)
        mean_size(attrs) = sum(
            simulate(ModelSpec(pool; progression = prog, attributes = attrs);
                rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:15) / 15

        full = mean_size(transmission_traits(susceptibility = 1.0))
        @test mean_size(transmission_traits(susceptibility = 0.5)) < full
        @test mean_size(transmission_traits(susceptibility = 0.2)) <
              mean_size(transmission_traits(susceptibility = 0.5))
        # Susceptibility 0 blocks every contact, so only the seeds are infected.
        none = simulate(
            ModelSpec(pool; progression = prog,
                attributes = transmission_traits(susceptibility = 0.0));
            rng = StableRNG(1), n_initial = 3)
        @test none.cumulative_cases == 3

        # Infectiousness acts on the other side of the same pair.
        @test mean_size(transmission_traits(infectiousness = 0.5)) < full
        silent = simulate(
            ModelSpec(pool; progression = prog,
                attributes = transmission_traits(infectiousness = 0.0));
            rng = StableRNG(1), n_initial = 3)
        @test silent.cumulative_cases == 3

        # Blocking thins the force of infection, so a susceptibility of s is the
        # same process as a transmission rate scaled by s. Compare attack rates.
        half_beta = HomogeneousProcess(; transmission_rate = 1.0, population_size = N)
        scaled = sum(
            simulate(ModelSpec(half_beta; progression = prog);
                rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:40) / 40
        blocked = sum(
            simulate(
                ModelSpec(pool; progression = prog,
                    attributes = transmission_traits(susceptibility = 0.5));
                rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:40) / 40
        @test isapprox(blocked, scaled; rtol = 0.15)
    end

    @testset "a user-defined leaky vaccination bites on the pool" begin
        # The seam is open from outside: an intervention returning a time-tagged
        # Risk changes the pool's results without touching the package.
        N = 600
        prog = [Transition(:recovered; from = :infection,
            delay = Exponential(1.0), terminal = true)]
        pool = HomogeneousProcess(; transmission_rate = 2.0, population_size = N)
        mean_size(ivs) = sum(
            simulate(ModelSpec(pool; progression = prog, interventions = ivs);
                rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:15) / 15

        base = mean_size(AbstractIntervention[])
        @test mean_size([LeakyVaccine(0.0, 0.0)]) == base
        @test mean_size([LeakyVaccine(0.5, 0.0)]) < 0.9 * base
        @test mean_size([LeakyVaccine(1.0, 0.0)]) == 3         # only the seeds
        # A dose that arrives after the outbreak has burnt out changes nothing.
        @test mean_size([LeakyVaccine(1.0, 1000.0)]) == base
    end

    @testset "a pool that can never finish is refused, not looped" begin
        # With no removal transition the infectious window never closes, so the
        # force of infection never decays; with every contact certainly blocked
        # there is no end to reach. The pool says so rather than spinning.
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 2.0, population_size = 40);
            attributes = transmission_traits(susceptibility = 0.0))
        @test_throws ErrorException simulate(m; rng = StableRNG(1), n_initial = 2)

        # A rare but possible infection is no endless loop. At susceptibility
        # 1e-4 about 10,000 contacts are blocked between one infection and the
        # next, and about 2 million over the run, more than the guard allows in a
        # row; counting them across infections would refuse a model that finishes.
        rare = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 200);
            attributes = transmission_traits(susceptibility = 1e-4))
        @test simulate(rare; rng = StableRNG(1), n_initial = 2).cumulative_cases == 200

        # A removal transition is all it takes: the outbreak ends at the seeds.
        with_removal = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 40);
            progression = [Transition(:recovered; from = :infection,
                delay = Exponential(1.0), terminal = true)],
            attributes = transmission_traits(susceptibility = 0.0))
        @test simulate(with_removal; rng = StableRNG(1), n_initial = 2).cumulative_cases ==
              2
    end

    @testset "Scheduled interventions gate on the running clock on the pool" begin
        # The loop exposes each case's infection time as the running clock, so a
        # Scheduled(Isolation; start_time) isolates only cases infected at/after
        # start_time — curtailing when it activates, and gating by time.
        N = 1000
        prog = [
            Transition(:onset; from = :infection, delay = 0.1),
            Transition(:recovered; from = :infection,
                delay = Exponential(1.0), terminal = true)
        ]
        mk(start_time) = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = prog,
            interventions = [Scheduled(
                Isolation(onset_to_isolation_delay = Exponential(0.1)); start_time)])
        mean_cc(m) = mean(simulate(m; rng = StableRNG(s), n_initial = 3).cumulative_cases
        for s in 1:15)

        base = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = N);
            progression = prog)
        # Active from t = 0: curtails hard. Start far past the outbreak: barely.
        @test mean_cc(mk(0.0)) < 0.1 * mean_cc(base)
        @test mean_cc(mk(1000.0)) > 0.5 * mean_cc(base)

        # Mid start_time gates on the case's own infection time.
        st = simulate(mk(3.0); rng = StableRNG(1), n_initial = 3)
        @test !any(ind -> ind.infection_time < 3.0 && is_isolated(ind), st.individuals)
        @test any(ind -> ind.infection_time >= 3.0 && is_isolated(ind), st.individuals)

        # A Scheduled wrapping an unhonoured intervention still warns.
        sched_ct = ModelSpec(
            HomogeneousProcess(; transmission_rate = 1.5, population_size = 200);
            progression = prog,
            interventions = [Scheduled(
                ContactTracing(probability = 0.5,
                    isolation_to_trace_delay = Exponential(1.0)); start_time = 5.0)])
        @test_logs (:warn, r"does not honour"i) match_mode=:any simulate(
            sched_ct; rng = StableRNG(1), n_initial = 2)
    end

    @testset "removal before infectious onset never infects" begin
        # A latent period opens the window at :infectious, but isolation fires
        # first (close_t <= open_t). Such a case is never infectious: it must be
        # skipped rather than pop a close event against an id never made infectious.
        N = 100
        progression = [
            Transition(:infectious; from = :infection, delay = (rng, ind) -> 5.0),
            Transition(:recovered; from = :infectious,
                delay = (rng, ind) -> 1.0, terminal = true),
            Transition(:isolated; from = :infection, delay = (rng, ind) -> 0.1)
        ]
        # The infectious window is derived to open at :infectious (a latent period
        # produces it).
        @test EpiBranch._resolve_infectious_from(nothing, progression) === :infectious
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 5.0, population_size = N);
            progression = progression)
        # No index case reaches :infectious, so no secondary transmission occurs
        # and the run completes with only the seeds infected.
        state = simulate(m; rng = StableRNG(1), n_initial = 5)
        @test state.cumulative_cases == 5
    end

    @testset "onset is measured from each case's own infection time" begin
        # Pool members are created, and their incubation periods drawn, before
        # the pool sets their infection times. Isolation depends on onset, so
        # onset must be counted from the time each case was infected.
        clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3))
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0),
            test_sensitivity = 1.0)
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 1.0, population_size = 500);
            progression = [Transition(:recovered; from = :infection, delay = 10.0,
                terminal = true)],
            attributes = clinical, interventions = [iso])
        state = simulate(m; n_initial = 1, rng = StableRNG(3))
        secondary = [ind
                     for ind in state.individuals
                     if is_infected(ind) && ind.parent_id != 0]
        @test !isempty(secondary)
        @test all(onset_time(ind) >= ind.infection_time for ind in secondary)
        @test all(onset_time(ind) - ind.infection_time ≈ ind.state[:incubation_period]
        for ind in secondary)
        @test all(isolation_time(ind) >= onset_time(ind) for ind in secondary)
    end

    @testset "line list and timing" begin
        progression = [
            Transition(:infectious; from = :infection, delay = Exponential(1.0)),
            Transition(:recovered; from = :infectious, delay = Exponential(2.0),
                terminal = true)
        ]
        @test EpiBranch._resolve_infectious_from(nothing, progression) === :infectious
        m = ModelSpec(HomogeneousProcess(; transmission_rate = 2.0, population_size = 500);
            progression = progression)
        state = simulate(m; rng = StableRNG(1), n_initial = 5)
        ll = linelist(state)
        @test size(ll, 1) == state.cumulative_cases
        @test EpiBranch._timetype(state) === Float64
        @test :date_infectious in propertynames(ll)
        @test :date_recovered in propertynames(ll)
    end

    @testset "transmission_rate is the transmission parameter" begin
        # β is the per-infective rate, stored and used directly (no R0 map).
        p = HomogeneousProcess(; transmission_rate = 2.0, population_size = 100)
        @test p.transmission_rate == 2.0
        # `transmission_rate` is required.
        @test_throws UndefKeywordError HomogeneousProcess(; population_size = 10)
        # β must be finite and non-negative; negative, infinite and NaN rates
        # are rejected before they reach the Sellke hazard.
        @test_throws ArgumentError HomogeneousProcess(; transmission_rate = -1.0,
            population_size = 10)
        @test_throws ArgumentError HomogeneousProcess(; transmission_rate = Inf,
            population_size = 10)
        @test_throws ArgumentError HomogeneousProcess(; transmission_rate = NaN,
            population_size = 10)
        # β = 0 (no transmission) is a valid degenerate model.
        @test HomogeneousProcess(; transmission_rate = 0.0,
            population_size = 10).transmission_rate == 0.0
        # A non-positive population is rejected at construction.
        @test_throws ArgumentError HomogeneousProcess(; transmission_rate = 2.0,
            population_size = 0)
    end

    # A small helper that runs the structured pool directly: tag a real
    # attribute (`:age_band`) on a fixed population of size N via `band_of`, name
    # it as the mixing attribute with `mixing_by = (:age_band,)`, supply a force
    # keyed on the band value, and return the band each infected case fell in.
    function _run_pool(N, band_of, force; n_initial = 5, rng)
        process = HomogeneousProcess(; transmission_rate = 1.0, population_size = N)
        prog = [Transition(:recovered; from = :infection, delay = Exponential(1.0),
            terminal = true)]
        state = EpiBranch.new_state(process, prog, NoAttributes(), rng)
        EpiBranch.add_individuals!(state, N, AbstractIntervention[];
            setup = (ind, i) -> (ind.state[:age_band] = band_of(ind)))
        EpiBranch._sellke_pool!(state, collect(1:N), rng;
            mixing_by = (:age_band,), force = force,
            n_initial = n_initial, from = :infection,
            until = (:recovered, :died, :isolated))
        return [ind.state[:age_band]
                for ind in state.individuals
                if get(ind.state, :infected, false)]
    end

    @testset "two-band uniform matrix reduces to one pool" begin
        # Two bands with uniform contact behave as a single pool of size N: the
        # force felt is β/N·(total infectious) regardless of band, so the
        # major-outbreak attack rate matches the homogeneous law (z ≈ 0.7968 at
        # R0 = 2, since β = 2 and mean infectious period = 1).
        N = 3000
        β = 2.0
        band_of = ind -> (ind.id <= N ÷ 2 ? 1 : 2)
        force = (type, counts) -> β / N * sum(values(counts))
        finals = [length(_run_pool(N, band_of, force; rng = StableRNG(s)))
                  for s in 1:40]
        major = filter(x -> x > 0.3 * N, finals)
        @test length(major) > 20
        @test all(x -> x <= N, finals)
        @test isapprox(mean(major) / N, 0.7968; atol = 0.03)
    end

    @testset "asymmetric mixing orders attack rates" begin
        # A 2×2 contact matrix keyed by band, where band 1 mixes far more than
        # band 2. With equal band sizes, the force on a band-`b` susceptible is
        # (1/half)·Σ_h M[b,h]·counts[(h,)]. The high-contact band should suffer a
        # strictly higher attack rate than the low-contact band, over replicates.
        N = 2000
        half = N ÷ 2
        M = [3.0 0.5; 0.5 0.5]         # band 1 mixes much more than band 2
        band_of = ind -> (ind.id <= half ? 1 : 2)
        force = (type, counts) -> begin
            b = type[1]
            sum(M[b, h] * get(counts, (h,), 0) for h in 1:2) / half
        end
        ar1 = Float64[]
        ar2 = Float64[]
        for s in 1:40
            bands = _run_pool(N, band_of, force; n_initial = 10,
                rng = StableRNG(s))
            n1 = count(==(1), bands)
            n2 = count(==(2), bands)
            # Keep major outbreaks only, so the ordering is about who is hit hardest.
            (n1 + n2) > 0.3 * N || continue
            push!(ar1, n1 / half)
            push!(ar2, n2 / half)
        end
        @test length(ar1) > 20
        @test mean(ar1) > mean(ar2)
    end

    @testset "structured pool refuses risks that depend on the infector" begin
        # Two bands that never mix (M = diag(2, 2)). Band 1's epidemic cannot
        # depend on anything about band 2's infectives, but a contact's infector
        # is drawn from everyone infectious, so a block read off the infector
        # would let band 2 thin band 1's contacts. The pool refuses such risks,
        # while a risk acting on the contact alone leaves band 1 untouched.
        N = 2000
        half = N ÷ 2
        force = (type, counts) -> 2.0 * get(counts, type, 0) / half
        prog = [Transition(:recovered; from = :infection, delay = Exponential(1.0),
            terminal = true)]
        function band1_attack(s; interventions = AbstractIntervention[],
                band2! = ind -> nothing)
            rng = StableRNG(s)
            process = HomogeneousProcess(; transmission_rate = 1.0, population_size = N)
            state = EpiBranch.new_state(process, prog, NoAttributes(), rng)
            EpiBranch.add_individuals!(state, N, interventions;
                setup = (ind, i) -> begin
                    ind.state[:band] = i <= half ? 1 : 2
                    i > half && band2!(ind)
                end)
            EpiBranch._sellke_pool!(state, collect(1:N), rng; mixing_by = (:band,),
                force, n_initial = 20, from = :infection, until = (:recovered,),
                interventions)
            return count(ind -> ind.state[:band] == 1 && is_infected(ind),
                state.individuals) / half
        end
        major(ars) = mean(filter(>(0.2), ars))

        @test_throws r"infectiousness" band1_attack(1;
            band2! = ind -> (ind.infectiousness = 0.0))
        leaky = Isolation(onset_to_isolation_delay = Exponential(1.0),
            post_isolation_transmission = 0.5)
        @test_throws r"Isolation" band1_attack(1; interventions = [leaky])
        @test_throws r"Isolation" band1_attack(1;
            interventions = [Scheduled(leaky; start_time = 5.0)])
        # A user's own risk may read the infector, so it is refused too.
        @test_throws r"LeakyVaccine" band1_attack(1;
            interventions = [LeakyVaccine(0.5, 0.0)])
        # Perfect isolation closes the window, so it never blocks a drawn contact.
        @test !EpiBranch._blocks_by_infector(
            Isolation(onset_to_isolation_delay = Exponential(1.0)))

        # Band 2's susceptibility acts on its own contacts only, so band 1's
        # attack rate is the SIR final size at R0 = 2 either way.
        base = major([band1_attack(s) for s in 1:30])
        immune2 = major([band1_attack(s; band2! = ind -> (ind.susceptibility = 0.0))
                         for s in 1:30])
        @test isapprox(base, 0.7968; atol = 0.03)
        @test isapprox(immune2, 0.7968; atol = 0.03)

        # One mixing type attributes every contact exactly, so nothing is refused.
        pool = HomogeneousProcess(; transmission_rate = 2.0, population_size = 200)
        @test simulate(
            ModelSpec(pool; progression = prog,
                attributes = transmission_traits(infectiousness = 0.5));
            rng = StableRNG(1), n_initial = 3).cumulative_cases >= 3
    end

    @testset "positive force with empty infectious pool is index-labelled" begin
        # A custom force with a count-independent positive hazard (external
        # importation) keeps firing infections even when no one is infectious. The
        # first infection draws its source from an empty pool: without a guard that
        # throws; with the guard it falls back to the index-case label 0.
        N = 50
        process = HomogeneousProcess(; transmission_rate = 1.0, population_size = N)
        prog = [Transition(:recovered; from = :infection, delay = Exponential(1.0),
            terminal = true)]
        rng = StableRNG(1)
        state = EpiBranch.new_state(process, prog, NoAttributes(), rng)
        EpiBranch.add_individuals!(state, N, AbstractIntervention[])
        EpiBranch._sellke_pool!(state, collect(1:N), rng; mixing_by = (),
            force = (type, counts) -> 0.5, n_initial = 0,
            from = :infection, until = (:recovered, :died, :isolated))
        @test count(ind -> get(ind.state, :infected, false), state.individuals) > 0
    end

    @testset "conditioned simulation and show" begin
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 500);
            progression = prog)
        # `condition` retries until the final size falls in the range
        state = simulate(spec; condition = 100:500, n_initial = 5, rng = StableRNG(1))
        @test state.cumulative_cases in 100:500

        # show renders the β form
        @test occursin("β=",
            repr(HomogeneousProcess(; transmission_rate = 2.0, population_size = 10)))
    end

    @testset "termination controls warn on the fixed pool" begin
        prog = [Transition(:recovered; from = :infection, delay = 1.0, terminal = true)]
        spec = ModelSpec(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 200);
            progression = prog)
        # A set termination control has no effect on the extinction-run pool, so
        # `simulate` warns rather than silently ignoring it.
        @test_logs (:warn, r"ignores termination controls") simulate(
            spec; n_initial = 3, max_cases = 50, rng = StableRNG(1))
        # No termination keyword set → no warning.
        @test_logs simulate(spec; n_initial = 3, rng = StableRNG(1))
        # The trait itself: the pool ignores the controls, the generation engine
        # honours them.
        @test !EpiBranch._honours_termination_controls(
            HomogeneousProcess(; transmission_rate = 2.0, population_size = 10))
        @test EpiBranch._honours_termination_controls(
            BranchingProcess(Poisson(1.5), Exponential(2.0)))
    end
end
