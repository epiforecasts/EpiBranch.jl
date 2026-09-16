# The within-household final size and the household-level offspring law.
#
# The Markovian household (exponential contact interval, exponential infectious
# window) is resolved twice by independent routes — Ball's final-size recursion
# for the mean, a Markov-chain recursion for the whole offspring law — so the
# two checking each other is the backbone of these tests, with the simulator as
# a third opinion.

@testset "household_final_size" begin
    β, γ = 0.5, 1 / 4

    @testset "one susceptible: the index's escape probability" begin
        # The single susceptible escapes with the window's Laplace transform at
        # the kernel rate, E[exp(-βL)] = γ/(γ+β) for an exponential window.
        d = household_final_size(2, Exponential(1 / β), Exponential(1 / γ))
        @test support(d) == [1, 2]
        @test probs(d)[1] ≈ γ / (γ + β)

        # A fixed window leaves the plain escape probability.
        d = household_final_size(2, Exponential(1 / β), 6.0)
        @test probs(d)[1] ≈ exp(-β * 6.0)
    end

    @testset "a fixed window is the Reed-Frost household model" begin
        # With a fixed window every case escapes a given susceptible with the
        # same probability q, which is the chain-binomial model: from one index
        # among two susceptibles the final sizes are q², 2q²(1-q), (1-q)²(1+2q).
        L = 4.0
        q = exp(-β * L)
        d = household_final_size(3, Exponential(1 / β), L)
        @test probs(d) ≈ [q^2, 2 * q^2 * (1 - q), (1 - q)^2 * (1 + 2q)]
    end

    @testset "the quadrature route matches the closed form" begin
        # Gamma(1, θ) is Exponential(θ) in law but not in type, so it integrates
        # the escape probability numerically instead of taking the closed form.
        exact = household_final_size(5, Exponential(1 / β), Exponential(1 / γ))
        numeric = household_final_size(5, Exponential(1 / β), Gamma(1, 1 / γ))
        @test probs(exact) ≈ probs(numeric)
    end

    @testset "degenerate households and invalid arguments" begin
        @test probs(household_final_size(1, Exponential(2.0), 1.0)) == [1.0]
        d = household_final_size(3, Exponential(2.0), 1.0; initial_infectives = 3)
        @test support(d) == [3]
        @test_throws ArgumentError household_final_size(0, Exponential(2.0), 1.0)
        @test_throws ArgumentError household_final_size(
            3, Exponential(2.0), 1.0; initial_infectives = 4)
    end

    @testset "the recursion matches simulated households" begin
        n = 5
        model = ModelSpec(HouseholdProcess(fill(n, 20_000), Exponential(1 / β));
            progression = [Transition(:recovered; from = :infection, rate = γ,
                terminal = true)])
        state = simulate(model; rng = StableRNG(1))
        sizes = zeros(Int, 20_000)
        for ind in state.individuals
            is_infected(ind) && (sizes[ind.state[:household]] += 1)
        end
        simulated = [count(==(k), sizes) / length(sizes) for k in 1:n]
        exact = probs(household_final_size(n, Exponential(1 / β), Exponential(1 / γ)))
        @test maximum(abs, simulated .- exact) < 0.01
    end
end

@testset "household_offspring" begin
    β, γ, λG = 0.5, 1 / 4, 0.15
    _markov(n, households = 10) = ModelSpec(
        HouseholdProcess(fill(n, households), Exponential(1 / β));
        progression = [Transition(:recovered; from = :infection, rate = γ,
            terminal = true)])

    @testset "a household of one is the lone case's own offspring law" begin
        # With nobody to infect at home, the community contacts of a single case
        # are the Poisson events before its exponential window closes, which is
        # geometric with success probability γ/(γ+λG).
        o = household_offspring(_markov(1); global_rate = λG)
        law = household_offspring_law(o)
        p = γ / (γ + λG)
        @test probs(law)[1:5] ≈ [p * (1 - p)^k for k in 0:4]
        @test reproduction_number(o) ≈ λG / γ
    end

    @testset "R* is the community rate times the mean infectious person-time" begin
        n = 5
        o = household_offspring(_markov(n); global_rate = λG)
        final_size = household_final_size(n, Exponential(1 / β), Exponential(1 / γ))
        # A case's own window does not bear on whether it was infected, so the
        # mean total person-time is the mean final size times the mean window.
        @test reproduction_number(o) ≈ λG * mean(final_size) / γ
        # The offspring law is built from the same household epidemic, so its
        # mean agrees with R* up to the tail left outside the support.
        @test mean(household_offspring_law(o))≈reproduction_number(o) rtol=1e-6
    end

    @testset "the simulated route matches the exact one" begin
        # A Gamma(1, θ) window is an exponential window of a different type, so
        # the household epidemic is simulated rather than solved.
        n = 5
        exact = household_offspring(_markov(n); global_rate = λG)
        simulated = household_offspring(
            ModelSpec(HouseholdProcess(fill(n, 10), Exponential(1 / β));
                progression = [Transition(:recovered; from = :infection,
                    delay = Gamma(1, 1 / γ), terminal = true)]);
            global_rate = λG, n_samples = 20_000, rng = StableRNG(7))
        pe = probs(household_offspring_law(exact))
        ps = probs(household_offspring_law(simulated))
        @test maximum(abs, pe[1:20] .- ps[1:20]) < 0.01
        # The mean comes from the same closed form on both routes, because the
        # window law is still one delay of the progression.
        @test reproduction_number(simulated) ≈ reproduction_number(exact)
        @test extinction_probability(simulated)≈extinction_probability(exact) atol=0.01
    end

    @testset "contacts reach households in proportion to their size" begin
        sizes = [fill(2, 300); fill(4, 500); fill(6, 200)]
        o = household_offspring(
            ModelSpec(HouseholdProcess(sizes, Exponential(1 / β));
                progression = [Transition(:recovered; from = :infection, rate = γ,
                    terminal = true)]);
            global_rate = λG)
        @test o.sizes == [2, 4, 6]
        people = [2 * 300, 4 * 500, 6 * 200]
        @test o.mixing ≈ people ./ sum(people)
        # R* is the size-biased average of the per-size means, and it sits
        # between the smallest and largest household's own reproduction number.
        @test reproduction_number(o) ≈ sum(o.mixing .* o.means)
        @test o.means[1] < reproduction_number(o) < o.means[3]
        # A bigger household makes more onward infections, so its line is the
        # least likely to die out.
        q = extinction_probability(o)
        @test issorted(q; rev = true)
        @test all(0 .< q .< 1)
    end

    @testset "extinction solves the fixed point it is defined by" begin
        sizes = [fill(3, 100); fill(7, 100)]
        o = household_offspring(
            ModelSpec(HouseholdProcess(sizes, Exponential(1 / β));
                progression = [Transition(:recovered; from = :infection, rate = γ,
                    terminal = true)]);
            global_rate = λG)
        q = extinction_probability(o)
        s = sum(o.mixing .* q)
        pgf(law, x) = sum(pk * x^k for (k, pk) in zip(support(law), probs(law)))
        @test s≈sum(o.mixing[i] * pgf(o.laws[i], s) for i in eachindex(o.laws)) atol=1e-8
        @test all(q[i] ≈ pgf(o.laws[i], s) for i in eachindex(q))
        @test epidemic_probability(o) ≈ 1 .- q
    end

    @testset "a subcritical process is certain to die out" begin
        o = household_offspring(_markov(4); global_rate = 0.005)
        @test reproduction_number(o) < 1
        @test extinction_probability(o) == [1.0]
    end

    @testset "isolation shortens the window and lowers R*" begin
        prog = [Transition(:onset; from = :infection, delay = 0.5),
            Transition(:recovered; from = :infection, rate = γ, terminal = true)]
        process() = HouseholdProcess(fill(5, 10), Exponential(1 / β))
        plain = ModelSpec(process(); progression = prog)
        isolated = ModelSpec(process(); progression = prog,
            interventions = [Isolation(onset_to_isolation_delay = Exponential(1.0),
                eligibility = AllCases())])
        base = household_offspring(plain; global_rate = λG, n_samples = 5_000,
            rng = StableRNG(3))
        with_isolation = household_offspring(isolated; global_rate = λG,
            n_samples = 5_000, rng = StableRNG(3))
        # Isolation removes a case from the household, so it cuts both the
        # community contacts it makes and the household members it infects.
        @test reproduction_number(with_isolation) < reproduction_number(base) / 2
        @test extinction_probability(with_isolation) == [1.0]
    end

    @testset "invalid arguments" begin
        @test_throws ArgumentError household_offspring(_markov(4); global_rate = 0.0)
        @test_throws ArgumentError household_offspring(_markov(4); global_rate = λG,
            n_samples = 0)
        # Community introductions would seed a household with more than the one
        # index case a newly infected household has.
        external = ModelSpec(
            HouseholdProcess(fill(4, 10), Exponential(1 / β);
                external_hazard = 0.01, obs_end = 30.0);
            progression = [Transition(:recovered; from = :infection, rate = γ,
                terminal = true)])
        @test_throws ArgumentError household_offspring(external; global_rate = λG)
        o = household_offspring(_markov(4); global_rate = λG)
        @test_throws ArgumentError household_offspring_law(o, 5)
        @test household_offspring_law(o, 4) === o.laws[1]
    end

    @testset "a bare process takes its own progression" begin
        # Without a terminal transition the infectious window never closes, so a
        # household would infect unboundedly many others.
        @test_throws ArgumentError household_offspring(
            HouseholdProcess(fill(3, 10), Exponential(1 / β)); global_rate = λG,
            n_samples = 10, rng = StableRNG(1))
    end

    @testset "show" begin
        o = household_offspring(_markov(3); global_rate = λG)
        @test occursin("R*", sprint(show, o))
    end
end

@testset "published values" begin
    @testset "Ball, Mollison and Scalia-Tomba (1997), Figure 2" begin
        # Households of five, an exponential infectious period of mean 1 and a
        # per-pair local infection rate λ_L. The paper prints the mean number
        # infected besides the index case, and the global rate at which R*
        # reaches 1 (Ann. Appl. Probab. 7(1): 46-89, Figure 2 and eq. 3.14).
        for (λ_local, infected_besides_index, critical_rate) in [
            (0.125, 0.578283, 0.6336), (0.1875, 0.879315, 0.5321),
            (0.5, 2.033827, 0.3296), (1.25, 3.117259, 0.2429)]
            d = household_final_size(5, Exponential(1 / λ_local), Exponential(1.0))
            @test mean(d)≈1 + infected_besides_index atol=1e-6

            model = ModelSpec(HouseholdProcess(fill(5, 10), Exponential(1 / λ_local));
                progression = [Transition(:recovered; from = :infection, rate = 1.0,
                    terminal = true)])
            o = household_offspring(model; global_rate = critical_rate)
            @test reproduction_number(o)≈1.0 atol=2e-4
        end
    end

    @testset "Ball, Mollison and Scalia-Tomba (1997), Tecumseh influenza" begin
        # Their Section 5 fit to the Tecumseh A(H3N2) household data: 567
        # households of one to five, a per-pair local rate of 0.0423, a fixed
        # infectious period of 4.1 days and a global rate of 0.1950. They report
        # a size-biased mean household outbreak of 1.4145 and R* of 1.1309.
        counts = [133, 189, 108, 106, 31]
        sizes = vcat([fill(n, counts[n]) for n in 1:5]...)
        model = ModelSpec(HouseholdProcess(sizes, Exponential(1 / 0.0423));
            progression = [Transition(:recovered; from = :infection, delay = 4.1,
                terminal = true)])
        o = household_offspring(model; global_rate = 0.1950, n_samples = 200,
            rng = StableRNG(1))
        @test sum(o.mixing .* o.means) / (0.1950 * 4.1)≈1.4145 atol=1e-4
        @test reproduction_number(o)≈1.1309 atol=1e-4
    end

    @testset "House and Keeling (2008) mean household outbreak" begin
        # Their closed forms for the mean number infected in a household of two
        # and of three, for a per-pair rate τ against a removal rate γ.
        τ, γ = 4.0, 1.0
        kernel, window = Exponential(1 / τ), Exponential(1 / γ)
        @test mean(household_final_size(2, kernel, window)) ≈ (2τ + γ) / (τ + γ)
        @test mean(household_final_size(3, kernel, window)) ≈
              (6τ^3 + 13τ^2 * γ + 6τ * γ^2 + γ^3) / ((τ + γ)^2 * (2τ + γ))
    end
end
