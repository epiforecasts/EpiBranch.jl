# An infection layer recording each host's immunity time in `host_times`, as a
# companion package's layer read out of a vaccinated outbreak does.
struct _VaxInfections{S, H} <: InfectionLayer
    structure::S
    infection_time::Vector{Float64}
    infectious_time::Vector{Float64}
    removal_time::Vector{Float64}
    is_index::Vector{Bool}
    obs_end::Float64
    followup_end::Float64
    host_times::H
end
function _VaxInfections(
        structure, inf, infectious, removal, index, imm;
        obs_end = Inf, followup_end = Inf
    )
    return _VaxInfections(
        structure, Float64.(inf), Float64.(infectious), Float64.(removal),
        Vector{Bool}(index), Float64(obs_end), Float64(followup_end),
        (; immunity_time = imm)
    )
end
EpiBranch.contact_structure(d::_VaxInfections) = d.structure

# A susceptibility effect written outside the package: from `start` on, a
# fraction `p` of hosts is protected outright and the rest have their hazard
# halved, whatever host times the layer records.
struct _HalfOrNothing
    start::Float64
    p::Float64
end
function EpiBranch.susceptibility_components(e::_HalfOrNothing, host)
    return (
        e.p => EpiBranch.HazardScaling(e.start, 0.0),
        (1 - e.p) => EpiBranch.HazardScaling(e.start, 0.5),
    )
end

@testset "Vaccine risk in the pairwise likelihood" begin
    # household {1, 2}: 1 is the index, infectious from 0 and never removed; 2
    # is the susceptible under test, immune from τ = 1. `Exponential(2.0)` has
    # a constant hazard 1/2, so cumhazard(t) = t/2 and loghazard(t) = -log(2)
    # throughout — simple enough to check by hand.
    k = Exponential(2.0)
    cumh(t) = t / 2
    logh(t) = -log(2)
    τ = 1.0

    data(inf2; followup_end = Inf) = _VaxInfections(
        [1, 1], [0.0, inf2],
        [0.0, isnan(inf2) ? NaN : inf2], [Inf, Inf], [true, false], [Inf, τ];
        followup_end
    )

    @testset "LeakyMode discounts exposure past immunity" begin
        vaccine = VaccineEffect(efficacy = 0.4, mode = LeakyMode())

        # escapes forever, observed to t = 5: unprotected hazard to τ, a 0.6×
        # discount from τ to 5.
        escaped = data(NaN; followup_end = 5.0)
        expected = -(cumh(τ) + 0.6 * (cumh(5.0) - cumh(τ)))
        @test pairwise_surv_loglik(k, escaped; susceptibility = vaccine) ≈ expected

        # infected at t = 3, past τ: the same escape term up to 3, and the
        # event hazard at 3 also discounted by 0.6.
        infected = data(3.0)
        expected_event = -(cumh(τ) + 0.6 * (cumh(3.0) - cumh(τ))) +
            (logh(3.0) + log(0.6))
        @test pairwise_surv_loglik(k, infected; susceptibility = vaccine) ≈ expected_event

        # infected before τ: immunity never comes into play, so the result is
        # exactly the unvaccinated computation.
        early = data(0.5)
        @test pairwise_surv_loglik(k, early; susceptibility = vaccine) ≈
            pairwise_surv_loglik(k, early; susceptibility = nothing)
    end

    @testset "AllOrNothingMode mixes over responder status" begin
        vaccine = VaccineEffect(efficacy = 0.3, mode = AllOrNothingMode())
        e = 0.3

        # escapes forever: e·(fully protected from τ) + (1-e)·(unprotected).
        escaped = data(NaN; followup_end = 5.0)
        expected = log(e * exp(-cumh(τ)) + (1 - e) * exp(-cumh(5.0)))
        @test pairwise_surv_loglik(k, escaped; susceptibility = vaccine) ≈ expected

        # infected at t = 3, past τ: a responder cannot be, so only the
        # unprotected branch contributes.
        infected = data(3.0)
        expected_event = log(1 - e) + logh(3.0) - cumh(3.0)
        @test pairwise_surv_loglik(k, infected; susceptibility = vaccine) ≈ expected_event

        # infected before τ: both branches agree, so the mixture collapses to
        # the unvaccinated computation exactly (log(e·L + (1-e)·L) = log(L)).
        early = data(0.5)
        @test pairwise_surv_loglik(k, early; susceptibility = vaccine) ≈
            pairwise_surv_loglik(k, early; susceptibility = nothing)
    end

    @testset "unvaccinated hosts are untouched by a shared vaccine effect" begin
        # host 2 has no immunity time (Inf): scoring it under a vaccine
        # effect must not change its contribution, whichever mode.
        unvacc = _VaxInfections(
            [1, 1], [0.0, 2.0], [0.0, 2.0], [Inf, Inf],
            [true, false], [Inf, Inf]
        )
        for mode in (LeakyMode(), AllOrNothingMode())
            vaccine = VaccineEffect(efficacy = 0.5, mode = mode)
            @test pairwise_surv_loglik(k, unvacc; susceptibility = vaccine) ≈
                pairwise_surv_loglik(k, unvacc; susceptibility = nothing)
        end
    end

    @testset "waning scales the discount continuously" begin
        escaped = data(NaN; followup_end = 5.0)

        # a waning function that never decays reproduces the constant-discount
        # closed form exactly.
        flat = VaccineEffect(efficacy = 0.4, mode = LeakyMode(), waning = dt -> 1.0)
        flat_ll = pairwise_surv_loglik(k, escaped; susceptibility = flat)
        no_waning = VaccineEffect(efficacy = 0.4, mode = LeakyMode())
        @test flat_ll ≈ pairwise_surv_loglik(k, escaped; susceptibility = no_waning)

        # an exponentially decaying dose, checked against the same integral
        # computed directly with the quadrature the package uses internally.
        decay = dt -> exp(-dt / 2)
        waned = VaccineEffect(efficacy = 0.4, mode = LeakyMode(), waning = decay)
        discounted, _ = EpiBranch.quadgk(
            s -> 0.5 * (1 - 0.4 * decay(s - τ)), τ, 5.0
        )
        expected = -(cumh(τ) + discounted)
        @test pairwise_surv_loglik(k, escaped; susceptibility = waned) ≈ expected rtol = 1.0e-6

        # infected past τ: the event hazard picks up the retained fraction at
        # that exposure.
        infected = data(3.0)
        expected_event = -(
            cumh(τ) + first(
                EpiBranch.quadgk(
                    s -> 0.5 * (1 - 0.4 * decay(s - τ)), τ, 3.0
                )
            )
        ) +
            (logh(3.0) + log1p(-0.4 * decay(3.0 - τ)))
        @test pairwise_surv_loglik(k, infected; susceptibility = waned) ≈ expected_event rtol = 1.0e-6
    end

    @testset "differentiable in efficacy" begin
        escaped = data(NaN; followup_end = 5.0)
        for mode in (LeakyMode(), AllOrNothingMode())
            f(θ) = pairwise_surv_loglik(
                k, escaped;
                susceptibility = VaccineEffect(efficacy = θ[1], mode = mode)
            )
            fd = ForwardDiff.gradient(f, [0.4])
            step = 1.0e-6
            numeric = (f([0.4 + step]) - f([0.4 - step])) / 2step
            @test only(fd) ≈ numeric rtol = 1.0e-4
        end
    end

    @testset "differentiable in efficacy at the ends of its range" begin
        # A gradient-based fit can start from no effect or a perfect vaccine,
        # so the derivative must hold at efficacy 0 and 1 as well as inside.
        for case in (data(NaN; followup_end = 5.0), data(3.0)),
                mode in (LeakyMode(), AllOrNothingMode()), e in (0.0, 1.0)

            f(θ) = pairwise_surv_loglik(
                k, case;
                susceptibility = VaccineEffect(efficacy = θ[1], mode = mode)
            )
            isfinite(f([e])) || continue
            step = 1.0e-6
            inner = e == 0.0 ? e + step : e - step
            numeric = (f([inner]) - f([e])) / (inner - e)
            @test only(ForwardDiff.gradient(f, [e])) ≈ numeric rtol = 1.0e-3 atol = 1.0e-5
        end
    end

    @testset "an efficacy above 1 is rejected under AD as well" begin
        escaped = data(NaN; followup_end = 5.0)
        f(θ) = pairwise_surv_loglik(
            k, escaped;
            susceptibility = VaccineEffect(efficacy = θ[1], mode = LeakyMode())
        )
        @test_throws ArgumentError f([1.5])
        @test_throws ArgumentError ForwardDiff.gradient(f, [1.5])
        for vaccine in (
                VaccineEffect(efficacy = 1.5, mode = LeakyMode(), waning = dt -> 1.0),
                VaccineEffect(efficacy = 1.5, mode = AllOrNothingMode()),
                VaccineEffect(efficacy = -0.5, mode = LeakyMode()),
            )
            @test_throws ArgumentError pairwise_surv_loglik(
                k, escaped; susceptibility = vaccine
            )
        end
        waned(θ) = pairwise_surv_loglik(
            k, escaped;
            susceptibility = VaccineEffect(
                efficacy = θ[1], mode = LeakyMode(), waning = dt -> 1.0
            )
        )
        @test_throws ArgumentError ForwardDiff.gradient(waned, [1.5])
    end

    @testset "efficacy must be a fixed value" begin
        drawn = VaccineEffect(efficacy = Beta(2.0, 2.0), mode = LeakyMode())
        @test_throws ArgumentError pairwise_surv_loglik(k, data(NaN); susceptibility = drawn)
    end

    @testset "a host without a recorded immunity time is unvaccinated" begin
        missing_imm = _VaxInfections(
            [1, 1], [0.0, 2.0], [0.0, 2.0], [Inf, Inf],
            [true, false], [missing, missing]
        )
        vaccine = VaccineEffect(efficacy = 0.5, mode = LeakyMode())
        @test pairwise_surv_loglik(k, missing_imm; susceptibility = vaccine) ≈
            pairwise_surv_loglik(k, missing_imm)
    end

    @testset "differentiable in efficacy for an infected host" begin
        infected = data(3.0)
        for mode in (LeakyMode(), AllOrNothingMode())
            f(θ) = pairwise_surv_loglik(
                k, infected;
                susceptibility = VaccineEffect(efficacy = θ[1], mode = mode)
            )
            fd = ForwardDiff.gradient(f, [0.4])
            step = 1.0e-6
            numeric = (f([0.4 + step]) - f([0.4 - step])) / 2step
            @test only(fd) ≈ numeric rtol = 1.0e-4
        end
    end

    @testset "a model's interventions score as their vaccination's effect" begin
        mv = MassVaccination(efficacy = 0.4, eligibility_time = 0.0)
        interventions = [Isolation(onset_to_isolation_delay = 1.0), mv]
        for d in (data(NaN; followup_end = 5.0), data(3.0))
            @test pairwise_surv_loglik(k, d; susceptibility = interventions) ≈
                pairwise_surv_loglik(k, d; susceptibility = EpiBranch.vaccine_effect(mv))
        end
    end

    @testset "a user-defined effect reaches the likelihood" begin
        effect = _HalfOrNothing(τ, 0.3)
        escaped = data(NaN; followup_end = 5.0)
        expected = log(
            0.3 * exp(-cumh(τ)) +
                0.7 * exp(-(cumh(τ) + 0.5 * (cumh(5.0) - cumh(τ))))
        )
        @test pairwise_surv_loglik(k, escaped; susceptibility = effect) ≈ expected

        # infected past `start`: only the halved component can be
        infected = data(3.0)
        expected_event = log(0.7) - (cumh(τ) + 0.5 * (cumh(3.0) - cumh(τ))) +
            logh(3.0) + log(0.5)
        @test pairwise_surv_loglik(k, infected; susceptibility = effect) ≈ expected_event

        # the per-component breakdown sums to the total
        layout = compile_contact_pairs(infected)
        @test sum(
            pairwise_surv_loglik_by_component(k, infected, layout; susceptibility = effect)
        ) ≈ pairwise_surv_loglik(k, infected, layout; susceptibility = effect)
    end

    @testset "per-component breakdown under a vaccine" begin
        # two households, one with a vaccinated susceptible
        two = _VaxInfections(
            [1, 1, 2, 2], [0.0, 3.0, 0.0, 2.0], [0.0, 3.0, 0.0, 2.0],
            [Inf, Inf, 4.0, Inf], [true, false, true, false], [Inf, τ, Inf, Inf];
            followup_end = 6.0
        )
        for mode in (LeakyMode(), AllOrNothingMode())
            vaccine = VaccineEffect(efficacy = 0.4, mode = mode)
            parts = pairwise_surv_loglik_by_component(k, two; susceptibility = vaccine)
            @test length(parts) == 2
            @test sum(parts) ≈ pairwise_surv_loglik(k, two; susceptibility = vaccine)
            @test parts[2] ≈ pairwise_surv_loglik_by_component(k, two)[2]
        end
    end
end
