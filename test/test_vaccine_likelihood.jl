# An infection layer carrying `immunity_time`, as a companion package would
# define once this field exists (`_TestInfections` in test_pairwise_likelihood.jl
# predates it, and is reused there to check the no-field fallback).
struct _VaxInfections{S} <: InfectionLayer
    structure::S
    infection_time::Vector{Float64}
    infectious_time::Vector{Float64}
    removal_time::Vector{Float64}
    is_index::Vector{Bool}
    obs_end::Float64
    followup_end::Float64
    immunity_time::Vector{Float64}
end
function _VaxInfections(
        structure, inf, infectious, removal, index, imm;
        obs_end = Inf, followup_end = Inf
    )
    return _VaxInfections(
        structure, Float64.(inf), Float64.(infectious), Float64.(removal),
        Vector{Bool}(index), Float64(obs_end), Float64(followup_end), Float64.(imm)
    )
end
EpiBranch.contact_structure(d::_VaxInfections) = d.structure

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
        @test pairwise_surv_loglik(k, escaped; vaccine) ≈ expected

        # infected at t = 3, past τ: the same escape term up to 3, and the
        # event hazard at 3 also discounted by 0.6.
        infected = data(3.0)
        expected_event = -(cumh(τ) + 0.6 * (cumh(3.0) - cumh(τ))) +
            (logh(3.0) + log(0.6))
        @test pairwise_surv_loglik(k, infected; vaccine) ≈ expected_event

        # infected before τ: immunity never comes into play, so the result is
        # exactly the unvaccinated computation.
        early = data(0.5)
        @test pairwise_surv_loglik(k, early; vaccine) ≈
            pairwise_surv_loglik(k, early; vaccine = nothing)
    end

    @testset "AllOrNothingMode mixes over responder status" begin
        vaccine = VaccineEffect(efficacy = 0.3, mode = AllOrNothingMode())
        e = 0.3

        # escapes forever: e·(fully protected from τ) + (1-e)·(unprotected).
        escaped = data(NaN; followup_end = 5.0)
        expected = log(e * exp(-cumh(τ)) + (1 - e) * exp(-cumh(5.0)))
        @test pairwise_surv_loglik(k, escaped; vaccine) ≈ expected

        # infected at t = 3, past τ: a responder cannot be, so only the
        # unprotected branch contributes.
        infected = data(3.0)
        expected_event = log(1 - e) + logh(3.0) - cumh(3.0)
        @test pairwise_surv_loglik(k, infected; vaccine) ≈ expected_event

        # infected before τ: both branches agree, so the mixture collapses to
        # the unvaccinated computation exactly (log(e·L + (1-e)·L) = log(L)).
        early = data(0.5)
        @test pairwise_surv_loglik(k, early; vaccine) ≈
            pairwise_surv_loglik(k, early; vaccine = nothing)
    end

    @testset "unvaccinated hosts are untouched by a shared vaccine argument" begin
        # host 2 has no immunity time (Inf): scoring it under a vaccine
        # argument must not change its contribution, whichever mode.
        unvacc = _VaxInfections(
            [1, 1], [0.0, 2.0], [0.0, 2.0], [Inf, Inf],
            [true, false], [Inf, Inf]
        )
        for mode in (LeakyMode(), AllOrNothingMode())
            vaccine = VaccineEffect(efficacy = 0.5, mode = mode)
            @test pairwise_surv_loglik(k, unvacc; vaccine) ≈
                pairwise_surv_loglik(k, unvacc; vaccine = nothing)
        end
    end

    @testset "waning scales the discount continuously" begin
        escaped = data(NaN; followup_end = 5.0)

        # a waning function that never decays reproduces the constant-discount
        # closed form exactly.
        flat = VaccineEffect(efficacy = 0.4, mode = LeakyMode(), waning = dt -> 1.0)
        flat_ll = pairwise_surv_loglik(k, escaped; vaccine = flat)
        no_waning = VaccineEffect(efficacy = 0.4, mode = LeakyMode())
        @test flat_ll ≈ pairwise_surv_loglik(k, escaped; vaccine = no_waning)

        # an exponentially decaying dose, checked against the same integral
        # computed directly with the quadrature the package uses internally.
        decay = dt -> exp(-dt / 2)
        waned = VaccineEffect(efficacy = 0.4, mode = LeakyMode(), waning = decay)
        discounted, _ = EpiBranch.quadgk(
            s -> 0.5 * (1 - 0.4 * decay(s - τ)), τ, 5.0
        )
        expected = -(cumh(τ) + discounted)
        @test pairwise_surv_loglik(k, escaped; vaccine = waned) ≈ expected rtol = 1.0e-6

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
        @test pairwise_surv_loglik(k, infected; vaccine = waned) ≈ expected_event rtol = 1.0e-6
    end

    @testset "differentiable in efficacy" begin
        escaped = data(NaN; followup_end = 5.0)
        for mode in (LeakyMode(), AllOrNothingMode())
            f(θ) = pairwise_surv_loglik(
                k, escaped;
                vaccine = VaccineEffect(efficacy = θ[1], mode = mode)
            )
            fd = ForwardDiff.gradient(f, [0.4])
            step = 1.0e-6
            numeric = (f([0.4 + step]) - f([0.4 - step])) / 2step
            @test only(fd) ≈ numeric rtol = 1.0e-4
        end
    end

    @testset "efficacy must be a fixed value" begin
        drawn = VaccineEffect(efficacy = Beta(2.0, 2.0), mode = LeakyMode())
        @test_throws ArgumentError pairwise_surv_loglik(k, data(NaN); vaccine = drawn)
    end
end
