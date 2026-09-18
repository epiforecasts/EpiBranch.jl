struct LikelihoodWindowRemoval <: EpiBranch.AbstractIntervention end
function EpiBranch.infectious_removal_time(::LikelihoodWindowRemoval, ind::EpiBranch.Individual)
    ind.infection_time + 1.0
end
EpiBranch.infection_likelihood_compatible(::LikelihoodWindowRemoval) = true

struct LikelihoodHazardChange <: EpiBranch.AbstractIntervention end
function EpiBranch.competing_risk(::LikelihoodHazardChange, parent, contact, state)
    Risk(block_probability = 0.5)
end

function test_structured_composition(make_process, extract)
    @testset "Structured likelihood composition" begin
        progression = [Transition(:recovered; from = :infection, delay = 3.0, terminal = true)]
        process = make_process(Exponential(2.0))
        removal = LikelihoodWindowRemoval()
        spec = ModelSpec(process; progression, interventions = [removal])
        state = simulate(spec; rng = StableRNG(44))
        data = extract(state, spec)
        infected = findall(!isnan, data.infection_time)
        @test all(i -> data.removal_time[i] == data.infection_time[i] + 1, infected)
        @test isfinite(loglikelihood(data, spec))
        @test loglikelihood(data, spec) == pairwise_surv_loglik(Exponential(2.0), data)
        clinical = clinical_presentation(incubation_period = Exponential(0.2))
        iso = Isolation(onset_to_isolation_delay = Exponential(0.2))
        supported = ModelSpec(process; progression, attributes = clinical, interventions = [iso])
        sample = extract(simulate(supported; rng = StableRNG(45)), supported)
        @test isfinite(loglikelihood(sample, supported))
        for bad in (LikelihoodHazardChange(),
            Isolation(onset_to_isolation_delay = Exponential(0.2), post_isolation_transmission = 0.5),
            Scheduled(LikelihoodHazardChange(); start_time = 0.0))
            @test_throws ArgumentError loglikelihood(data,
                ModelSpec(process; progression, interventions = [bad]))
        end
        @test_throws ArgumentError loglikelihood(data,
            ModelSpec(process; progression, attributes = transmission_traits(susceptibility = 0.5)))
        # A supplied effective kernel uses the existing differentiable scoring path.
        effective(i, j) = Exponential(4.0)
        @test pairwise_surv_loglik(effective, data) ==
              pairwise_surv_loglik(Exponential(4.0), data)
        f(scale) = loglikelihood(data,
            ModelSpec(make_process(Exponential(scale));
                progression, interventions = [removal]))
        @test isfinite(ForwardDiff.derivative(f, 2.0))
    end
end
