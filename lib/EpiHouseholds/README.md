# EpiHouseholds

Household-structured transmission for
[EpiBranch.jl](https://github.com/epiforecasts/epiBranch.jl), built on its
public extension API.

A `HouseholdProcess` partitions the population into households. Within a
household, every infectious member can infect every susceptible household-mate;
the timing of infectious contact is drawn from a **contact-interval** kernel
(Kenah 2011), and transmission happens while the infector is infectious — the
window the model's `progression` opens and closes. The contact interval is the
one required input; the latent period, infectious period, symptom onset and
testing are flexible `Transition`s on the shared natural-history timeline, so a
household line list is an EpiBranch line list.

`HouseholdProcess` is a structure-driven `TransmissionModel`, like
`NetworkProcess` in [epiNetwork](../epiNetwork). Unlike the generation-based
engine, it simulates by the **Sellke construction** in continuous time — the
exact generative model of its pairwise likelihood — because a finite, depleting
clique needs time-ordered competing-risk resolution.

## Inference

Infections are latent — the model generates them, and the progression maps each
to its observable outcomes (onset, test). So the contact-process likelihood
scores the *infection layer*, not the observed onsets:

```julia
state = simulate(model)
data  = household_infections(state, model)   # the latent infection layer
loglikelihood(data, model)                   # contact-process density, via pairwise_surv_loglik
```

`pairwise_surv_loglik(kernel, rows)` is the order-free, right-censored
counting-process primitive underneath. In real inference the infection times are
**latent**: a Turing `@model` augments them, scores them with the contact
process, and conditions the observed onsets/tests through the progression's
delays. The inference-friendly `pairwise_surv_loglik(kernel, data; external_hazard)`
takes the kernel and infection layer separately, so the fitted parameters and the
augmented state vary without rebuilding the model:

```julia
@model function household_fit(onset, household_of, is_index; obs_end)
    logβ ~ Normal(-1, 1)                                   # within-household log-rate
    incubation ~ ...                                       # latent natural-history params
    infection_time  ~ ...                                  # augment the latent infections
    infectious_time = infection_time                       # (+ latent period, if any)
    removal_time    = infectious_time .+ 6.0               # infectious window
    infections = HouseholdInfections(household_of, infection_time,
        infectious_time, removal_time, is_index; obs_end)
    @addlogprob! pairwise_surv_loglik(Exponential(1 / exp(logβ)), infections)  # contact process
    onset ~ ... infection_time .+ incubation ...           # progression observation, fit to data
end
```

## Status

Work in progress. The `HouseholdProcess` model, its continuous-time (Sellke)
`simulate`, and the pairwise contact-process likelihood — including the external
(community) hazard term, scalar or time-varying — are in place, with an exact
`simulate → loglikelihood` round trip. Still to come: helpers for the
latent-infection `@model`, and the docs/CI wiring into the monorepo.
