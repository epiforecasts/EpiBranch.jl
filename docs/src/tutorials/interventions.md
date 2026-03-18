# Interventions

EpiBranch.jl models interventions through a **competing risks** framework.
The branching process generates potential contacts. Each contact's fate is
determined by competing risks: when would transmission occur (generation time)
vs when is the parent isolated (intervention time)?

This connects to survival analysis — the generation time CDF is the survival
function of remaining potential transmission, and isolation truncates it.

## Built-in interventions

### Isolation

[`Isolation`](@ref) isolates symptomatic, test-positive individuals after
a delay from symptom onset. It requires clinical state on individuals,
provided by [`clinical_presentation`](@ref):

```@example interventions
using EpiBranch
using Distributions
using StableRNGs

model = BranchingProcess(Poisson(3.0), Exponential(5.0))
clinical = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
iso = Isolation(delay = Exponential(2.0))

rng = StableRNG(42)
results = simulate_batch(model, 200;
    interventions = [iso],
    attributes = clinical,
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)
println("Containment: $(round(containment_probability(results), digits=3))")
```

The effectiveness depends on how quickly isolation happens relative to the
generation time. Faster isolation = more of the infectious period truncated:

```@example interventions
for d in [0.5, 2.0, 10.0]
    iso = Isolation(delay = Exponential(d))
    rng = StableRNG(42)
    results = simulate_batch(model, 200;
        interventions = [iso],
        attributes = clinical,
        sim_opts = SimOpts(max_cases = 500),
        rng = rng,
    )
    println("Delay ~ Exp($d): containment = $(round(containment_probability(results), digits=3))")
end
```

#### Leaky isolation

With `residual_transmission > 0`, isolated individuals still transmit at
a reduced rate (e.g. household contacts):

```@example interventions
iso_leaky = Isolation(delay = Exponential(2.0), residual_transmission = 0.3)

rng = StableRNG(42)
results = simulate_batch(model, 200;
    interventions = [iso_leaky],
    attributes = clinical,
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)
println("Leaky isolation: $(round(containment_probability(results), digits=3))")
```

### Contact tracing

[`ContactTracing`](@ref) identifies contacts of isolated cases. With
quarantine, traced contacts can be isolated before symptom onset:

```@example interventions
iso = Isolation(delay = Exponential(2.0))
ct = ContactTracing(probability = 0.7, delay = Exponential(1.0), quarantine_on_trace = true)

rng = StableRNG(42)
results = simulate_batch(model, 200;
    interventions = [iso, ct],
    attributes = clinical,
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)
println("Isolation + tracing: $(round(containment_probability(results), digits=3))")
```

## Asymptomatic cases and test sensitivity

Asymptomatic cases escape symptom-based surveillance. Imperfect testing
means some symptomatic cases are also missed. Both are configured via
[`clinical_presentation`](@ref):

```@example interventions
clinical_hard = compose(
    clinical_presentation(
        incubation_period = LogNormal(1.5, 0.5),
        prob_asymptomatic = 0.3,
    ),
    testing(sensitivity = 0.8),
)

rng = StableRNG(42)
results = simulate_batch(model, 200;
    interventions = [iso, ct],
    attributes = clinical_hard,
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)
println("30% asymptomatic, 80% test sensitivity: $(round(containment_probability(results), digits=3))")
```

## Effort tracking

Because all contacts are stored (infected and non-infected), intervention
effort is fully trackable:

```@example interventions
rng = StableRNG(42)
state = simulate_conditioned(model, 50:200;
    interventions = [iso, ct],
    attributes = clinical,
    sim_opts = SimOpts(max_cases = 200),
    rng = rng,
)

total = length(state.individuals)
infected = count(is_infected, state.individuals)
traced = count(is_traced, state.individuals)
println("Contacts: $total, Infections: $infected, Traced: $traced")
println("Contacts per case: $(round(total / infected, digits=1))")
```

## Writing a custom intervention

Any struct subtyping [`AbstractIntervention`](@ref) can be an intervention.
Implement one or more of:
- `initialise_individual!` — set up fields on new contacts
- `resolve_individual!` — determine state before transmission
- `apply_post_transmission!` — act on contacts after creation

```@example interventions
# A gathering limit that caps the number of contacts per individual
struct GatheringLimit <: AbstractIntervention
    max_contacts::Int
end

function EpiBranch.apply_post_transmission!(gl::GatheringLimit, state, new_contacts)
    # Count contacts per parent, mark excess as not infected
    parent_counts = Dict{Int, Int}()
    for c in new_contacts
        count = get(parent_counts, c.parent_id, 0) + 1
        parent_counts[c.parent_id] = count
        if count > gl.max_contacts
            c.state[:infected] = false
        end
    end
end

# Test it
gl = GatheringLimit(5)
rng = StableRNG(42)
results_gl = simulate_batch(
    BranchingProcess(NegBin(2.5, 0.16), Exponential(5.0)), 200;
    interventions = [gl],
    sim_opts = SimOpts(max_cases = 500),
    rng = rng,
)
println("With gathering limit (max 5): $(round(containment_probability(results_gl), digits=3))")
```
