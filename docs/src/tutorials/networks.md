# Network models

How does an outbreak spread when each person can only infect the people they
are actually in contact with, and how much do isolation and contact tracing
achieve on that contact structure? `NetworkProcess` answers this for a fixed
contact network.

The model works as follows:

- The population is a fixed set of people, and a contact network says who is
  in contact with whom. Each person is a *node* of the network, and each link
  (an *edge*) joins two people who can infect each other.
- Each person can be infected once.
- An infected person is infectious for a period set by the disease timeline.
  For each of their contacts, the time until they would make an infecting
  contact is drawn from a **contact-interval distribution** (passed as the
  `kernel` argument): the time from the start of the infector's infectious
  period to a contact with that person that would transmit if nothing
  intervened.
- Transmission happens only if that contact falls inside the infector's
  infectious period. If the infector has recovered by then, or is in
  isolation at the time, the
  contact does not infect.

The contact interval is a waiting time for one pair of people. It is not the
generation time or the serial interval, which also depend on the latent and
infectious periods and on how many contacts a case has. An `Exponential(θ)`
contact interval (mean θ days) means a constant rate of infecting contacts of
1/θ per day along each link.

There is no reproduction number to set. In a [`BranchingProcess`](@ref) the
offspring distribution says how many people each case infects. Here the number
follows from how many contacts each person has, the contact-interval
distribution and the infectious period. Because transmission depends on how
long someone is infectious, ending the infectious period early prevents onward
transmission, which a model with a fixed per-link probability of transmission
cannot represent.

The network models live in the companion `EpiNetwork` package.

## Defining a network

`NetworkProcess` takes the contact network and the contact-interval
distribution. The network is a list with one entry per person, numbered from
1: entry `i` holds the numbers of the people person `i` is in contact with.
In the examples on this page every link shares the same contact-interval
distribution; it can also depend on who is in contact with whom (see
[Fitting on a network](#Fitting-on-a-network)).

The disease timeline is a `progression` of [`Transition`](@ref)s, attached
with a [`ModelSpec`](@ref):

- an optional latent period, `Transition(:infectious; from = :infection, …)`;
- an infectious period, ending in recovery, marked `terminal = true`.

Contact intervals are timed from the start of the infectious period: from
becoming infectious if you give a latent period, otherwise from infection.

The code below builds a small population by hand: 20 households of four
people, in which everyone is in contact with everyone else in their household.
Each household is linked to the next by one contact between two of their
members, which closes the households into a ring. You can skip the details of
`household_ring`; it only produces the list of contacts.

```@example networks
using EpiBranch
using EpiNetwork
using Distributions
using StableRNGs

function household_ring(n_households, household_size)
    n = n_households * household_size
    contacts = [Int[] for _ in 1:n]          # an empty contact list per person
    for h in 0:(n_households - 1)
        members = (h * household_size + 1):(h * household_size + household_size)
        for i in members, j in members
            if i != j                        # nobody is their own contact
                push!(contacts[i], j)        # add j to i's contacts
            end
        end
        # one contact between this household and the next
        a = h * household_size + 1
        b = (mod(h + 1, n_households)) * household_size + 1
        push!(contacts[a], b)
        push!(contacts[b], a)
    end
    return [sort(unique(c)) for c in contacts]
end

adjacency = household_ring(20, 4)
model = ModelSpec(NetworkProcess(adjacency, Exponential(3.0));
    progression = [
        Transition(:infectious; from = :infection, delay = LogNormal(1.6, 0.5)),
        Transition(:recovered; from = :infectious, delay = 7.0, terminal = true),
    ])
```

Here the latent period is `LogNormal(1.6, 0.5)`, whose parameters are the mean
and standard deviation of the *log* of the delay, giving a median of about 5
days. Each case is then infectious for 7 days. With a mean contact interval of
3 days, a case infects any one of its contacts with probability
1 - exp(-7/3), about 90%, if nothing intervenes. Most links transmit, but not
all.

A matrix works too: `NetworkProcess(A, kernel)` treats any nonzero `A[i, j]`
as a two-way contact between `i` and `j`, whatever `A[j, i]` holds. Values in
the matrix are ignored; the matrix only says who is in contact, and every
contact shares the same contact-interval distribution.

!!! note "One-way contacts"
    The network can be directed, for one-way contacts such as a carer visiting
    a patient. Then person `i`'s list holds the people `i` can infect, who
    need not be the people who can infect `i`. The matrix form always makes
    contacts two-way, so build one-way contacts as a list or as a directed
    Graphs.jl graph.

## Generating a network with Graphs.jl

Building a contact list by hand suits small or bespoke structures. For
realistic contact networks it is easier to use the generators in
[Graphs.jl](https://juliagraphs.org/Graphs.jl/), the standard Julia graph
library. Run `using Graphs` first and you can pass a graph straight to
`NetworkProcess`: each vertex of the graph becomes a person and its neighbours
become their contacts. The list and matrix forms above need nothing extra.

```@example networks
using Graphs

# A small-world network of 400 people: mostly local contacts, so that the
# contacts of a person tend to be in contact with each other, with a few
# long-range links.
g = watts_strogatz(400, 6, 0.1; rng = StableRNG(1))

model_ws = ModelSpec(NetworkProcess(g, Exponential(3.0));
    progression = [Transition(:recovered; from = :infection, delay = 7.0, terminal = true)])
state = simulate(model_ws; n_initial = 1, rng = StableRNG(3))
println("Final size: ", state.cumulative_cases, " of ", nv(g))
```

With most links transmitting, a single index case reaches the whole network.

Any generator that returns a graph works, and the choice of graph is a
modelling assumption. A few that map onto common assumptions, for `n` people:

- `watts_strogatz(n, d, p)`: small-world. Each person starts linked to their
  `d` nearest neighbours on a ring, and each link is rewired at random with
  probability `p`, giving local clustering with a few long-range links.
- `barabasi_albert(n, d)`: scale-free. The number of contacts per person is
  heavy-tailed: a minority of highly connected people drive spread. This is
  roughly the network counterpart of an overdispersed offspring distribution
  (small k).
- `stochastic_block_model(...)`: groups that are densely connected inside and
  sparsely connected between them, a natural fit for households or
  communities.
- `euclidean_graph(n, dims; cutoff)`: people placed at random in space and
  linked when they are close. Clustering then comes from proximity, as in
  spatial outbreak models. It returns the graph and the distances together;
  take the graph with `g, _ = euclidean_graph(...)`.

The contact-interval distribution, the disease timeline and population
characteristics attach in the same way whatever the source of the graph.

## Which interventions work on a network

| Intervention | What it does on a network |
|:-- | :-- |
| Isolation of cases: [`Isolation`](@ref), or an `:isolated` transition | Stops the case transmitting while they are isolated. An `:isolated` transition, or `Isolation(duration = Inf)`, ends their infectious period; with a finite `duration` a case still infectious when released transmits again ([below](#Several-routes-at-once)). |
| Partial protection: leaky isolation, a vaccine's efficacy | Blocks a fraction of the infecting contacts. The pair keeps meeting, and blocking a fraction `p` of contacts lowers the rate of infecting contacts along that link by the factor `1 - p`. |
| Individual differences in susceptibility or infectiousness | Multiply the rate of infecting contacts along each of that person's links by the person's factor. |
| [`ContactTracing`](@ref) | Traces a case's contacts in the network and quarantines them ([below](#Contact-tracing-on-a-network)). |
| [`RingVaccination`](@ref), [`GroupVaccination`](@ref) | Supported, including delays to vaccination and capacity limits, with the limits in the box below. |
| [`MassVaccination`](@ref) | Not supported on networks. |

Protection that people already have before the outbreak, such as prior
immunity, can be given as a population characteristic, through a
contact-interval distribution that depends on who is in contact with whom (see
[Covariates and time-varying transmission](@ref)), or as an intervention of
your own (see [Extending EpiBranch](extending.md)).

!!! warning "Limits on interventions on a network"
    - Mass vaccination is not supported. When a model has an intervention that
      the network cannot apply, `simulate` warns and runs as if that
      intervention were absent.
    - Ring vaccination works only without an `eligibility_window` (the
      default). Checking whether a contact was exposed recently enough needs
      their infection time, which is not yet known when they are vaccinated.
      With a window set, `simulate` warns and ignores the ring vaccination.
    - Tracing is forward only. When a case is detected, the people they are in
      contact with can be traced. Backward (source) tracing, to find the
      person who infected the case, is not supported.
    - Whether a contact is protected when exposed depends on whether they had
      been traced or vaccinated by then. Once a contact has been given a ring
      vaccination date, tracing them again later does not change it. A case
      whose infection has already happened in the simulation is not changed by
      anything learned later.

## Simulating

[`simulate`](@ref) runs one outbreak and returns its full record, and
[`linelist`](@ref) turns that record into a table with one row per case.

```@example networks
rng = StableRNG(42)
state = simulate(model; n_initial = 1, rng = rng)

println("Final outbreak size: ", state.cumulative_cases, " of ", length(adjacency))
```

`state.cumulative_cases` is the total number of people infected. The
population is the network, so an outbreak cannot grow beyond the number of
people in it: susceptible people run out.

One run is one possible outbreak. To see the distribution of final sizes, run
the outbreak many times, each with a different random seed (here 200 runs,
seeds 1 to 200):

```@example networks
sizes = [simulate(model; n_initial = 1, rng = StableRNG(i)).cumulative_cases
         for i in 1:200]
println("Mean size: ", round(sum(sizes) / length(sizes), digits = 1))
```

The mean is below the population of 80 because some outbreaks stop before
leaving the first household or two.

## Age, risk group and other individual characteristics

Each person's characteristics, such as age or risk group, are drawn once at
the start and do not change during the outbreak. They appear as columns of the
line list.

```@example networks
attrs = [
    demographics(age_distribution = Uniform(0, 80)),
    clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
]

model_attrs = ModelSpec(NetworkProcess(adjacency, Exponential(3.0));
    progression = [Transition(:recovered; from = :infection, delay = 7.0, terminal = true)],
    attributes = attrs)
state = simulate(model_attrs; n_initial = 1, rng = StableRNG(7))

df = linelist(state)
println("Cases: ", size(df, 1),
    "; mean age: ", round(sum(df.age) / size(df, 1), digits = 1))
```

Transmission here does not depend on age, and the cases' mean age is close to
that of the whole population, 40 years.

## Isolation curtails onward spread

Transmission can happen only during the infectious period. A case who is
isolated soon after becoming infectious infects fewer of their contacts. The
simplest way to model this is to add an `:isolated` transition to the disease
timeline: every case isolates after a delay, and isolation ends their
infectious period. By default a network's infectious period ends at whichever
comes first of recovery, death and isolation. For isolation triggered by
symptom onset, with testing and only some cases detected, use the
[`Isolation`](@ref) intervention instead, as in the next section.

Here a fast contact interval (mean 1.5 days) spreads through the whole ring of
households when nothing stops it. Isolating each case after a delay with a
mean of 2 days from infection holds the outbreak back.

```@example networks
kernel = Exponential(1.5)

baseline = ModelSpec(NetworkProcess(adjacency, kernel);
    progression = [
        Transition(:recovered; from = :infection, delay = 10.0, terminal = true)])

isolating = ModelSpec(NetworkProcess(adjacency, kernel);
    progression = [
        Transition(:recovered; from = :infection, delay = 10.0, terminal = true),
        Transition(:isolated; from = :infection, delay = Exponential(2.0))])

base_sizes = [simulate(baseline; n_initial = 1, rng = StableRNG(i)).cumulative_cases
              for i in 1:200]
iso_sizes = [simulate(isolating; n_initial = 1, rng = StableRNG(i)).cumulative_cases
             for i in 1:200]

println("Mean size, no isolation:   ",
    round(sum(base_sizes) / length(base_sizes), digits = 1))
println("Mean size, with isolation: ",
    round(sum(iso_sizes) / length(iso_sizes), digits = 1))
```

Without isolation every outbreak infects the whole population. With isolation,
most outbreaks stay within a few households.

## Contact tracing on a network

In a branching process a case's contacts are new people, created as they are
needed. On a network the contacts are fixed people, and different cases share
contacts. The same person can be named by several cases, and in a clustered
network tracing often finds people who have already been found.

[`ContactTracing`](@ref) works on a network as it does elsewhere. A traced
contact is quarantined when they are traced, for the `duration` given to
the `Quarantine` action. If they are infected, or become infected later, they
do not transmit while in quarantine.

In the example below, cases develop symptoms after an incubation period with a
median of about 3 days (`LogNormal(1.0, 0.3)`) and isolate after a further
delay with a mean of 2 days. Isolation lasts 7 days (`duration = 7.0`), and every
case is still isolated when their 7-day infectious period ends. With probability `p`, each of an isolated case's
contacts is traced and quarantined, after a delay with a mean of 1 day from
the case's isolation, and stays in quarantine for good
(`Quarantine(duration = Inf)`). The network is a small world of 400 people with 6
contacts each. With a mean contact interval of 16 days and a 7-day infectious
period, each contact is infected with probability 1 - exp(-7/16), about a
third. Each case then infects around two people if nothing intervenes.

```@example networks
clinical = clinical_presentation(incubation_period = LogNormal(1.0, 0.3),
    prob_asymptomatic = 0.0)
iso = Isolation(onset_to_isolation_delay = Exponential(2.0), test_sensitivity = 1.0,
    duration = 7.0)

ws = watts_strogatz(400, 6, 0.1; rng = StableRNG(1))

# The same model with different sets of interventions
build(interventions) = ModelSpec(NetworkProcess(ws, Exponential(16.0));
    progression = [Transition(:recovered; from = :infection, delay = 7.0,
        terminal = true)],
    interventions = interventions, attributes = clinical)

# Mean final size over 100 simulated outbreaks
function mean_final_size(interventions)
    sizes = [simulate(build(interventions); n_initial = 1,
                 rng = StableRNG(s)).cumulative_cases for s in 1:100]
    return round(sum(sizes) / length(sizes), digits = 1)
end

no_interventions = AbstractIntervention[]   # an empty list of interventions

println("no control:               ", mean_final_size(no_interventions))
println("isolation:                ", mean_final_size([iso]))
for p in (0.5, 1.0)
    ct = ContactTracing(probability = p, isolation_to_trace_delay = Exponential(1.0),
        action = Quarantine(duration = Inf))
    println("isolation + $(round(Int, 100 * p))% tracing:  ", mean_final_size([iso, ct]))
end
```

Isolation alone cuts the mean outbreak size sharply. Tracing reduces it
further, by more when a larger share of contacts is traced.

## Several routes at once

In the models above every link works the same way: one contact-interval
distribution and one infectious period. Anything that ends the infectious
period therefore stops all transmission at once. That does not fit the most
common control measure: someone who self-isolates stops mixing in the
community but goes on infecting the people they live with.

`RoutedNetwork` gives different kinds of contact, here household and
community, each their own contact network, contact-interval distribution and
rule for when transmission along them stops. Each kind of contact is a
*route*, given as a [`RouteWindow`](@ref). Self-isolation can then stop
community transmission while household transmission continues.

The `until` argument of each route answers one question: does isolation stop
transmission on this route?

- `until = (:recovered,)`: no. Only recovery ends transmission on this route.
  For a household route this is self-isolation at home. The trailing comma is
  needed to make a list of one item.
- `until = (:recovered, EpiBranch.INTERVENTION_REMOVAL)`: yes. Transmission
  on this route stops while the case is isolated or quarantined, and ends at
  recovery. With `duration = Inf` isolation ends it for good; with a finite
  `duration`, a case who is still infectious when released transmits on this
  route again.

The population below has 150 households of four, in which everyone is in
contact with everyone else, plus a sparser random network of community
contacts over the same 600 people, about six per person. Again you can skip
the details of the function that builds them.

```@example networks
function households_and_community(n_households, household_size, rng)
    n = n_households * household_size
    household = [Int[] for _ in 1:n]
    for h in 0:(n_households - 1)
        members = (h * household_size + 1):(h * household_size + household_size)
        for i in members, j in members
            if i != j
                push!(household[i], j)
            end
        end
    end
    community = [Int[] for _ in 1:n]
    for _ in 1:(3 * n)                   # 3n random links, about 6 per person
        a = rand(rng, 1:n)
        b = rand(rng, 1:n)
        if a != b && !(b in community[a])
            push!(community[a], b)
            push!(community[b], a)
        end
    end
    return household, community
end

hh_adj, comm_adj = households_and_community(150, 4, StableRNG(99))

clinical2 = clinical_presentation(incubation_period = LogNormal(0.5, 0.3),
    prob_asymptomatic = 0.0)
iso2 = Isolation(onset_to_isolation_delay = Exponential(1.0), test_sensitivity = 1.0,
    duration = Inf)

stopped_by_isolation = (:recovered, EpiBranch.INTERVENTION_REMOVAL)
not_stopped_by_isolation = (:recovered,)

# The scenarios differ only in whether isolation stops household
# transmission. Isolation always stops community transmission.
routes(household_until) = [
    RouteWindow(:household; until = household_until,
        kernel = Weibull(1.5, 4.0), reach = hh_adj),
    RouteWindow(:community; until = stopped_by_isolation,
        kernel = Exponential(30.0), reach = comm_adj)]

# Mean final size over 80 simulated outbreaks, each started by 3 index cases
function mean_size(route_list, interventions)
    m = ModelSpec(RoutedNetwork(route_list);
        progression = [Transition(:recovered; from = :infection, delay = 10.0,
            terminal = true)],
        interventions = interventions, attributes = clinical2)
    sizes = [simulate(m; n_initial = 3, rng = StableRNG(s)).cumulative_cases
             for s in 1:80]
    return round(sum(sizes) / length(sizes), digits = 1)
end

println("no control:                                 ",
    mean_size(routes(not_stopped_by_isolation), AbstractIntervention[]))
println("isolation away from home (all routes stop): ",
    mean_size(routes(stopped_by_isolation), [iso2]))
println("self-isolation at home:                     ",
    mean_size(routes(not_stopped_by_isolation), [iso2]))
```

The household contact interval is `Weibull(1.5, 4.0)` (shape 1.5 and scale 4
days, a mean of about 3.6 days). The community contact interval has a mean of
30 days, which makes each community link much less likely to transmit than a
household one. Isolation lasts for the rest of the infectious period
(`duration = Inf`).

Self-isolation at home prevents fewer cases than isolation away from home,
because household transmission continues. A model with a single infectious
period can only represent isolation away from home. Using one to stand for
self-isolation overstates what self-isolation achieves.

The basic reproduction number, what a case would achieve if never isolated, is
the same in all three scenarios. Isolation changes the number of people each
case actually infects, by an amount that depends on which routes it stops.

### Which contacts can be traced

Routes also differ in which contacts a case can name to contact tracers. A
case can name everyone in their household, but most community contacts are
strangers. `traceable` on a route is the probability that a case can name a
contact made on it (default 1). Contact tracing then reaches a named contact
with its own `probability`, so the chance that a community contact is traced
is `traceable × probability`. Someone who is both a household and a community
contact is named with the higher of the two `traceable` probabilities, since a
case can name the people they live with whether or not they also meet them
elsewhere.

```@example networks
# Slower isolation and more community contact than above, so that tracing has
# transmission left to prevent.
iso3 = Isolation(onset_to_isolation_delay = Exponential(4.0), test_sensitivity = 1.0,
    duration = Inf)
ct3 = ContactTracing(probability = 0.9, isolation_to_trace_delay = Exponential(0.5),
    action = Quarantine(duration = Inf))
traced_routes(community_traceable) = [
    RouteWindow(:household; until = stopped_by_isolation,
        kernel = Weibull(1.5, 4.0), reach = hh_adj),
    RouteWindow(:community; until = stopped_by_isolation,
        kernel = Exponential(15.0), reach = comm_adj,
        traceable = community_traceable)]

println("isolation only:                               ",
    mean_size(traced_routes(1.0), [iso3]))
traced = [mean_size(traced_routes(p), [iso3, ct3]) for p in (0.0, 0.5, 1.0)]
for (p, size) in zip((0.0, 0.5, 1.0), traced)
    println("isolation + tracing, community traceable $p: ", size)
end
issorted(traced; rev = true) && last(traced) < first(traced) / 2 ||  # hide
    error("the paragraph below reads these numbers as falling with " *  # hide
        "traceability, and they no longer do: $traced")  # hide
nothing  # hide
```

Tracing household contacts alone already helps. The more community contacts a
case can name, the more tracing prevents. Treating every route as fully
traceable overstates what tracing achieves whenever much of the transmission
happens between people who cannot name each other.

## Community introductions

Without introductions from outside, the outbreak starts from the index cases
and spreads only along the network. `external_hazard` adds a constant force of
infection from outside the network: the rate, per susceptible person per day,
of being infected from the community. `external_hazard = 0.02` gives each
susceptible person a chance of about 2% per day. `obs_end` is the day after
which no more introductions happen; spread along the network continues after
it.

```@example networks
model_ext = ModelSpec(NetworkProcess(adjacency, Exponential(3.0);
        external_hazard = 0.02, obs_end = 60.0);
    progression = [Transition(:recovered; from = :infection, delay = 7.0, terminal = true)])
state = simulate(model_ext; rng = StableRNG(11))
df = linelist(state)

println("Cases: ", size(df, 1),
    "; community introductions: ", count(df.index))
```

The line list's `index` column is `true` for cases with no infector in the
network: here, the community introductions. The other cases were infected
along the network, by someone in their own household or a neighbouring one.

## Fitting on a network

A `NetworkProcess` model can be fitted to data as well as simulated.

**Data you need:**

- the contact network;
- who was infected and when;
- for each case, when their infectious period started and when it ended, by
  recovery or isolation, and any stretch of isolation or quarantine they were
  released from before it ended;
- which cases started the outbreak (the index cases), unless you fit a
  community hazard.

You do not need to know who infected whom.

**What you can estimate:** the parameters of the contact-interval
distribution, such as its mean, which sets the rate of infecting contacts
along each link, and, if you include one, the rate of community introductions.

The method is the pairwise survival likelihood of [Kenah
(2011)](https://doi.org/10.1093/biostatistics/kxq068), which the [household
models](households.md) also use. Each person is at risk from each of their
contacts for as long as that contact is infectious. For each case, the
likelihood counts the combined risk from all their contacts who were
infectious at the time they were infected. The simulation and the likelihood
describe the same model, and fitting a simulated outbreak should recover the
values it was simulated with.

[`network_infections`](@ref) extracts these data from a simulated outbreak,
and [`pairwise_surv_loglik`](@ref) evaluates the log-likelihood of a given
contact-interval distribution. If the model's interventions isolate or
quarantine a case, the data record when that happened, and their contacts
are not at risk from them during it. A case isolated for good counts as infectious only
up to isolation.

!!! warning "Partial protection is not in the data"
    Isolation that still lets some transmission through is not recorded in
    the data. `pairwise_surv_loglik` ignores it, which pulls the estimated
    rate of infecting contacts down, and `loglikelihood(data, model)` stops
    with an error for such a model. A vaccine's efficacy against infection is
    allowed for by `loglikelihood(data, model)` with the vaccination in
    `model`.

There is no one-call fitting function. Here we evaluate the log-likelihood at
a grid of mean contact intervals and take the best one, the maximum likelihood
estimate, for an outbreak simulated with a mean of 6 days on a small-world
network of 2000 people:

```@example networks
g_fit = watts_strogatz(2000, 6, 0.1; rng = StableRNG(5))
truth = ModelSpec(NetworkProcess(g_fit, Exponential(6.0));
    progression = [
        Transition(:infectious; from = :infection, delay = LogNormal(0.5, 0.3)),
        Transition(:recovered; from = :infectious, delay = 5.0, terminal = true)])
data = network_infections(simulate(truth; n_initial = 5, rng = StableRNG(6)), truth)

# log-likelihood of an exponential contact interval with mean `scale` days
ll(scale) = pairwise_surv_loglik(Exponential(scale), data)
grid = 3.0:0.25:9.0
grid[argmax(ll.(grid))]                # ll.(grid) evaluates ll at every grid value
```

The estimate is close to the 6 days the outbreak was simulated with.
`loglikelihood(data, truth)` gives the log-likelihood at the model's own
contact-interval distribution.

The contact-interval distribution can be the same for every link, depend on
who is in contact with whom (a function of the infector's and the susceptible
person's numbers, `(infector, susceptible) -> Distribution`, for example to
let it depend on age), or be set link by link (a list laid out like the
contact list). [Covariates and time-varying transmission](@ref) covers
distributions that also depend on the calendar or on events during the
outbreak. The [household models](households.md) page shows how to fit with
Bayesian inference in Turing.

A network within households, in which not everyone in a household is in
contact with everyone else, is fitted the same way: give the actual
within-household contacts as the network. On a directed network, the people
who can infect person `i` are those who list `i` as a contact.

!!! warning "Data that stop before the outbreak ends (right-censoring)"
    If your data stop at a date, pass that date as `followup_end` to
    `network_infections` (or to `NetworkInfections` when you build the data
    yourself). Infections and exposure after that date are then ignored, and
    cases still infectious at that date are treated as not yet recovered.
    Without it, the likelihood assumes that follow-up continued until the
    outbreak was over.

!!! note "Community introductions and index cases"
    With an `external_hazard`, pass the same value to `pairwise_surv_loglik`
    (`pairwise_surv_loglik(kernel, data; external_hazard = 0.02)`). The
    community then has to explain the cases with no infectious contact, up to
    the data's `obs_end`. Without one, those index cases are taken as given. A
    model with some community transmission and a model with none cannot be
    compared by letting `external_hazard` shrink towards zero: fit the two
    separately. [Fitting a community hazard](@ref) explains why.

!!! warning "What cannot be fitted yet"
    - Models with several routes (`RoutedNetwork`) have no likelihood yet.
    - In real data infection times are not observed. They have to be estimated
      together with the parameters, from symptom onsets and the disease
      timeline (data augmentation), as described for [household
      models](households.md). This page has no worked example of that.
    - If the log-likelihood is `-Inf`, the data contain a case the model
      cannot explain, such as someone infected when none of their contacts was
      infectious and there is no community hazard.
    - To fit a `Gamma` contact interval or community hazard with Turing, use
      `NUTS(; adtype = AutoMooncake())`, with the Mooncake package installed
      and loaded (`using Mooncake`); the default sampler settings fail for
      `Gamma`. `Exponential` and `Weibull` work with the defaults.
