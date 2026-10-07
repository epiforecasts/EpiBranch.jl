# Household models

How far does an infection spread within a household once it gets in, and does
an outbreak of infected households grow or die out? `HouseholdProcess`
simulates transmission within households, and the household reproduction
number R\* and the probability that an introduction dies out follow from it.

The model works as follows:

- The population is divided into households.
- Within a household, every infectious person can infect every susceptible
  person they live with.
- Each person can be infected once, and a household can run out of people to
  infect.
- For each pair of housemates, the time until an infectious person would make
  an infecting contact with the other is drawn from a **contact-interval
  distribution** (passed as the `kernel` argument). The contact interval runs
  from the start of the infector's infectious period to a contact that would
  transmit if nothing intervened ([Kenah,
  2011](https://doi.org/10.1093/biostatistics/kxq068)). It is a waiting time
  for one pair; the generation time and serial interval also depend on the
  latent and infectious periods.
- Transmission happens only if that contact falls inside the infector's
  infectious period. If the infector has recovered by then, or is in
  isolation at the time, the
  contact does not infect.
- A simulation starts each household from one index case. Infection between
  households is not simulated person by person: it is either summarised by R\*
  (see [below](#The-reproduction-number-between-households)) or added as
  introductions from the community (`external_hazard`).

!!! warning "What isolation means here"
    A household model has a single infectious period per case. Isolating a
    case therefore stops all of their transmission, including to the people
    they live with. Read isolation and quarantine on this model as removal
    from the household, such as hospitalisation or a stay in an isolation
    facility, for as long as the `duration` lasts. A case released while still
    infectious goes back to infecting their housemates. Self-isolation at home, where household transmission continues,
    needs household and community contact as separate routes; see
    [Several routes at once](@ref) on the network page.

The household models live in the companion `EpiHouseholds` package.

## Defining a household model

`HouseholdProcess` takes the size of each household and the contact-interval
distribution. The disease timeline is a `progression` of
[`Transition`](@ref)s, attached with a [`ModelSpec`](@ref).

```@example households
using EpiBranch
using EpiHouseholds
using Distributions
using StableRNGs

# fill(4, 300) is a list of 300 fours: 300 households of four people
model = ModelSpec(HouseholdProcess(fill(4, 300), Weibull(1.5, 3.0));
    progression = [Transition(:recovered; from = :infection, delay = 6.0, terminal = true)])
```

The contact interval is a Weibull distribution with shape 1.5 and scale 3
days, a mean of about 2.7 days. Each case is infectious for 6 days from
infection, long enough to infect most housemates.

## Simulating

[`simulate`](@ref) runs the outbreak, and [`linelist`](@ref) turns the result
into a table with one row per case. Each household starts with one index case,
and the outbreak spreads within it.

```@example households
state = simulate(model; rng = StableRNG(1))
df = linelist(state)
(cases = size(df, 1), index_cases = count(df.index),
    secondary_cases = size(df, 1) - count(df.index))
```

There are 300 index cases, one per household, and the secondary cases are the
housemates infected after them.

## The disease timeline

The disease timeline works as for [`BranchingProcess`](@ref):

- a latent period is `Transition(:infectious; from = :infection, …)`;
- the infectious period ends in recovery, marked `terminal = true`;
- symptom onset, testing and other events are further transitions.

Contact intervals are timed from the start of the infectious period: from
becoming infectious if you give a latent period, otherwise from infection.
Each state in the timeline becomes a `date_...` column of the line list.

```@example households
using DataFrames

clinical = ModelSpec(HouseholdProcess(fill(4, 300), Weibull(1.5, 3.0));
    progression = [
        Transition(:infectious; from = :infection, delay = LogNormal(1.2, 0.4)),  # latent period
        Transition(:recovered; from = :infectious, delay = Gamma(6, 1),           # infectious period
            terminal = true)])
names(linelist(simulate(clinical; rng = StableRNG(2))))
```

The latent period `LogNormal(1.2, 0.4)` takes the mean and standard deviation
of the *log* of the delay, giving a median of about 3.3 days. The infectious
period `Gamma(6, 1)` (shape 6, scale 1 day) has a mean of 6 days. The table
gets `date_infectious` and `date_recovered` columns for the two states.

## The reproduction number between households

A household model describes what happens inside a household. Spread *between*
households is a branching process of its own, in which the unit is a whole
household: an infected household infects other households through its members'
contacts in the community. Its reproduction number is the **household
reproduction number R\***: the mean number of other households that one
infected household infects. An epidemic of households can grow only if
R\* > 1. R\* is not the same as the individual reproduction number, because
it counts the community infections made by everyone infected in the
household.

[`household_offspring`](@ref) gives the distribution of the number of other
households one infected household infects. It needs one number that the
household model does not have:

- `global_rate`: the rate, per day, at which an infectious person makes
  infecting contacts with people outside their household.

Early in an epidemic, each such contact reaches a susceptible person in a
household with no infection yet. The number of households one household
infects is then Poisson, with mean `global_rate` times the total number of
person-days that its members are infectious. That total varies from one
household to the next, because the outbreak within the household is random.

```@example households
offspring = household_offspring(model; global_rate = 0.1, rng = StableRNG(5))
reproduction_number(offspring)
```

With 0.1 infecting community contacts per day, each infected person makes on
average 0.6 over their 6 infectious days. Most of the household is infected,
which makes R\* several times larger than that.

The distribution itself is a `Distributions.jl` distribution, so it can be
plotted, sampled, or used as the offspring distribution of a
[`BranchingProcess`](@ref) to simulate chains of infected households. Here are
the probabilities that an infected household infects 0, 1 or 2 others:

```@example households
law = household_offspring_law(offspring)
(pdf(law, 0), pdf(law, 1), pdf(law, 2))
```

!!! warning "The early phase of an epidemic only"
    R\* assumes that every community contact reaches a household with no
    infection yet. This holds while infected households are a small share of
    the total. The supply of uninfected households never runs out. The model
    therefore covers the early phase: R\*, and the probability that a single
    introduction dies out. It cannot give the peak of the epidemic or its
    final size in the whole population, which need a finite number of
    households that can run out.

### Households of different sizes

With households of different sizes, this is a multi-type branching process in
which a household's type is its size. A community contact reaches a *person*,
and with them their household. Larger households are therefore reached more
often than their share of households would suggest (size-biased sampling).
They then go on to infect more households, because more of their members are
infected. Both effects are in the result:

```@example households
mixed = ModelSpec(HouseholdProcess([fill(2, 400); fill(5, 200)], Weibull(1.5, 12.0));
    progression = [Transition(:recovered; from = :infection, delay = 6.0,
        terminal = true)])
sized = household_offspring(mixed; global_rate = 0.1, rng = StableRNG(6))
(sizes = sized.sizes, mixing = sized.mixing, means = sized.means)
```

For each household type:

- `sizes` is the household size;
- `mixing` is the probability that a community contact lands in a household of
  that type;
- `means` is the mean number of other households that a household of that type
  infects.

Here 400 households have two people and 200 have five, and the contact
interval is slower (Weibull scale 12 days, a mean of about 11 days), and not
every housemate is infected. Two-person households are two thirds of the
households but hold under half of the people. Community contacts reach them
less than half the time, and each five-person household infects more others.

[`extinction_probability`](@ref) answers the question most often asked of a
household model: if one household is infected, how likely is it that
transmission dies out without an epidemic? It gives one probability per type
of the first household, since only the first household's type is not set by
who community contacts reach.

```@example households
extinction_probability(sized)
```

Introductions into larger households are less likely to die out.

### Comparing with a multi-type branching process

The same process can be written as a multi-type [`BranchingProcess`](@ref)
through its mean offspring matrix: entry `(i, j)` is the expected number of
households of type `i` infected by one household of type `j`. A household's
type sets how many households it infects, and the mixing probabilities set
which types those are, whatever the infecting household's type. Each entry is
therefore the mixing probability of type `i` times the mean of type `j`:

```@example households
M = sized.mixing * sized.means'   # (i, j) entry: mixing[i] * means[j]
# Poisson offspring with the mean the matrix gives
household_bp = BranchingProcess(M, R -> Poisson(R), Exponential(5.0))
(exact = reproduction_number(sized), multitype = reproduction_number(household_bp))
```

R\* agrees, because it depends only on the means. The extinction probability
depends on the whole offspring distribution, and here the multi-type process
assumes a Poisson number of households infected. The household model uses the
exact distribution, which is not Poisson:

```@example households
(exact = extinction_probability(sized),
    poisson_approximation = extinction_probability(household_bp))
```

Take R\* from either, and the extinction probability from
[`household_offspring`](@ref).

### When households differ in who lives in them

If transmission differs between people, two households of the same size need
not behave alike. The contact-interval distribution can then be a function of
the infector's and the susceptible person's numbers, `(infector, susceptible)
-> Distribution`. The offspring distribution is built from the model's own
households, each started from a member picked at random. Households whose
contact-interval distributions are the same for every pair of members count as
one type.

In this example every third household transmits faster than the rest, giving
two types of each size. First, record which household each person lives in:

```@example households
sizes = [fill(2, 400); fill(5, 200)]   # 400 two-person and 200 five-person households

household_of = Int[]                   # household_of[i] is person i's household
for (h, n) in enumerate(sizes)         # h counts households, n is their size
    append!(household_of, fill(h, n))
end
```

Next, mark the fast households and write the contact-interval distribution as
a function of the infector and the susceptible person. The mean contact
interval is about 3.6 days in a fast household (Weibull scale 4) and about 11
days otherwise (scale 12):

```@example households
fast_household = [h % 3 == 0 for h in eachindex(sizes)]   # every third household

function contact_interval(infector, susceptible)
    if fast_household[household_of[infector]]
        return Weibull(1.5, 4.0)
    else
        return Weibull(1.5, 12.0)
    end
end
nothing # hide
```

Finally, build the model and the offspring distribution as before:

```@example households
covariate = ModelSpec(HouseholdProcess(sizes, contact_interval);
    progression = [Transition(:recovered; from = :infection, delay = 6.0,
        terminal = true)])
typed = household_offspring(covariate; global_rate = 0.1, rng = StableRNG(8))
(sizes = typed.sizes, mixing = typed.mixing, means = typed.means)
```

Each size now appears twice, once for slow and once for fast households. The
fast households infect more others.

### Interventions and R\*

Interventions apply here too. Isolating a case for good
(`duration = Inf`) ends their infectious period early. That reduces both the housemates they infect and their community
contacts, and lowers R\*. Here every case isolates after a delay with a mean
of 1 day from symptom onset, which comes 1 day after infection. `AllCases()`
makes every case eligible for isolation; the default isolates only cases with
symptoms, which needs a [`clinical_presentation`](@ref).

```@example households
isolated = ModelSpec(HouseholdProcess(fill(4, 300), Weibull(1.5, 3.0));
    progression = [Transition(:onset; from = :infection, delay = 1.0),
        Transition(:recovered; from = :infection, delay = 6.0, terminal = true)],
    interventions = [Isolation(onset_to_isolation_delay = Exponential(1.0),
        eligibility = AllCases(), duration = Inf)])
(without_isolation = reproduction_number(offspring),
    with_isolation = reproduction_number(household_offspring(isolated;
        global_rate = 0.1, rng = StableRNG(7))))
```

Isolation brings R\* below 1. As the box at the top of the page says,
isolation here also stops transmission to housemates.

!!! warning "R\* with a finite isolation duration"
    `household_offspring` counts each case's time in the community up to
    recovery, or up to an isolation that never ends. It does not subtract the
    time spent in an isolation with a finite `duration`, so R\* comes out too
    high, even when isolation outlasts the infectious period. When isolation
    lasts at least as long as cases stay infectious, use `duration = Inf` to
    compute R\*.

[`ContactTracing`](@ref) (see [Interventions](interventions.md)) also works on
a household model, where a case's contacts are their housemates. Each
housemate is traced separately, with the same probability and delay as any
other contact. Tracing does not reach a whole household at once.

### The outbreak within one household

[`household_final_size`](@ref) gives the exact distribution of how many
members of a household are infected in the end. Its arguments are the
household size, the contact-interval distribution and the infectious period in
days:

```@example households
d = household_final_size(4, Weibull(1.5, 12.0), 6.0)
(mean_infected = mean(d), probability_all_four = pdf(d, 4))
```

With this slow contact interval, an outbreak that starts with one case in a
household of four infects on average about half of the household, and infects
all four in about one household in five.

### Check: households of one

In a household of one, the number of households infected is just that one
case's community contacts, and the model reduces to an ordinary branching
process. With a fixed 6-day infectious period, those contacts arrive at a
constant rate and their number should be Poisson:

```@example households
lone = ModelSpec(HouseholdProcess(fill(1, 1), Exponential(1.0));
    progression = [Transition(:recovered; from = :infection, delay = 6.0,
        terminal = true)])
lone_law = household_offspring_law(household_offspring(lone; global_rate = 0.1,
    rng = StableRNG(9)))
R_lone = mean(lone_law)
# largest difference from a Poisson distribution with the same mean
maximum(abs(pdf(lone_law, k) - pdf(Poisson(R_lone), k)) for k in support(lone_law))
```

The difference is tiny. It is not exactly zero because the distribution is
stored as a table of probabilities cut off at a maximum number of households.
The total number of households infected in the chain then follows the Borel
distribution, the standard chain-size distribution for Poisson offspring:

```@example households
chain_size_distribution(BranchingProcess(Poisson(R_lone)))
```

With an exponentially distributed infectious period, the number of contacts is
geometric instead: a negative binomial with dispersion k = 1. The chain size
then follows a Gamma-Borel distribution. `NegBin(R, k)` is the negative
binomial with mean R and dispersion k:

```@example households
lone_exp = ModelSpec(HouseholdProcess(fill(1, 1), Exponential(1.0));
    progression = [Transition(:recovered; from = :infection,
        delay = Exponential(6.0), terminal = true)])
R_lone_exp = reproduction_number(household_offspring(lone_exp; global_rate = 0.1,
    rng = StableRNG(9)))
chain_size_distribution(BranchingProcess(NegBin(R_lone_exp, 1.0)))
```

Larger households have no closed-form chain-size distribution, because the
number of households infected depends on the household's own final-size
distribution, which is not a standard one. [`chain_size_distribution`](@ref)
works only for Poisson and negative binomial offspring; use simulation for
chain sizes with larger households. [`reproduction_number`](@ref) and
[`extinction_probability`](@ref) work for any household size.

### Exact results and simulation noise

When the contact interval and the infectious period are both exponential and
there are no interventions, the offspring distribution is computed exactly.
Otherwise `household_offspring` simulates households of each size
(`n_samples`, 10,000 by default), and its results have Monte Carlo error. To
reduce it, increase `n_samples`; to make results reproducible, fix `rng`. With
a contact-interval distribution that depends on who lives in the household,
the whole model is simulated until at least `n_samples` households have run.

The simulation noise is only in the outbreak within each household. With the
same contact-interval distribution for every pair and a single infectious
period from the disease timeline, the mean number of households infected is
computed exactly even when the distribution is simulated. The exception is
large households with weak transmission and a random infectious period, where
the exact calculation loses accuracy and the simulated mean is used.

The model is the two-level mixing model of [Ball, Mollison and Scalia-Tomba
(1997)](https://doi.org/10.1214/aoap/1034625252), and R\* is their R\*. The
final size within a household comes from the recursion of [Ball
(1986)](https://doi.org/10.2307/1427301).

## Estimating transmission from household data

If you know who was infected in each household and when, you can estimate the
contact-interval distribution, which sets how fast infection spreads within
households.

**Data you need:**

- the household each person belongs to;
- who was infected and when;
- for each case, when their infectious period started and ended;
- which case in each household was the index case, through whom the household
  was found.

You do not need to know who infected whom.

**What you can estimate:** the parameters of the contact-interval
distribution, such as its mean, and, if you include one, the rate of community
introductions.

### The pairwise likelihood

The method is the pairwise survival likelihood of Kenah (2011). Each person is
at risk from each infectious housemate for as long as that housemate is
infectious. For each case, the likelihood counts the combined risk from all
housemates who were infectious at the time they were infected. The simulation
is the same model as the likelihood, and fitting a simulated outbreak should
recover the values it was simulated with.

[`household_infections`](@ref) extracts these data from a simulated outbreak,
and [`pairwise_surv_loglik`](@ref) evaluates the log-likelihood of a given
contact-interval distribution. Here we evaluate it at a grid of mean contact
intervals and take the best one, the maximum likelihood estimate, for an
outbreak in 500 households of four simulated with a mean contact interval of 4
days:

```@example households
truth = ModelSpec(HouseholdProcess(fill(4, 500), Exponential(4.0));
    progression = [Transition(:recovered; from = :infection, delay = 6.0, terminal = true)])
data = household_infections(simulate(truth; rng = StableRNG(3)), truth)

# log-likelihood of an exponential contact interval with mean `scale` days
ll(scale) = pairwise_surv_loglik(Exponential(scale), data)
grid = 2.0:0.5:6.0
grid[argmax([ll(s) for s in grid])]
```

The estimate matches the 4 days the outbreak was simulated with.

!!! warning "Partial protection changes what is estimated"
    Suppose the model that produced the data had partial protection in place
    throughout, such as a vaccine's efficacy, a leaky isolation, or
    differences in susceptibility or infectiousness. Each blocks a fraction
    `p` of contacts and lowers the rate of infecting contacts by the factor `1
    - p`. Fitting then estimates that lower rate instead of the rate the model
      was given. Protection that starts partway through someone's infectious
      period, such as isolation or a vaccine dose given after tracing, is not
      in the likelihood at all.

The contact-interval distribution can also depend on who infects whom, for
example adults transmitting faster than children, through a function
`(infector, susceptible) -> Distribution` of the two people's numbers, as in
[the example above](#When-households-differ-in-who-lives-in-them). The same
function works for simulating and for fitting, and fitting a simulated
outbreak recovers each of its parameters.
[Covariates and time-varying transmission](@ref) covers contact intervals
that also depend on the infector's infection date, on the calendar, or on
events recorded during the outbreak.

### Fitting with Turing

For Bayesian inference, add the log-likelihood to a model in
[Turing](https://turinglang.org/). The example below puts a prior on the log
of the contact rate (1 over the mean contact interval) and samples it with
NUTS, a Hamiltonian Monte Carlo sampler. [`compile_household_pairs`](@ref)
prepares the household structure once, outside the model, so that each
evaluation is faster; the result is the same.

```@example households
using Turing

layout = compile_household_pairs(data)   # household structure, prepared once

@model function household_fit(data, layout)
    log_rate ~ Normal(-1, 1)             # prior: log of the within-household contact rate
    mean_interval = 1 / exp(log_rate)    # mean contact interval in days
    Turing.@addlogprob! pairwise_surv_loglik(Exponential(mean_interval), data, layout)
end

chain = sample(StableRNG(4), household_fit(data, layout), NUTS(), 300; progress = false)

# posterior median and 95% credible interval of the mean contact interval
mean_interval = exp.(-vec(chain[:log_rate]))
quantile(mean_interval, [0.025, 0.5, 0.975])
```

The posterior median is close to the true 4 days. 300 draws from one chain
keep this example fast; for a real analysis, run several chains for longer and
check convergence (for example the R-hat values in `summarize(chain)`).

### Real data

!!! warning "Infection times are not observed"
    In real data you do not see infection times. They have to be estimated
    together with the parameters, from symptom onsets and test results and the
    delays in the disease timeline (data augmentation), with
    `pairwise_surv_loglik` giving the likelihood of each set of infection
    times. Prepare the layout once with `compile_household_pairs`, outside the
    model, and reuse it: it stays valid as long as the households and the set
    of people ever infected stay the same. There is no worked example of this
    in the documentation yet.

!!! warning "Data that stop before the outbreak ends (right-censoring)"
    If your data stop at a date, outbreaks in some households may still be
    going. Pass that date as `followup_end` to `household_infections` (or to
    `HouseholdInfections` when you build the data yourself). Infections and
    exposure after that date are then ignored, and cases still infectious at
    that date are treated as not yet recovered.

!!! warning "Index-case ascertainment"
    Households are usually found through their first detected case, the
    recruited index case. Without a community hazard, the likelihood takes
    each household's index case as given: they need no infector. But the first
    detected case need not be the first infected. When infection times are
    estimated, a housemate can be given an earlier infection time than the
    index case. That housemate then has nobody who could have infected
    them, and those infection times are impossible.

    Use `compile_household_pairs(data; condition_on = EarliestInfected())` to
    take as given whichever member has the earliest infection time. That
    person can change from one set of estimated infection times to the next,
    and the layout is then rebuilt at every evaluation, which is slower.
    Passing `condition_on = EarliestInfected()` to
    `loglikelihood(data, model)` does the same without building the layout
    yourself.

If the log-likelihood is `-Inf`, the data contain a case the model cannot
explain, such as someone infected when none of their housemates was infectious
and there is no community hazard.

### Fitting a community hazard

`external_hazard` is the rate, per person per day, of infection from the
community (outside the household). It can be fitted too, with one caution: a
model with some community transmission and a model with none cannot be
compared by letting `external_hazard` shrink towards zero. Fit the two
separately and compare them.

The two models treat index cases differently. With `external_hazard = 0`,
index cases are taken as given and add nothing to the likelihood. With any
positive `external_hazard`, written α, the model has to explain the index
cases as community introductions: an index case infected at time `t` adds
`log(α) - α t` to the log-likelihood. As α shrinks towards zero, introductions
become so rare that the observed ones are explained ever worse, and the
log-likelihood falls towards `-Inf`. The log-likelihood therefore jumps at α =
0, and a likelihood ratio between "some community transmission" and "none"
cannot be read off near zero.

!!! details "The log-likelihood near zero"
    Away from that one point, the log-likelihood behaves regularly. Dropping
    the terms that do not involve α, near zero it is `n_ext log(α) - α T`,
    where `n_ext` is the number of cases only the community can explain and
    `T` is the total time the population is exposed to the community. On the
    log(α) scale this is a straight line of slope `n_ext`. In 400 households
    of four with `n_ext = 401`, the slope is 401.0 at α = 1e-6 and 393.5 at α
    = 1e-3, and falls to zero at the maximum near α = 0.052.

!!! note "Gamma distributions with Turing"
    To fit a `Gamma` contact interval or community hazard with Turing, use
    `NUTS(; adtype = AutoMooncake())`. The default sampler settings fail for
    `Gamma` with a `MethodError` that comes from a dependency of this package.
    `Exponential` and `Weibull`, the distributions used above, work with the
    defaults.
