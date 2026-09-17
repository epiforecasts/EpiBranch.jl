# ── The household-level offspring law ────────────────────────────────
#
# The classical households construction. Within a household the epidemic runs
# over a small, depleting clique; between households it is a branching process
# whose unit is a whole household. One infected household's offspring are the
# households its members infect through community contact, so the household-level
# offspring law is a compound: the community contact rate applied to the total
# infectious person-time the within-household epidemic generates, which is itself
# random.
#
# Two pieces have to be right, and both are easy to get wrong by hand.
#
# The compounding. Community contacts arrive at rate `global_rate` for as long as
# any member is infectious, so given the household's total community-infectious
# person-time A the number of contacts is Poisson(global_rate · A), and the
# offspring law is that Poisson mixed over the law of A. Early in an epidemic
# every contact reaches a fresh, fully susceptible household, which is the
# branching approximation this whole construction rests on.
#
# The size-biasing. A community contact reaches a *person*, and with them their
# household, so a household of a given size is reached in proportion to its size
# times how common it is. Type is therefore household size, and the mean matrix
# `M[i, j]` — the expected number of size-`i` households infected by a size-`j`
# one — is a product of a column that depends only on the parent's size and a row
# of size-biased weights that does not depend on it at all. Being rank one is what
# collapses the multi-type threshold to a single weighted sum, R*, and makes the
# extinction probability a one-dimensional fixed point.

const _OffspringLaw = DiscreteNonParametric{Int, Float64, Vector{Int}, Vector{Float64}}

"""
The household-level offspring specification of a household-structured model:
how many *households* one infected household infects. Returned by
[`household_offspring`](@ref).

A household's type in the branching process over households is what its
within-household epidemic depends on. With one contact-interval distribution
shared by every pair that is its size. With a kernel that varies by pair, it is
the household's own members: households are the same type only when their
kernels agree pair for pair, and otherwise each is a type of its own. The fields
hold one entry per type, in increasing size order:

- `sizes` — the household size of each type.
- `households` — the ids of the model's households of each type.
- `mixing` — the probability that a community contact reaches a household of
  each type. A contact reaches a person and with them their household, so this
  is proportional to the number of people in households of that type.
- `laws` — the offspring law of a household of each type, as a
  `DiscreteNonParametric` over the number of households it infects.
- `means` — the mean of each of those laws, exact where the within-household
  final size has a closed form (see [`household_offspring`](@ref)).
- `global_rate` — the community contact rate the law was built with.

[`reproduction_number`](@ref) gives R*, [`extinction_probability`](@ref) the
probability that a chain started by one infected household of each type dies
out, and [`household_offspring_law`](@ref) the offspring law itself as a
`Distributions.jl` distribution.
"""
struct HouseholdOffspring
    sizes::Vector{Int}
    households::Vector{Vector{Int}}
    mixing::Vector{Float64}
    laws::Vector{_OffspringLaw}
    means::Vector{Float64}
    global_rate::Float64
end

function Base.show(io::IO, o::HouseholdOffspring)
    print(io, "HouseholdOffspring(sizes=", unique(o.sizes))
    length(o.sizes) > length(unique(o.sizes)) && print(io, ", types=", length(o.sizes))
    print(io, ", R*=", round(reproduction_number(o); digits = 3), ")")
end

"""
    household_offspring(model; global_rate, n_samples = 10_000, rng, tol, max_offspring)

The household-level offspring law of a household-structured `model` (a
[`HouseholdProcess`](@ref) or a `ModelSpec` wrapping one): how many households
one infected household infects, one law per household type (its size, or its
members when the kernel varies by pair).

`global_rate` is the rate at which an infectious individual makes contacts
*outside* its own household. Early in an epidemic each such contact reaches a
susceptible person in a fresh household, so the number of households one
infected household infects is Poisson with mean `global_rate` times the total
infectious person-time of its within-household epidemic. The returned
[`HouseholdOffspring`](@ref) carries that compound law per type, together with
the size-biased probabilities that a contact reaches each type.

The household sizes come from the model, so the mixing weights follow the
population it describes. The model's own layers apply: the infectious window is
the one its `progression` and `interventions` produce, so isolation shortens each
case's community-infectious period and lowers R* exactly as it lowers
transmission in a simulation. Households are seeded with a single index case, as
a newly infected household is, so the process must carry no `external_hazard`.
The index case is the member the community contact reached, uniformly at random
among the household's members.

A covariate kernel `(infector, susceptible) -> Distribution` makes the derivation
simulate the model itself, so the kernel and every other layer see the model's
own individuals and each household's epidemic follows the covariates of its
actual members. Households whose kernels agree pair for pair, in member order,
share a type, and its law pools all of them; the rest are types of their own.
With a shared kernel, size is the type and each size's households are simulated
on stand-in individuals, so the other layers must treat all members alike. A progression or
attributes that read an individual's covariates need the kernel given as a
function, even a constant one such as `(i, j) -> Exponential(3.0)`.
Each household's epidemic runs on its own clock, so interventions cannot be
wrapped in `Scheduled`. To see what switching a policy on does, derive the law
twice, once without the intervention and once with it unwrapped: the two R*
values are the reproduction numbers before and after the switch.

The within-household epidemic is resolved exactly where a closed form exists —
an exponential contact-interval kernel and an exponential infectious window, with
no interventions, make the household a Markov chain — and by simulating
households otherwise. With a shared kernel `n_samples` households of each size
are simulated. A covariate kernel has no closed form and is always simulated:
the whole model is simulated as many times as it takes to reach `n_samples`
households, and at least once. When every household is a type of its own, as
with a continuous individual covariate, each type's law comes from those few
runs and is rough, while the mixture law and R* average over all of them. Whichever route is taken, the Poisson compounding is analytical, so the simulated route carries Monte Carlo
error only in the within-household epidemic. With a shared kernel the mean is
exact whenever the infectious window is a single delay of the progression,
because the mean total
infectious person-time is then the mean final size times the mean window
(a case's own window does not bear on whether it was infected).

`tol` bounds the offspring-law tail left outside the returned support and
`max_offspring` caps it.

# Examples

```julia
using EpiBranch, EpiHouseholds, Distributions

model = ModelSpec(HouseholdProcess(fill(4, 1000), Exponential(1 / 0.5));
    progression = [Transition(:recovered; from = :infection, rate = 1 / 4,
        terminal = true)])

offspring = household_offspring(model; global_rate = 0.15)
reproduction_number(offspring)        # R*
extinction_probability(offspring)     # one per household type
household_offspring_law(offspring)    # the law a contacted household follows
```
"""
function household_offspring(spec::ModelSpec{<:HouseholdProcess};
        global_rate::Real,
        n_samples::Int = 10_000,
        rng::AbstractRNG = Random.default_rng(),
        tol::Real = 1e-10,
        max_offspring::Int = 10_000)
    global_rate > 0 ||
        throw(ArgumentError("global_rate must be positive, got $global_rate"))
    n_samples >= 1 || throw(ArgumentError("n_samples must be ≥ 1"))
    process = spec.process
    _ext_active(process.external_hazard) && throw(ArgumentError(
        "the branching process over households starts each household from a single " *
        "index case, as a newly infected household does; build the process without " *
        "an `external_hazard`"))
    # Every household in the branching process over households starts its own
    # epidemic at its own time, so a gate on the population's clock or case count
    # has no counterpart; in the pooled sample it would switch on at an
    # arbitrary point set by `n_samples`.
    any(iv -> iv isa Scheduled, spec.interventions) && throw(ArgumentError(
        "a `Scheduled` intervention gates on the population's clock or case count, " *
        "which the branching process over households does not have. For R* before " *
        "and after the policy starts, call `household_offspring` once without the " *
        "intervention and once with it unwrapped"))

    return _household_offspring(process.kernel, spec, Float64(global_rate);
        n_samples, rng, tol = Float64(tol), max_offspring)
end

# A community contact reaches a person, so a type is reached in proportion to the
# number of people in its households.
function _mixing(sizes, households)
    weights = sizes .* length.(households)
    return weights ./ sum(weights)
end

# A shared kernel: the within-household epidemic depends on the household only
# through its size, so size is the type.
function _household_offspring(kernel::ContinuousUnivariateDistribution,
        spec::ModelSpec, global_rate::Float64; n_samples::Int, rng::AbstractRNG,
        tol::Float64, max_offspring::Int)
    all_sizes = household_sizes(spec.process)
    sizes = sort(unique(all_sizes))
    households = [findall(==(n), all_sizes) for n in sizes]

    window = _window_length_law(spec)
    laws = _OffspringLaw[]
    means = Float64[]
    for n in sizes
        law, μ = _size_offspring(n, kernel, window, global_rate;
            spec, n_samples, rng, tol, max_offspring)
        push!(laws, law)
        push!(means, μ)
    end
    return HouseholdOffspring(sizes, households, _mixing(sizes, households), laws,
        means, global_rate)
end

# A pair-varying kernel: the epidemic depends on who the members are, so the
# model itself is simulated, often enough for `n_samples` households in all.
# Every layer then sees the model's own individuals, whatever it reads from them,
# and each type pools the person-time of all its households.
function _household_offspring(kernel, spec::ModelSpec, global_rate::Float64;
        n_samples::Int, rng::AbstractRNG, tol::Float64, max_offspring::Int)
    members = spec.process.members
    households = _kernel_types(kernel, members)
    sizes = [length(members[first(h)]) for h in households]

    runs = cld(n_samples, length(members))
    person_time = [Float64[] for _ in households]
    for _ in 1:runs
        pt = _simulated_person_time(spec, spec.process, rng)
        for (t, h) in enumerate(households)
            append!(person_time[t], view(pt, h))
        end
    end

    laws = [_law(_compound_poisson_pmf(pt, global_rate, tol, max_offspring))
            for pt in person_time]
    means = [global_rate * mean(pt) for pt in person_time]
    return HouseholdOffspring(sizes, households, _mixing(sizes, households), laws,
        means, global_rate)
end

# Group households whose kernels agree for every ordered pair of members, in
# member order. Agreement is `===`, which for the immutable parametric
# distributions a kernel usually returns compares their parameters; anything it
# cannot compare that way leaves households apart, which costs simulation time
# and never mixes two different households.
function _kernel_types(kernel, members)
    index = IdDict{Any, Int}()
    types = Vector{Int}[]
    for (h, mem) in enumerate(members)
        key = (length(mem), Tuple(kernel(a, b) for a in mem for b in mem if a != b))
        t = get!(index, key) do
            push!(types, Int[])
            length(types)
        end
        push!(types[t], h)
    end
    return sort!(types; by = h -> (length(members[first(h)]), first(h)))
end

function household_offspring(process::HouseholdProcess; kwargs...)
    return household_offspring(ModelSpec(process); kwargs...)
end

"""
    reproduction_number(offspring::HouseholdOffspring)

R*, the mean number of households infected by one infected household: the mean
offspring count of each household type, averaged over the size-biased
probabilities that a community contact reaches that type. The epidemic can grow
between households only if it exceeds 1.

A household's type bears on how many households it infects, and leaves their
types alone: whatever infects them, they are reached through a contact with one
of their members. The next-generation matrix over household types therefore has
rank one, and its dominant eigenvalue is this single weighted sum.
"""
EpiBranch.reproduction_number(o::HouseholdOffspring) = sum(o.mixing .* o.means)

"""
    extinction_probability(offspring::HouseholdOffspring; tol = 1e-10, max_iter = 1000)

The probability that a chain of household-to-household transmission started by
one infected household dies out, one entry per household type (see
[`HouseholdOffspring`](@ref)).

Every household infected after the first is reached by a community contact, so
its type is drawn from the size-biased mixing weights whatever its parent's type
was. The probability `s` that such a household's line dies out is therefore the
same for all of them and solves `s = Σₙ mixing[n] · Gₙ(s)`, where `Gₙ` is the
probability generating function of the offspring law of type `n`. The answer
for a household of type `n` is then `Gₙ(s)`, which differs across types only
through how many households that first one infects.
"""
function EpiBranch.extinction_probability(o::HouseholdOffspring;
        tol::Real = 1e-10, max_iter::Int = 1000)
    reproduction_number(o) <= 1 && return ones(length(o.sizes))
    s = 0.0
    for _ in 1:max_iter
        s_new = sum(o.mixing[i] * _pgf(o.laws[i], s) for i in eachindex(o.laws))
        abs(s_new - s) < tol && break
        s = s_new
    end
    return [_pgf(law, s) for law in o.laws]
end

"""
    epidemic_probability(offspring::HouseholdOffspring; kwargs...)

The probability that one infected household of each type in `offspring` starts a chain of household-to-household transmission that does not die out.
"""
function EpiBranch.epidemic_probability(o::HouseholdOffspring; kwargs...)
    return 1 .- extinction_probability(o; kwargs...)
end

"""
    household_offspring_law(offspring::HouseholdOffspring)
    household_offspring_law(offspring::HouseholdOffspring, size)

The household-level offspring law as a `Distributions.jl` distribution: the
number of households infected by one infected household of the given `size`, or,
with no size, by a household reached through a community contact — the
size-biased mixture, which is the law every household after the first one
follows. When several types share a size, as they can with a covariate kernel,
the law for that size is their mixture, weighted as a contact reaches them.

The mixture is the offspring distribution of the branching process over
households as a single-type process, so it can be handed straight to
`BranchingProcess` to simulate chains of infected households.
"""
household_offspring_law(o::HouseholdOffspring) = _mixture(o.mixing, o.laws)

function household_offspring_law(o::HouseholdOffspring, size::Integer)
    types = findall(==(size), o.sizes)
    isempty(types) && throw(ArgumentError(
        "no households of size $size in this model (sizes: $(unique(o.sizes)))"))
    length(types) == 1 && return o.laws[only(types)]
    return _mixture(o.mixing[types], o.laws[types])
end

function _mixture(weights, laws)
    n_max = maximum(length(probs(law)) for law in laws)
    p = zeros(n_max)
    for (w, law) in zip(weights, laws)
        for (k, pk) in zip(support(law), probs(law))
            p[k + 1] += w * pk
        end
    end
    return _law(p)
end

# ── Per-size offspring laws ──────────────────────────────────────────

# The offspring law of one household size under a shared kernel, and its mean. The route is chosen by
# dispatch on what the within-household epidemic is: an exponential contact
# interval racing an exponential infectious window is a Markov chain and is
# resolved exactly; anything else — a non-exponential kernel or window, an
# interventions layer, a window the progression does not make a single delay
# (signalled by a `nothing` window) — is simulated.
function _size_offspring(n::Int, kernel::Exponential, window::Exponential,
        global_rate::Float64; tol::Float64, max_offspring::Int, kwargs...)
    p = _markov_offspring_pmf(n, rate(kernel), rate(window), global_rate,
        tol, max_offspring)
    return _law(p), global_rate * _mean_person_time(n, kernel, window)
end

function _size_offspring(n::Int, kernel, window, global_rate::Float64;
        spec::ModelSpec, n_samples::Int, rng::AbstractRNG, tol::Float64,
        max_offspring::Int)
    sample = HouseholdProcess(fill(n, n_samples), kernel;
        from = spec.process.from, until = spec.process.until)
    person_time = _simulated_person_time(spec, sample, rng)
    # Compounding the Poisson analytically over the simulated person-times, rather
    # than drawing a count per household, leaves Monte Carlo error only in the
    # within-household epidemic.
    p = _compound_poisson_pmf(person_time, global_rate, tol, max_offspring)
    exact = _mean_person_time(n, kernel, window)
    μ = global_rate * (exact === nothing ? mean(person_time) : exact)
    return _law(p), μ
end

# A probability vector indexed from zero as a distribution over 0, 1, 2, ….
function _law(p::Vector{Float64})
    return DiscreteNonParametric(collect(0:(length(p) - 1)), p ./ sum(p))
end

function _pgf(law::_OffspringLaw, s::Real)
    sum(pk * s^k for (k, pk) in zip(support(law), probs(law)))
end

# ── The Markovian household, resolved exactly ────────────────────────

# Probability of exactly `k` community contacts from a household of `n` members
# seeded with one case, when each ordered pair has an exponential contact
# interval of rate `β` and each case an exponential infectious window of rate `γ`.
#
# With both exponential, the household state (susceptibles left, infectives now,
# community contacts made so far) is a Markov chain: from a state with `s`
# susceptibles and `i` infectives the next event is a within-household infection
# at rate `β·s·i`, a removal at rate `γ·i` or a community contact at rate
# `global_rate·i`, and conditioning on which it is gives a recursion that runs
# over states in increasing `s` and `i` and increasing `k`. The `i` in every rate
# cancels, so the household's own clock never enters — only the competition
# between the three events does.
function _markov_offspring_pmf(n::Int, β::Real, γ::Real, global_rate::Real,
        tol::Float64, max_offspring::Int)
    k_max = min(max(16, ceil(Int, 4 * global_rate * n / γ)), max_offspring)
    while true
        p = _markov_pmf_upto(n, β, γ, global_rate, k_max)
        (sum(p) > 1 - tol || k_max >= max_offspring) && return p
        k_max = min(2 * k_max, max_offspring)
    end
end

function _markov_pmf_upto(n::Int, β::Real, γ::Real, global_rate::Real, k_max::Int)
    # f[k + 1, s + 1, i + 1]: the probability of exactly `k` further community
    # contacts from the state with `s` susceptibles and `i` infectives left.
    f = zeros(k_max + 1, n, n + 1)
    for k in 0:k_max
        for s in 0:(n - 1)
            f[k + 1, s + 1, 1] = k == 0 ? 1.0 : 0.0   # no infectives: nothing follows
            for i in 1:(n - s)
                acc = γ * f[k + 1, s + 1, i]                       # a removal
                s > 0 && (acc += β * s * f[k + 1, s, i + 2])       # an infection
                k > 0 && (acc += global_rate * f[k, s + 1, i + 1]) # a community contact
                f[k + 1, s + 1, i + 1] = acc / (β * s + γ + global_rate)
            end
        end
    end
    return [f[k + 1, n, 2] for k in 0:k_max]   # seeded with one case: s = n-1, i = 1
end

# ── The within-household epidemic, simulated ─────────────────────────

# Total community-infectious person-time of each household of `sample`, a
# household process simulated under the model's own layers. The households are
# independent, so one run of the model's simulator gives them all.
function _simulated_person_time(spec::ModelSpec, sample::HouseholdProcess,
        rng::AbstractRNG)
    process = spec.process
    sample_spec = ModelSpec(sample;
        progression = spec.progression, interventions = spec.interventions,
        attributes = spec.attributes, observation = spec.observation)
    state = simulate(sample_spec; rng)

    from = _resolve_infectious_from(process.from, spec.progression)
    person_time = zeros(length(sample.members))
    for ind in state.individuals
        is_infected(ind) || continue
        opened = EpiBranch._window_open(ind, from)
        closed = min(EpiBranch._window_close(ind, process.until),
            EpiBranch._intervention_removal_time(ind, spec.interventions))
        # A case that never becomes infectious, or is removed before it does (a
        # recovery or isolation during a latent period), makes no contacts.
        (isfinite(opened) && closed > opened) || continue
        isfinite(closed) || throw(ArgumentError(
            "a case's infectious window has no finite length, so a household infects " *
            "unboundedly many others; give the progression a terminal transition " *
            "listed in the process's `until` states, reached by every case"))
        person_time[ind.state[:household]] += closed - opened
    end
    return person_time
end

# The offspring law of a sample of households with the given total
# community-infectious person-times: a Poisson count per household, averaged
# over the sample.
function _compound_poisson_pmf(person_time::Vector{Float64}, global_rate::Real,
        tol::Float64, max_offspring::Int)
    k_max = min(
        quantile(Poisson(global_rate * maximum(person_time)), 1 - tol) + 1,
        max_offspring)
    p = zeros(k_max + 1)
    for a in person_time
        d = Poisson(global_rate * a)
        for k in 0:k_max
            p[k + 1] += pdf(d, k)
        end
    end
    return p ./ length(person_time)
end

# ── The within-household final size ──────────────────────────────────

"""
    household_final_size(size, kernel, window; initial_infectives = 1)

The final-size distribution of the epidemic within a single household of `size`
members: how many of them are ultimately infected, counting the
`initial_infectives` it starts with.

`kernel` is the within-household contact-interval distribution — the time from
a case becoming infectious to it making infectious contact with a given
household-mate, as in [`HouseholdProcess`](@ref) — and `window` the length of a
case's infectious period, a distribution or a fixed number. Contact happens only
if it falls inside the window, and contact with someone already infected is
wasted.

The distribution comes from Ball's (1986) triangular recursion, which is exact
for any kernel and window. It runs on the probability that a specified set of
`m` susceptibles all escape one case, `E[S(L)ᵐ]` for a window length `L` and a
kernel survival function `S`: contacts from one case are independent given its
window, so this one quantity carries everything about the kernel that the final
size depends on.

# Examples

```julia
using EpiHouseholds, Distributions

# a household of four, exponential contact intervals, a fixed six-day window
d = household_final_size(4, Exponential(8.0), 6.0)
mean(d)     # mean number infected, the index case included
pdf(d, 4)   # probability the whole household is infected
```
"""
function household_final_size(size::Integer, kernel, window;
        initial_infectives::Integer = 1)
    size >= 1 || throw(ArgumentError("household size must be ≥ 1, got $size"))
    1 <= initial_infectives <= size || throw(ArgumentError(
        "initial_infectives must be between 1 and the household size, got $initial_infectives"))
    n = size - initial_infectives
    p = _final_size_pmf(n, initial_infectives, kernel, window)
    return DiscreteNonParametric(collect(initial_infectives:size), p ./ sum(p))
end

# Ball's recursion for the number of the `n` initial susceptibles ultimately
# infected, with `a` initial infectives. For each `j`, the expected number of
# ways to pick `j` susceptibles of whom a given `n - j` all escape every case
# gives one equation; the system is triangular, so each `P[j]` follows from the
# ones before it.
function _final_size_pmf(n::Int, a::Int, kernel, window)
    n == 0 && return [1.0]
    p = _final_size_recursion(Float64, kernel, window, n, a)
    _is_accurate(p) && return max.(p, 0.0)
    # The recursion subtracts terms far larger than the probabilities they
    # leave, which in Float64 fails for households of a few dozen when escape
    # probabilities are close to 1. A binomial coefficient costs at most `n`
    # bits, so extended precision with a margin of a few bits per member
    # recovers them.
    p = setprecision(BigFloat, 64 + 4n) do
        Float64.(_final_size_recursion(BigFloat, kernel, window, n, a))
    end
    _is_accurate(p) || throw(ErrorException(
        "the final-size recursion lost accuracy for a household of $(n + a), " *
        "because it subtracts terms far larger than the probabilities they leave"))
    return max.(p, 0.0)
end

# Rounding can leave a probability a hair below zero; anything worse is a real
# loss of accuracy.
_is_accurate(p) = all(>=(-1e-8), p)

# The recursion in number type `T`. Each equation is multiplied through by
# `ψ^(j + a)`, because dividing by a power of a small escape probability
# underflows to zero when transmission is strong. The coefficients are floating
# point: integer ones overflow from a household of 68.
function _final_size_recursion(T::Type{<:AbstractFloat}, kernel, window, n::Int,
        a::Int)
    p = zeros(T, n + 1)
    for j in 0:n
        ψj = _escape(T, kernel, window, n - j)
        p[j + 1] = binomial(T(n), j) * ψj^(j + a) -
                   sum(
            binomial(T(n - k), j - k) * p[k + 1] * ψj^(j - k)
            for k in 0:(j - 1); init = zero(T))
    end
    return p
end

# The probability that a specified set of `m` susceptibles all escape one case:
# its contacts with them are independent given its infectious window, so each
# escapes with the kernel's survival at the window length and the set escapes
# with the `m`-th power, averaged over the window. The power is taken in `T`: the
# recursion amplifies any inconsistency between the escape probabilities of
# different set sizes, so rounding each power separately in Float64 would undo
# the extended precision.
_escape(T::Type, kernel, window::Real, m::Int) = T(ccdf(kernel, window))^m
function _escape(T::Type, kernel, window::UnivariateDistribution, m::Int)
    m == 0 && return one(T)
    # Integrating over the window's quantiles keeps the range bounded whatever
    # the window distribution is.
    return T(first(quadgk(u -> ccdf(kernel, quantile(window, u))^m, 0.0, 1.0)))
end
# Both exponential, the escape probability is the window's Laplace transform at
# `m` times the kernel's rate.
function _escape(T::Type, kernel::Exponential, window::Exponential, m::Int)
    return T(rate(window)) / (T(rate(window)) + m * T(rate(kernel)))
end

# Mean total community-infectious person-time of a household of `n` members
# seeded with one case. A case's own infectious window does not bear on whether
# it was infected — only the windows of the others do — so the mean total is the
# mean final size times the mean window. `nothing` when the model gives no window
# law to average over; the mean is then taken from the simulated households.
function _mean_person_time(n::Int, kernel::UnivariateDistribution,
        window::Union{Real, UnivariateDistribution})
    return mean(household_final_size(n, kernel, window)) * mean(window)
end
_mean_person_time(::Int, kernel, window) = nothing

# ── The infectious window as a single delay ──────────────────────────

# The law of a case's infectious-window length, when the composed model makes it
# one delay of the progression, and `nothing` when it does not. Interventions
# cut the window at a time that depends on the rest of the case's timeline, and
# a window closed by several transitions is a minimum of delays, so neither has
# a window law to read off; both are simulated instead.
function _window_length_law(spec::ModelSpec{<:HouseholdProcess})
    isempty(spec.interventions) || return nothing
    from = _resolve_infectious_from(spec.process.from, spec.progression)
    closers = filter(t -> _enters(t, spec.process.until), spec.progression)
    length(closers) == 1 || return nothing
    closer = closers[1]
    closer isa Transition || return nothing
    closer.from === from || return nothing
    # The final-size recursion and the mean both give every infected case the
    # window, which holds only if every case reaches the state that opens it.
    _reached_by_every_case(from, spec.progression) || return nothing
    # A probability gate leaves the window open for the cases that fail it.
    (closer.probability isa Real && closer.probability == 1) || return nothing
    (closer.delay isa UnivariateDistribution || closer.delay isa Real) || return nothing
    return closer.delay
end

# Whether every case reaches `state`: it is the infection itself, or the target
# of exactly one ungated transition out of a state every case reaches.
function _reached_by_every_case(state::Symbol, progression, depth::Int = 0)
    state === :infection && return true
    depth > length(progression) && return false
    into = filter(t -> _enters(t, (state,)), progression)
    length(into) == 1 || return false
    t = into[1]
    t isa Transition || return false
    (t.probability isa Real && t.probability == 1) || return false
    t.from isa Symbol || return false
    return _reached_by_every_case(t.from, progression, depth + 1)
end

# Whether `transition` moves a case into one of `states`.
function _enters(transition, states::Tuple)
    return hasproperty(transition, :state) && getfield(transition, :state) in states
end
