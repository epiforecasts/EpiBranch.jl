# ── Simulating a branching process by generation, as counts ──────────
#
# `simulate_by_generation` tracks only each generation's case count, never
# creating an `Individual` for a contact. This is possible because a
# single-window, discrete-generation, finite-population `BranchingProcess` has
# no ordering between the cases of a generation.
#
# `PerContactDepletion` mirrors `_offspring_builtin_risk_blocks` on the
# individual-based race. Both thin a contact against the susceptible share of
# the type it would infect. With several types they therefore give the same
# law for the counts of each generation.
#
# A shared `multiplier` is drawn once per generation. With `Poisson` offspring
# the total of a generation is `Poisson` scaled by the same draw, and
# marginalising over a `Gamma` multiplier gives a negative binomial total with
# the shape of the multiplier. Summing independent per-case negative binomials
# would instead give a shape that grows with the number of cases.

"""
    AbstractDepletion

How [`simulate_by_generation`](@ref) removes susceptibles as cases are made.
[`PerContactDepletion`](@ref) is the only rule.
"""
abstract type AbstractDepletion end

"""
    PerContactDepletion()

Decide every potential contact one at a time: a contact of a type is kept with
probability equal to the susceptible share of the type *at that moment*, and
each kept contact lowers the share for the next. The individual-based
simulation follows the same rule for one type or several, depleting each type
against its own population. This is the default of
[`simulate_by_generation`](@ref).

The rule is drawn exactly, without visiting every contact. Where the offspring
law is a mixed Poisson (`Poisson`, with or without `multiplier`, and `NegBin`),
the contacts of a generation are Poisson with a mean equal to its total
infectiousness, spread at random over the people of a type. Each person then
receives an independent Poisson number of contacts, and the number of new
infections is binomial: one draw per type, at a cost independent of the number
of cases. For any other offspring law the number of contacts between two that
reach a susceptible is geometric. Skipping over them makes the cost grow
with the number of new infections alone.
"""
struct PerContactDepletion <: AbstractDepletion end

"""
    GenerationCounts

The state of a [`BranchingProcess`](@ref) simulated by generation
(`simulate_by_generation`), holding case counts rather than individuals. A
single-type model holds an `Int` for each count below, a multi-type one a
`Vector{Int}` with one entry per type:

- `counts`: the current generation's case count.
- `cumulative_cases`: the total number of cases so far.
- `population_size`: the model's own (`NoPopulation()` if unlimited).
- `susceptibles`: the number left to infect, `NoPopulation()` if the population
  is unlimited.
- `type_sizes`: the size of the population of each type, `NoPopulation()` if
  unlimited.
- `generation`: the last generation simulated.
- `generation_length`: the length of a generation in days, or `nothing` if
  none was given; the state is at day `generation * generation_length`.
- `doses_used`: for each vaccination campaign, how many doses it has given, per
  budget (one for a shared budget, one per type for a [`PerGroup`](@ref) one).
  Empty if no campaign has been applied.

Continue it with `simulate(model, state; max_generations)`.
"""
struct GenerationCounts{
        C <: Union{Int, Vector{Int}}, P <: Union{Int, NoPopulation},
        S <: Union{Int, Vector{Int}, NoPopulation}, L <: Union{Float64, Nothing},
        D <: AbstractDepletion, R <: AbstractRNG,
    }
    counts::C
    cumulative_cases::Int
    population_size::P
    susceptibles::S
    type_sizes::S
    generation::Int
    generation_length::L
    depletion::D
    rng::R
    doses_used::Vector{Vector{Int}}
end

function Base.show(io::IO, s::GenerationCounts)
    return print(
        io,
        "GenerationCounts(counts=$(s.counts), cumulative_cases=$(s.cumulative_cases), " *
            "generation=$(s.generation))"
    )
end

_extinct(counts::Int) = counts == 0
_extinct(counts::Vector{Int}) = all(iszero, counts)

# The window `simulate_by_generation` needs: a single one, with no timing of
# its own (discrete generations), open for the whole generation (no `until`
# to close it partway through, which a count has no per-case clock to time)
# and no calendar (there is no calendar time to scale it by). Returns its
# offspring specification.
function _generation_offspring(model::BranchingProcess)
    length(model.infectiousness) == 1 || throw(
        ArgumentError(
            "simulate_by_generation needs a single infectiousness window; this model has " *
                "$(length(model.infectiousness))"
        )
    )
    w = only(model.infectiousness)
    w.kernel isa NoGenerationTime || throw(
        ArgumentError(
            "simulate_by_generation needs discrete generations: build the BranchingProcess " *
                "with no generation-time kernel (the default), not a $(typeof(w.kernel))"
        )
    )
    isempty(w.until) || throw(
        ArgumentError(
            "simulate_by_generation needs a window with no `until`: a generation's cases " *
                "have no individual clock to know when transmission would stop partway " *
                "through it"
        )
    )
    w.calendar === nothing || throw(
        ArgumentError(
            "simulate_by_generation needs a window with no `calendar`: there is no " *
                "calendar time, in a model with discrete generations, to scale it by"
        )
    )
    offspring = w.offspring
    offspring isa Union{Distribution, MultiTypeOffspring} || throw(
        ArgumentError(
            "simulate_by_generation needs a Distribution or an offspring matrix " *
                "(MultiTypeOffspring), not $(typeof(offspring)): a count has no individual " *
                "state for a function to read"
        )
    )
    return offspring
end

# A per-type count argument (`n_initial`, `immune`) as the starting counts:
# an `Int` for a single-type model, a vector of one count per type for a
# multi-type one.
function _per_type_counts(model::BranchingProcess, x, name::Symbol)
    nt = n_types(model)
    if nt == 1
        x isa Integer || throw(
            ArgumentError(
                "simulate_by_generation needs $name as a single Int for a single-type model"
            )
        )
        x >= 0 || throw(ArgumentError("$name must be non-negative"))
        return Int(x)
    end
    x isa AbstractVector{<:Integer} || throw(
        ArgumentError(
            "simulate_by_generation needs $name as a vector of $nt counts (one per " *
                "type) for a multi-type model"
        )
    )
    length(x) == nt ||
        throw(ArgumentError("$name has $(length(x)) entries for $nt types"))
    all(>=(0), x) || throw(ArgumentError("$name must be non-negative"))
    return collect(Int, x)
end

# The size of the population of each type, read from the model's population: the
# size of a single-type model, and the sizes by type of a multi-type one.
# `population_by_type` is deprecated: it splits a model's `population_size`
# between types, which a `Population(sizes)` now gives directly.
_type_sizes(model::BranchingProcess, ::Nothing, nt) = _type_sizes(population(model), nt)
_type_sizes(::NoPopulation, nt) = NoPopulation()
function _type_sizes(p::SinglePopulation, nt)
    nt == 1 && return p.size
    throw(
        ArgumentError(
            "a multi-type model with a population_size needs the size of each type's " *
                "population: give it `population = Population(sizes)`"
        )
    )
end
_type_sizes(p::GroupedPopulation, nt) = nt == 1 ? only(p.sizes) : copy(p.sizes)
function _type_sizes(model::BranchingProcess, by_type, nt)
    Base.depwarn(
        "`population_by_type` is deprecated; give the model its sizes by type, " *
            "`BranchingProcess(...; population = Population(sizes))`.",
        :simulate_by_generation; force = true
    )
    population(model) isa GroupedPopulation && throw(
        ArgumentError(
            "the model already has sizes by type in its population; drop `population_by_type`"
        )
    )
    return _split_population(model, population_size(model), by_type, nt)
end

_split_population(::BranchingProcess, ::NoPopulation, by_type, nt) = throw(
    ArgumentError("population_by_type needs a model with a finite population")
)
function _split_population(model::BranchingProcess, pop::Int, by_type, nt)
    if nt == 1
        throw(ArgumentError("population_by_type is for a multi-type model; use population_size"))
    end
    by_type isa AbstractVector{<:Integer} && length(by_type) == nt || throw(
        ArgumentError("population_by_type needs a vector of $nt sizes, one per type")
    )
    all(>=(0), by_type) || throw(ArgumentError("population_by_type must be non-negative"))
    sum(by_type) == pop || throw(
        ArgumentError(
            "population_by_type adds up to $(sum(by_type)), not to population_size ($pop)"
        )
    )
    return collect(Int, by_type)
end

"""
    simulate_by_generation(model::BranchingProcess; n_initial = 1,
                           max_generations = 100, rng = Random.default_rng(),
                           multiplier = nothing,
                           immune = 0,
                           population_by_type = nothing,
                           depletion = PerContactDepletion(),
                           generation_length = nothing)

Simulate `model` by generation, as case counts rather than individuals: see
[`GenerationCounts`](@ref). `model` needs a single infectiousness window with
no generation-time kernel (discrete generations), no `until` and no
`calendar`, and an offspring specification of a `Distribution` or an
offspring matrix (`MultiTypeOffspring`); any other model raises an
`ArgumentError` naming which of these it fails.

- `n_initial`: the number of index cases, an `Int` for a single-type model or
  a vector of one count per type for a multi-type one (1 index case by
  default, which a multi-type model must override).
- `multiplier`: scales the mean of the offspring law of the cases in a
  generation before each draws from it; `nothing` (the default) draws every
  case's offspring independently with no shared variation. A single
  `Distribution` is drawn once per generation and shared by every type. A
  vector with one `Distribution` per type draws the multiplier of each type
  independently in each generation. The types get independent overdispersion,
  as in a model with separate noise for each group, and can differ in it. The
  length of the vector must equal the number of types, otherwise an
  `ArgumentError` is raised. Only `Poisson` offspring has a known mean-scaling
  rule built in; giving `multiplier` with another offspring law raises an
  `ArgumentError`.
- `immune`: the number immune at the start, an `Int` or a vector with one
  entry per type, as for `n_initial` (none by default). The susceptibles of a
  type start at its population minus those immune and its index cases.
- `population_by_type`: deprecated. The size of each type's population is read
  from the model's [`Population`](@ref), `Population(sizes)` for a multi-type
  model.
- `depletion`: how contacts are thinned against the susceptibles left,
  [`PerContactDepletion`](@ref) (the default), the only rule. For a
  `Poisson` or `NegBin` offspring law a generation costs the same whatever
  the number of cases; for another law it costs in proportion to the number
  of new infections.
- `generation_length`: the length of a generation in days. The state records
  it and is at day `generation * generation_length`. The default, `nothing`,
  leaves it unset.

A multi-type model splits the offspring of each type between types by the
columns of the next-generation matrix, and thins the contacts of each type by
its own susceptible share, which scales the rows of the matrix by that share.

Returns a [`GenerationCounts`](@ref); continue it with
`simulate(model, state; max_generations)`.

# Vaccination campaigns

Give a [`ModelSpec`](@ref) built on such a `BranchingProcess` instead of the
process to apply its vaccination campaigns to the contacts of each generation,
at the calendar time `generation * generation_length`.
This needs a `generation_length` and a finite population. A campaign is a
[`GroupVaccination`](@ref) with `group_key = :type`, so that its groups are the
types, wrapped in [`CapacityConstrained`](@ref) for its doses, and in
[`Scheduled`](@ref) with a `start_time` and `end_time` if it has a window, in
either order. Any other intervention, or a condition other than a window,
raises an `ArgumentError` naming it, as do a `coverage` below one, a
`dose_delay`, a `delay_to_immunity`, `waning` and an efficacy that is not a
number.

The doses of generation `g` cover `[g * generation_length, (g + 1) *
generation_length)` and go to the contacts made by the cases of generation `g`,
which are generation `g + 1`. Each contact is given a dose in the order the
contacts are made, until the budget is spent, and a dosed contact is blocked
with probability `efficacy`, whatever the effect `mode`. The
`budget_per_period` is shared by the types: the doses go to the first contacts
made of all the types together. A [`PerGroup`](@ref) budget gives each type the
doses of its own budget instead. The doses are those of the periods that start
in the interval; with `carry_over`, doses not yet given stay available and are
counted in `doses_used` of the state. Several campaigns each dose the first
contacts made. A dosed person who is not infected is not protected against
later contacts. The `eligibility` of the campaign is not read.

Doses are dated by the interval, not the day: every contact of generation
`g + 1` can use any dose of the interval.

Doses left unused by earlier generations, from `carry_over` or a single
budget, still go to contacts of later generations after the window has
closed.

# Examples

```julia
using EpiBranch, Distributions, Random

model = BranchingProcess(NegBin(2.0, 0.5); population = Population(200_000))
state = simulate_by_generation(
    model; n_initial = 10, max_generations = 30, rng = Xoshiro(1)
)
state.cumulative_cases

# Poisson offspring with a shared gamma multiplier (mean 1, shape 0.5) gives
# a fixed-shape negative binomial total each generation, rather than one
# whose shape grows with the number of cases
model = BranchingProcess(Poisson(2.0))
state = simulate_by_generation(
    model; max_generations = 15, multiplier = Gamma(0.5, 2.0), rng = Xoshiro(1)
)

# A generation drawn in a fixed number of steps, however many cases it has,
# with 10% immune at the start, in weekly generations
model = BranchingProcess(Poisson(2.0); population = Population(2_500_000))
state = simulate_by_generation(
    model; n_initial = 10, max_generations = 30, immune = 250_000,
    generation_length = 7.0, multiplier = Gamma(6.5, 1 / 6.5), rng = Xoshiro(1)
)

# A vaccination campaign of 2,000 doses a day for a fortnight
model = BranchingProcess([2.0 0.5; 0.5 2.0], R -> Poisson(R), EpiBranch.NoGenerationTime();
    population = Population([40_000, 210_000]))
campaign = CapacityConstrained(GroupVaccination(efficacy = 0.9, group_key = :type);
    budget_per_period = Steps([140.0, 154.0], [0.0, 2_000.0, 0.0]), period = 1.0)
state = simulate_by_generation(
    ModelSpec(model; interventions = [campaign]); n_initial = [10, 2], max_generations = 30,
    immune = [20_000, 190_000],
    generation_length = 14.0
)
```
"""
function simulate_by_generation(
        model::BranchingProcess;
        n_initial::Union{Integer, AbstractVector{<:Integer}} = 1,
        max_generations::Integer = _DEFAULT_MAX_GENERATIONS,
        rng::AbstractRNG = Random.default_rng(),
        multiplier = nothing,
        immune::Union{Integer, AbstractVector{<:Integer}} = 0,
        population_by_type::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        depletion::AbstractDepletion = PerContactDepletion(),
        generation_length::Union{Real, Nothing} = nothing
    )
    state = _initial_generation_state(
        model, n_initial, rng, immune, population_by_type, depletion, generation_length
    )
    return simulate(model, state; max_generations, multiplier)
end

function simulate_by_generation(
        spec::ModelSpec;
        n_initial::Union{Integer, AbstractVector{<:Integer}} = 1,
        max_generations::Integer = _DEFAULT_MAX_GENERATIONS,
        rng::AbstractRNG = Random.default_rng(),
        multiplier = nothing,
        immune::Union{Integer, AbstractVector{<:Integer}} = 0,
        population_by_type::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        depletion::AbstractDepletion = PerContactDepletion(),
        generation_length::Union{Real, Nothing} = nothing
    )
    model = _generation_process(spec)
    state = _initial_generation_state(
        model, n_initial, rng, immune, population_by_type, depletion, generation_length
    )
    return simulate(spec, state; max_generations, multiplier)
end

function _initial_generation_state(
        model::BranchingProcess, n_initial, rng, immune, population_by_type, depletion,
        generation_length
    )
    nt = n_types(model)
    counts = _per_type_counts(model, n_initial, :n_initial)
    pop = population_size(model)
    sizes = _type_sizes(model, population_by_type, nt)
    cumulative = counts isa Int ? counts : sum(counts)
    immune_counts = immune isa Integer && nt > 1 ? fill(Int(immune), nt) : immune
    susceptibles = _initial_susceptibles(
        model, sizes, counts, _per_type_counts(model, immune_counts, :immune)
    )
    gl = _generation_length(generation_length)
    return GenerationCounts(
        counts, cumulative, pop, susceptibles, sizes, 0, gl, depletion, rng, Vector{Int}[]
    )
end

function _generation_process(spec::ModelSpec)
    process(spec) isa BranchingProcess || throw(
        ArgumentError(
            "simulate_by_generation needs a ModelSpec built on a BranchingProcess, not " *
                "$(nameof(typeof(process(spec))))"
        )
    )
    return process(spec)
end

_generation_length(::Nothing) = nothing
function _generation_length(length::Real)
    length > 0 || throw(ArgumentError("generation_length must be positive, got $length"))
    return Float64(length)
end

_initial_susceptibles(::BranchingProcess, ::NoPopulation, counts, immune) = NoPopulation()
function _initial_susceptibles(::BranchingProcess, sizes, counts, immune)
    s = sizes .- counts .- immune
    all(>=(0), s) || throw(
        ArgumentError(
            "n_initial ($counts) and immune ($immune) must not exceed the population " *
                "($sizes)"
        )
    )
    return s
end

"""
    simulate(model::BranchingProcess, state::GenerationCounts; max_generations,
             rng::AbstractRNG = state.rng, multiplier = nothing)

Continue `state`, a [`GenerationCounts`](@ref) returned by
[`simulate_by_generation`](@ref) or by this method itself, up to a later
`max_generations`, with the `depletion` and `generation_length` the state was
started with. `multiplier` is as in
`simulate_by_generation` and is read on every call. A later continuation can
enable, disable or change it. `state` is returned unchanged if its current
generation has gone extinct (every count 0), with no cases left
for a later `max_generations` to continue.

```julia
using EpiBranch, Distributions
model = BranchingProcess(NegBin(2.0, 0.5); population = Population(200_000))
state = simulate_by_generation(model; n_initial = 10, max_generations = 15)
state = simulate(model, state; max_generations = 30)
```
"""
function simulate(
        model::BranchingProcess, state::GenerationCounts;
        max_generations::Integer, rng::AbstractRNG = state.rng, multiplier = nothing
    )
    return _continue_generations(
        model, state, _Campaign[]; max_generations, rng, multiplier
    )
end

"""
    GenerationCampaigns

The vaccination campaigns of a [`ModelSpec`](@ref) and its branching process, as
returned by [`prepare_generations`](@ref). Hold one to continue many
[`GenerationCounts`](@ref) without building the campaigns again.
"""
struct GenerationCampaigns{M <: BranchingProcess, V <: AbstractVector}
    model::M
    campaigns::V
end

"""
    prepare_generations(spec::ModelSpec, state::GenerationCounts)

Build the vaccination campaigns of `spec` for continuing `state`. The
`generation_length` and whether the population is finite come from `state`. Pass the result to
`simulate` in place of `spec`.
"""
function prepare_generations(spec::ModelSpec, state::GenerationCounts)
    campaigns = _generation_campaigns(
        spec, state.generation_length, !(state.susceptibles isa NoPopulation)
    )
    return GenerationCampaigns(_generation_process(spec), campaigns)
end

"""
    simulate(spec::ModelSpec, state::GenerationCounts; max_generations,
             rng::AbstractRNG = state.rng, multiplier = nothing)
    simulate(prepared::GenerationCampaigns, state::GenerationCounts; max_generations,
             rng::AbstractRNG = state.rng, multiplier = nothing)

Continue `state` as for a `BranchingProcess`, applying the vaccination
campaigns of `spec` by giving their doses to the contacts of each generation
in order, as [`simulate_by_generation`](@ref) does. Pass the same `spec` the
state was started with, because a continuation with the bare model applies no
campaign. A [`prepare_generations`](@ref) result can replace `spec` and
reuses its campaigns.
"""
function simulate(
        spec::ModelSpec, state::GenerationCounts;
        max_generations::Integer, rng::AbstractRNG = state.rng, multiplier = nothing
    )
    return simulate(
        prepare_generations(spec, state), state; max_generations, rng, multiplier
    )
end

function simulate(
        prepared::GenerationCampaigns, state::GenerationCounts;
        max_generations::Integer, rng::AbstractRNG = state.rng, multiplier = nothing
    )
    return _continue_generations(
        prepared.model, state, prepared.campaigns; max_generations, rng, multiplier
    )
end

function _continue_generations(
        model::BranchingProcess, state::GenerationCounts, campaigns;
        max_generations::Integer, rng::AbstractRNG, multiplier
    )
    _extinct(state.counts) && return state
    max_generations > state.generation || throw(
        ArgumentError(
            "max_generations ($max_generations) must be past the state's own " *
                "($(state.generation)) to continue it"
        )
    )
    offspring = _generation_offspring(model)
    return _run_generations(
        model, offspring, state, campaigns, state.counts; max_generations, rng, multiplier
    )
end

# A single type keeps its count, susceptibles and size as plain integers and
# allocates no vectors for them in a generation.
function _run_generations(
        model::BranchingProcess, offspring, state::GenerationCounts, campaigns,
        counts::Int; max_generations::Integer, rng::AbstractRNG, multiplier
    )
    susceptibles = state.susceptibles isa NoPopulation ? nothing : _scalar(state.susceptibles)
    sizes = state.type_sizes isa NoPopulation ? nothing : _scalar(state.type_sizes)
    cumulative = state.cumulative_cases
    gen = state.generation
    used = _doses_used(state, campaigns)
    doses = zeros(Int, length(campaigns))
    while gen < max_generations && counts != 0
        dosed = _scalar_doses!(doses, campaigns, used, state.generation_length, gen)
        m = _single_multiplier(rng, multiplier)
        potential = _single_potential(rng, offspring, counts, m, susceptibles !== nothing)
        if susceptibles === nothing
            # Without a finite population nothing is thinned.
            counts = potential
        elseif dosed
            counts, susceptibles = _deplete_dosed(
                rng, state.depletion, potential, susceptibles, something(sizes),
                campaigns, doses, used
            )
        else
            counts, susceptibles = _deplete(
                rng, state.depletion, potential, susceptibles, something(sizes)
            )
        end
        cumulative += counts
        gen += 1
    end
    return GenerationCounts(
        counts, cumulative, state.population_size,
        susceptibles === nothing ? NoPopulation() : susceptibles,
        state.type_sizes, gen, state.generation_length, state.depletion, rng, used
    )
end

function _run_generations(
        model::BranchingProcess, offspring, state::GenerationCounts, campaigns,
        template::Vector{Int}; max_generations::Integer, rng::AbstractRNG, multiplier
    )
    counts = template
    susceptibles = state.susceptibles isa NoPopulation ? nothing :
        copy(_as_vector(state.susceptibles))
    sizes = state.type_sizes isa NoPopulation ? nothing : _as_vector(state.type_sizes)
    cumulative = state.cumulative_cases
    gen = state.generation
    used = _doses_used(state, campaigns)
    while gen < max_generations && !all(iszero, counts)
        doses, taken = _generation_doses(campaigns, used, state.generation_length, gen)
        m = if multiplier === nothing
            nothing
        elseif multiplier isa AbstractVector
            length(multiplier) == length(counts) || throw(
                ArgumentError(
                    "multiplier has $(length(multiplier)) distributions for " *
                        "$(length(counts)) types"
                )
            )
            [rand(rng, d) for d in multiplier]
        else
            rand(rng, multiplier)
        end
        potential = _generation_potential(
            rng, offspring, counts, m, susceptibles !== nothing
        )
        # Without a finite population nothing is thinned.
        counts = if susceptibles === nothing
            potential
        elseif isempty(doses)
            _deplete!(rng, state.depletion, potential, susceptibles, something(sizes))
        else
            _deplete_dosed!(
                rng, state.depletion, potential, susceptibles, sizes, doses, taken
            )
        end
        cumulative += sum(counts)
        gen += 1
    end
    return GenerationCounts(
        _like(template, counts), cumulative, state.population_size,
        susceptibles === nothing ? NoPopulation() : _like(template, susceptibles),
        state.type_sizes, gen, state.generation_length, state.depletion,
        rng, used
    )
end

# How many doses each campaign has given so far, per budget: those in the state,
# or zeros for campaigns it has not met. A continuation without campaigns
# leaves them as they were.
function _doses_used(state::GenerationCounts, campaigns)
    isempty(campaigns) && return state.doses_used
    length(state.doses_used) == length(campaigns) && return deepcopy(state.doses_used)
    return [zeros(Int, c.budget isa AbstractVector ? length(c.budget) : 1) for c in campaigns]
end

# The days the doses of generation `gen` cover. A campaign is only built when
# there is a `generation_length`, which rules out `nothing` here.
_generation_interval(length::Float64, gen) = (gen * length, (gen + 1) * length)
function _generation_interval(::Nothing, gen)
    throw(ArgumentError("a vaccination campaign needs a generation_length"))
end

_as_vector(x::Vector{Int}) = x
_as_vector(x::Int) = [x]
_scalar(x::Int) = x
_scalar(x::Vector{Int}) = only(x)
_like(::Vector{Int}, v::Vector{Int}) = v

# The multiplier of a single type: no draw, one shared draw, or the one
# distribution of a vector of length 1.
_single_multiplier(rng, ::Nothing) = nothing
function _single_multiplier(rng, multiplier::AbstractVector)
    length(multiplier) == 1 || throw(
        ArgumentError(
            "multiplier has $(length(multiplier)) distributions for 1 types"
        )
    )
    return rand(rng, only(multiplier))
end
_single_multiplier(rng, multiplier) = rand(rng, multiplier)

# ── The generation's potential contacts ──────────────────────────────

"""
    EpiBranch.sum_offspring(rng, law, n)

The total offspring of `n` cases, each drawing independently from `law`.
Where the sum of `n` copies of `law` has a closed form it is drawn in one
step, at a cost independent of `n`. Otherwise each case draws separately.
Defined for `Poisson`, `NegativeBinomial` and `Binomial`; add a method of
your own for another offspring family with a closed-form sum.
"""
sum_offspring(rng::AbstractRNG, law::Poisson, n::Int) = rand(rng, Poisson(n * mean(law)))
function sum_offspring(rng::AbstractRNG, law::NegativeBinomial, n::Int)
    return rand(rng, NegativeBinomial(n * law.r, law.p))
end
function sum_offspring(rng::AbstractRNG, law::Binomial, n::Int)
    return rand(rng, Binomial(n * ntrials(law), succprob(law)))
end
function sum_offspring(rng::AbstractRNG, law::Distribution, n::Int)
    return sum(rand(rng, law) for _ in 1:n)
end

# The Poisson mean of the contacts made by `n` cases drawing from `law`, where
# the law is a mixed Poisson: the contacts are then Poisson given this mean,
# which is what lets a finite population deplete in one binomial draw. `nothing`
# for any other law, whose contacts are drawn as a total.
_contact_rate(rng::AbstractRNG, law::Poisson, n::Int) = n * mean(law)
function _contact_rate(rng::AbstractRNG, law::NegativeBinomial, n::Int)
    # A negative binomial is a Poisson with a gamma-distributed mean, and the
    # sum of `n` of them has the gamma shape `n * r`.
    law.p >= 1 && return 0.0
    return rand(rng, Gamma(n * law.r, (1 - law.p) / law.p))
end
_contact_rate(::AbstractRNG, ::Distribution, ::Int) = nothing

# The multiplier of type `j`: one draw shared by every type, or one per type.
_type_multiplier(m::Real, ::Int) = m
_type_multiplier(m::AbstractVector, j::Int) = m[j]

# One entry per type of the contacts made by the cases of the generation,
# before any of them is thinned against the susceptibles. A single-type
# offspring law draws its one total, after the shared multiplier `m`, when
# given, has scaled the mean of the law. With a finite population and a mixed
# Poisson law the entry is the Poisson mean of the contacts (a `Float64`)
# rather than their number; `_deplete!` draws from it exactly.
function _generation_potential(
        rng::AbstractRNG, offspring::Distribution, current::Vector{Int}, m, finite::Bool
    )
    return [_single_potential(rng, offspring, only(current), m, finite)]
end

function _single_potential(rng::AbstractRNG, offspring::Distribution, n::Int, m, finite::Bool)
    n == 0 && return 0
    law = m === nothing ? offspring : scale_mean(offspring, _type_multiplier(m, 1))
    rate = finite ? _contact_rate(rng, law, n) : nothing
    rate === nothing || return Float64(rate)
    return sum_offspring(rng, law, n)
end

# Multi-type: each type's cases draw their total from its own law, scaled by
# the shared multiplier, and split multinomially by the next-generation
# matrix's columns (as `draw_offspring(::MultiTypeOffspring, ...)` does per
# case); accumulated over every type with cases this generation. A sink type
# (zero column) is skipped, as `draw_offspring` skips calling `dist_fn` for it.
# When every type's law is a mixed Poisson and the population is finite, the
# means are split by the columns instead. Poisson contacts split between types
# are independent Poisson, and no total is drawn.
function _generation_potential(
        rng::AbstractRNG, offspring::MultiTypeOffspring, current::Vector{Int}, m,
        finite::Bool
    )
    nt = _n_types(offspring)
    rates = zeros(nt)
    totals = zeros(Int, nt)
    by_rate = trues(nt)
    for j in eachindex(current)
        current[j] == 0 && continue
        R = offspring.R_by_type[j]
        R <= 0 && continue
        law = _type_law(offspring.dist_fn, R, j)
        law = m === nothing ? law : scale_mean(law, _type_multiplier(m, j))
        rate = finite ? _contact_rate(rng, law, current[j]) : nothing
        if rate === nothing
            totals[j] = sum_offspring(rng, law, current[j])
            by_rate[j] = false
        else
            rates[j] = rate
        end
    end
    if all(by_rate)
        return offspring.alloc_probs * rates
    end
    potential = zeros(Int, nt)
    for j in 1:nt
        total = by_rate[j] ? rand(rng, Poisson(rates[j])) : totals[j]
        total == 0 && continue
        potential .+= rand(rng, Multinomial(total, offspring.alloc_probs[:, j]))
    end
    return potential
end

# ── Depletion ────────────────────────────────────────────────────────

# The contacts kept after thinning `potential` against `susceptibles`, which
# is lowered by what is kept. Each potential contact is decided in turn
# against the susceptible share of its type at that moment, as
# `_offspring_builtin_risk_blocks` does on the individual-based race. The
# draws below give the same law without visiting every contact.

# Poisson contacts with mean `potential[i]`, spread at random over the `sizes[i]`
# people of the type. Each receives an independent Poisson number of them.
# Each susceptible is therefore reached with probability `1 - exp(-mean / size)`.
function _deplete!(
        rng, ::PerContactDepletion, potential::Vector{Float64}, susceptibles, sizes
    )
    kept = zeros(Int, length(potential))
    for i in eachindex(potential)
        (potential[i] <= 0 || susceptibles[i] <= 0) && continue
        reached = -expm1(-potential[i] / sizes[i])
        kept[i] = rand(rng, Binomial(susceptibles[i], reached))
        susceptibles[i] -= kept[i]
    end
    return kept
end

# The same draws for a single type, with its susceptibles as an integer:
# returns the contacts kept and the susceptibles left.
function _deplete(rng, ::PerContactDepletion, potential::Float64, susceptibles::Int, size::Int)
    (potential <= 0 || susceptibles <= 0) && return 0, susceptibles
    reached = -expm1(-potential / size)
    kept = rand(rng, Binomial(susceptibles, reached))
    return kept, susceptibles - kept
end

function _deplete(rng, ::PerContactDepletion, potential::Int, susceptibles::Int, size::Int)
    remaining = potential
    kept = 0
    while remaining > 0 && susceptibles > 0
        remaining -= rand(rng, Geometric(susceptibles / size)) + 1
        remaining < 0 && break
        kept += 1
        susceptibles -= 1
    end
    return kept, susceptibles
end

# A given number of contacts: the number that fall before the next one that
# reaches a susceptible is geometric, with the susceptible share as its
# success probability. Only the contacts that infect are visited.
function _deplete!(rng, ::PerContactDepletion, potential::Vector{Int}, susceptibles, sizes)
    kept = zeros(Int, length(potential))
    for i in eachindex(potential)
        remaining = potential[i]
        while remaining > 0 && susceptibles[i] > 0
            remaining -= rand(rng, Geometric(susceptibles[i] / sizes[i])) + 1
            remaining < 0 && break
            kept[i] += 1
            susceptibles[i] -= 1
        end
    end
    return kept
end

# ── The shared multiplier's mean-scaling rule ────────────────────────

"""
    EpiBranch.scale_mean(d, m)

`d`'s law with its mean scaled by `m`, the shared multiplier
[`simulate_by_generation`](@ref) draws once per generation. Defined for
`Poisson`, the family its negative-binomial identity needs; add a method of
your own for another offspring family to use it with `multiplier`.
"""
scale_mean(d::Poisson, m::Real) = Poisson(mean(d) * m)
function scale_mean(d::Distribution, m::Real)
    throw(
        ArgumentError(
            "simulate_by_generation's multiplier has no EpiBranch.scale_mean method for " *
                "$(typeof(d)); it is defined for Poisson. Add a method of your own for " *
                "another offspring family."
        )
    )
end
