# ── Simulating a branching process by generation, as counts ──────────
#
# A `BranchingProcess` with a single window, no generation-time kernel
# (discrete generations) and a finite `population_size` has the structure of
# a discrete-generation (TSIR) model: every case of one generation infects
# the next at once, with no ordering between them. `simulate_by_generation`
# exploits that to track only each generation's case count, never creating an
# `Individual` for a contact.
#
# A generation draws its potential contacts, then thins them against the
# susceptibles of each type left. Where the offspring law has a closed-form
# sum (`Poisson`, `NegativeBinomial`, `Binomial`), the total of a type is drawn
# in one step, so the cost does not grow with the number of cases. Any other
# law draws the offspring of every case one at a time.
#
# Two depletion rules thin the contacts. `PerContactDepletion` decides them
# one at a time, as `_offspring_builtin_risk_blocks` decides the fate of a
# contact on the individual-based race when there is a single type. A contact
# of type `i` is kept with probability `susceptibles[i] / type_sizes[i]`, and
# each kept contact lowers `susceptibles[i]` before the next is decided. With
# several types the individual-based simulation thins every contact against
# one shared population fraction, so the two differ.
# `StartOfGenerationDepletion` instead keeps a binomial share of the contacts
# of a type, with the share set to the susceptible fraction of the type at the
# start of the generation, capped at the susceptibles left, in one draw.
#
# `multiplier`, when given, is drawn once per generation and scales the mean
# of every case's offspring law before it draws: with `Poisson` offspring,
# summing over the generation's cases gives a `Poisson` total scaled by the
# same draw. Marginalising over a `Gamma` multiplier gives a negative
# binomial total with the shape of the multiplier. Summing independent
# per-case negative binomials would instead give a shape that grows with
# the number of cases.

"""
    AbstractDepletion

How [`simulate_by_generation`](@ref) removes susceptibles as cases are made:
[`PerContactDepletion`](@ref) or [`StartOfGenerationDepletion`](@ref).
"""
abstract type AbstractDepletion end

"""
    PerContactDepletion()

Decide every potential contact one at a time: a contact of a type is kept with
probability equal to the susceptible share of the type *at that moment*, and
each kept contact lowers the share for the next. For a single type this is the
rule of the individual-based simulation. For several types it differs, because
the individual-based simulation thins every contact against one shared
population fraction, while this rule uses the share of each type. It is the
default of [`simulate_by_generation`](@ref), and its run time is proportional
to the number of potential contacts.
"""
struct PerContactDepletion <: AbstractDepletion end

"""
    StartOfGenerationDepletion()

Thin the potential contacts of a type by its susceptible share at the *start*
of the generation, as the standard discrete-generation (TSIR) model does: one
binomial draw per type, capped at the susceptibles left, so that a generation
costs the same whatever the number of cases. Where the offspring law has a
closed-form sum, the whole generation is then drawn in a fixed number of
steps.

It differs from [`PerContactDepletion`](@ref) because the share does not fall
as the contacts of the generation are made. With many contacts relative to the
susceptibles left, the binomial rule keeps more of them (held back only by the
cap at the susceptibles), while the per-contact rule slows down as it uses
them up. The two agree closely while a generation uses only a small part of
the susceptibles, as in a large population early in an outbreak.
"""
struct StartOfGenerationDepletion <: AbstractDepletion end

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

# The size of the population of each type: `population_size` for a
# single-type model and `population_by_type` (which must add up to it) for a
# multi-type one, because a total can be split between types in many ways.
_type_sizes(::BranchingProcess, ::NoPopulation, ::Nothing, nt) = NoPopulation()
function _type_sizes(model::BranchingProcess, pop::Int, by_type, nt)
    if nt == 1
        by_type === nothing || throw(
            ArgumentError("population_by_type is for a multi-type model; use population_size")
        )
        return pop
    end
    by_type === nothing && throw(
        ArgumentError(
            "a multi-type model with a population_size needs `population_by_type`, the " *
                "size of each type's population, which must add up to population_size"
        )
    )
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
function _type_sizes(::BranchingProcess, ::NoPopulation, by_type, nt)
    throw(
        ArgumentError(
            "population_by_type needs a model with a finite population_size"
        )
    )
end

"""
    simulate_by_generation(model::BranchingProcess; n_initial = 1,
                           max_generations = 100, rng = Random.default_rng(),
                           multiplier = nothing, immune = 0,
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
- `multiplier`: a `Distribution` drawn once per generation and shared by every
  case in it, scaling the mean of their offspring law before each draws from
  it; `nothing` (the default) draws every case's offspring independently with
  no shared variation. Only `Poisson` offspring has a known mean-scaling rule
  built in; giving `multiplier` with another offspring law raises an
  `ArgumentError`.
- `immune`: the number immune at the start, an `Int` or a vector with one
  entry per type, as for `n_initial` (none by default). The susceptibles of a
  type start at its population minus those immune and its index cases.
- `population_by_type`: the size of the population of each type, for a
  multi-type model with a finite `population_size`, to which they must add up.
  The population of a single-type model is its `population_size`.
- `depletion`: how contacts are thinned against the susceptibles left,
  [`PerContactDepletion`](@ref) (the default) or
  [`StartOfGenerationDepletion`](@ref). The second draws the kept contacts of
  a type in one binomial draw. For an offspring law with a closed-form
  sum (`Poisson`, with or without `multiplier`, `NegBin` and `Binomial`), a
  generation then costs the same whatever the number of cases.
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
process to apply its vaccination campaigns to the susceptibles of each type
between generations, at the calendar time `generation * generation_length`.
This needs a `generation_length` and a finite population. A campaign is a
[`GroupVaccination`](@ref) with `group_key = :type`, so that its groups are the
types, wrapped in [`CapacityConstrained`](@ref) for its doses, and in
[`Scheduled`](@ref) with a `start_time` and `end_time` if it has a window, in
either order. Any other intervention, or a condition other than a window,
raises an `ArgumentError` naming it, as do a `coverage` below one, a
`dose_delay`, a `delay_to_immunity`, `waning` and an efficacy that is not a
number.

The doses of generation `g` cover `[g * generation_length, (g + 1) *
generation_length)` and are applied after generation `g` is drawn and before
`g + 1` is. Each type has the whole `budget_per_period` to itself, as in the
individual-based simulation, where `GroupVaccination` keeps one budget per
group. The doses are those of the periods that start in the interval; with
`carry_over`, budget left from earlier periods, or from before a window opens,
is given in the first interval open to it. A dose reaches a susceptible of a
type with probability equal to the share of susceptibles in the population of
the type at the start of the interval, and protects with probability
`efficacy`, whatever the effect `mode`: a protected susceptible is removed from
the susceptibles, as one who would be blocked at each contact. Doses that fall
on someone already protected are not tracked, so a campaign that gives a type
more doses than it has people is only approximate. The `eligibility` of the
campaign is not read: the doses are given whether or not a case has triggered
them.

Vaccinating between generations approximates a model whose generations are
calendar steps, as in a TSIR, where everyone vaccinated in a step is protected
before the next generation's contacts are made. It agrees with the
individual-based simulation only when the budget reaches everyone in the
window. That simulation gives the doses of a `BranchingProcess` only to
contacts as they are created, so a budget below the number of contacts covers a
share of contacts, not `doses / population`, and the counts will not agree.
The budget is also read per type here. In the mixing approximations the groups
share one budget, and in the individual-based simulation the budget is per
group when the groups are types and shared otherwise. Where the doses
belong in the intervention specification is not settled, so how a budget is
split between groups may change.

# Examples

```julia
using EpiBranch, Distributions, Random

model = BranchingProcess(NegBin(2.0, 0.5); population_size = 200_000)
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
model = BranchingProcess(Poisson(2.0); population_size = 2_500_000)
state = simulate_by_generation(
    model; n_initial = 10, max_generations = 30, immune = 250_000,
    depletion = StartOfGenerationDepletion(), generation_length = 7.0,
    multiplier = Gamma(6.5, 1 / 6.5), rng = Xoshiro(1)
)

# A vaccination campaign of 2,000 doses a day to each type for a fortnight
model = BranchingProcess([2.0 0.5; 0.5 2.0], R -> Poisson(R), EpiBranch.NoGenerationTime();
    population_size = 250_000)
campaign = CapacityConstrained(GroupVaccination(efficacy = 0.9, group_key = :type);
    budget_per_period = Steps([140.0, 154.0], [0.0, 2_000.0, 0.0]), period = 1.0)
state = simulate_by_generation(
    ModelSpec(model; interventions = [campaign]); n_initial = [10, 2], max_generations = 30,
    population_by_type = [40_000, 210_000], immune = [20_000, 190_000],
    depletion = StartOfGenerationDepletion(), generation_length = 14.0
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
    sizes = _type_sizes(model, pop, population_by_type, nt)
    cumulative = counts isa Int ? counts : sum(counts)
    immune_counts = immune isa Integer && nt > 1 ? fill(Int(immune), nt) : immune
    susceptibles = _initial_susceptibles(
        model, sizes, counts, _per_type_counts(model, immune_counts, :immune)
    )
    gl = _generation_length(generation_length)
    return GenerationCounts(
        counts, cumulative, pop, susceptibles, sizes, 0, gl, depletion, rng
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
model = BranchingProcess(NegBin(2.0, 0.5); population_size = 200_000)
state = simulate_by_generation(model; n_initial = 10, max_generations = 15)
state = simulate(model, state; max_generations = 30)
```
"""
function simulate(
        model::BranchingProcess, state::GenerationCounts;
        max_generations::Integer, rng::AbstractRNG = state.rng, multiplier = nothing
    )
    return _continue_generations(model, state, _Campaign[]; max_generations, rng, multiplier)
end

"""
    simulate(spec::ModelSpec, state::GenerationCounts; max_generations,
             rng::AbstractRNG = state.rng, multiplier = nothing)

Continue `state` as for a `BranchingProcess`, applying the vaccination
campaigns of `spec` between generations as [`simulate_by_generation`](@ref)
does. Pass the same `spec` the state was started with: a continuation with the
bare model applies no campaign.
"""
function simulate(
        spec::ModelSpec, state::GenerationCounts;
        max_generations::Integer, rng::AbstractRNG = state.rng, multiplier = nothing
    )
    campaigns = _generation_campaigns(
        spec, state.generation_length, !(state.susceptibles isa NoPopulation)
    )
    return _continue_generations(
        _generation_process(spec), state, campaigns; max_generations, rng, multiplier
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
    template = state.counts
    counts = _as_vector(state.counts)
    susceptibles = state.susceptibles isa NoPopulation ? nothing :
        copy(_as_vector(state.susceptibles))
    sizes = state.type_sizes isa NoPopulation ? nothing : _as_vector(state.type_sizes)
    cumulative = state.cumulative_cases
    gen = state.generation
    while gen < max_generations && !all(iszero, counts)
        # The doses of generation `gen`'s interval reach the susceptibles
        # before the next generation is drawn.
        for campaign in campaigns
            t0, t1 = _generation_interval(state.generation_length, gen)
            _vaccinate!(rng, campaign, t0, t1, susceptibles, sizes)
        end
        m = multiplier === nothing ? nothing : rand(rng, multiplier)
        potential = _generation_potential(rng, offspring, counts, m)
        # Without a finite population nothing is thinned.
        counts = susceptibles === nothing ? potential :
            _deplete!(rng, state.depletion, potential, susceptibles, sizes)
        cumulative += sum(counts)
        gen += 1
    end
    return GenerationCounts(
        _like(template, counts), cumulative, state.population_size,
        susceptibles === nothing ? NoPopulation() : _like(template, susceptibles),
        state.type_sizes, gen, state.generation_length, state.depletion,
        rng
    )
end

# The days the doses of generation `gen` cover. A campaign is only built when
# there is a `generation_length`, which rules out `nothing` here.
_generation_interval(length::Float64, gen) = (gen * length, (gen + 1) * length)
function _generation_interval(::Nothing, gen)
    throw(ArgumentError("a vaccination campaign needs a generation_length"))
end

_as_vector(x::Int) = [x]
_as_vector(x::Vector{Int}) = x
_like(::Int, v::Vector{Int}) = only(v)
_like(::Vector{Int}, v::Vector{Int}) = v

# ── The generation's potential contacts ──────────────────────────────

# The total offspring of `n` cases, each drawing independently from `law`.
# Where the sum of `n` copies of `law` has a closed form it is drawn in one
# step, at a cost that does not depend on `n`. Otherwise every case draws.
_sum_offspring(rng::AbstractRNG, law::Poisson, n::Int) = rand(rng, Poisson(n * mean(law)))
function _sum_offspring(rng::AbstractRNG, law::NegativeBinomial, n::Int)
    return rand(rng, NegativeBinomial(n * law.r, law.p))
end
function _sum_offspring(rng::AbstractRNG, law::Binomial, n::Int)
    return rand(rng, Binomial(n * ntrials(law), succprob(law)))
end
function _sum_offspring(rng::AbstractRNG, law::Distribution, n::Int)
    return sum(rand(rng, law) for _ in 1:n)
end

# One entry per type of the contacts made by the cases of the generation,
# before any of them is thinned against the susceptibles. A single-type
# offspring law draws its one total, after the shared multiplier `m`, when
# given, has scaled the mean of the law.
function _generation_potential(
        rng::AbstractRNG, offspring::Distribution, current::Vector{Int}, m
    )
    n = only(current)
    n == 0 && return [0]
    law = m === nothing ? offspring : _scale_mean(offspring, m)
    return [_sum_offspring(rng, law, n)]
end

# Multi-type: each type's cases draw their total from its own law, scaled by
# the shared multiplier, and split multinomially by the next-generation
# matrix's columns (as `draw_offspring(::MultiTypeOffspring, ...)` does per
# case); accumulated over every type with cases this generation. A sink type
# (zero column) is skipped, as `draw_offspring` skips calling `dist_fn` for it.
function _generation_potential(
        rng::AbstractRNG, offspring::MultiTypeOffspring, current::Vector{Int}, m
    )
    potential = zeros(Int, _n_types(offspring))
    for j in eachindex(current)
        current[j] == 0 && continue
        R = offspring.R_by_type[j]
        R <= 0 && continue
        law = _type_law(offspring.dist_fn, R, j)
        law = m === nothing ? law : _scale_mean(law, m)
        total = _sum_offspring(rng, law, current[j])
        total == 0 && continue
        potential .+= rand(rng, Multinomial(total, offspring.alloc_probs[:, j]))
    end
    return potential
end

# ── Depletion ────────────────────────────────────────────────────────

# The contacts kept after thinning `potential` against `susceptibles`, which
# is lowered by what is kept.

# One at a time, as `_offspring_builtin_risk_blocks` decides each contact on
# the individual-based race for a single type. Each type has its own share.
function _deplete!(rng, ::PerContactDepletion, potential, susceptibles, sizes)
    kept = similar(potential)
    for i in eachindex(potential)
        kept[i] = 0
        for _ in 1:potential[i]
            susceptibles[i] <= 0 && break
            if rand(rng) <= susceptibles[i] / sizes[i]
                kept[i] += 1
                susceptibles[i] -= 1
            end
        end
    end
    return kept
end

# One binomial draw per type at the susceptible share the generation started
# with, capped at the susceptibles left.
function _deplete!(rng, ::StartOfGenerationDepletion, potential, susceptibles, sizes)
    kept = similar(potential)
    for i in eachindex(potential)
        if potential[i] == 0 || susceptibles[i] <= 0
            kept[i] = 0
            continue
        end
        kept[i] = min(rand(rng, Binomial(potential[i], susceptibles[i] / sizes[i])), susceptibles[i])
        susceptibles[i] -= kept[i]
    end
    return kept
end

# ── The shared multiplier's mean-scaling rule ────────────────────────

"""
    EpiBranch._scale_mean(d, m)

`d`'s law with its mean scaled by `m`, the shared multiplier
[`simulate_by_generation`](@ref) draws once per generation. Defined for
`Poisson`, the family its negative-binomial identity needs; add a method of
your own for another offspring family to use it with `multiplier`.
"""
_scale_mean(d::Poisson, m::Real) = Poisson(mean(d) * m)
function _scale_mean(d::Distribution, m::Real)
    throw(
        ArgumentError(
            "simulate_by_generation's multiplier has no EpiBranch._scale_mean method for " *
                "$(typeof(d)); it is defined for Poisson. Add a method of your own for " *
                "another offspring family."
        )
    )
end
