# ── Simulating a branching process by generation, as counts ──────────
#
# A `BranchingProcess` with a single window, no generation-time kernel
# (discrete generations) and a finite `population_size` has the structure of
# a discrete-generation (TSIR) model: every case of one generation infects
# the next at once, with no ordering between them. `simulate_by_generation`
# exploits that to track only each generation's case count, never creating an
# `Individual` for a contact. Each generation draws every current case's
# offspring independently, sums them into that generation's potential
# contacts, then thins them one at a time against the population's remaining
# susceptibles, exactly as `_offspring_builtin_risk_blocks` decides a
# contact's fate on the individual-based race: kept with probability
# `(population_size - cumulative_cases) / population_size`, a kept contact
# raising `cumulative_cases` before the next is decided. The two therefore
# agree in distribution, generation by generation.
#
# `multiplier`, when given, is drawn once per generation and scales the mean
# of every case's offspring law before it draws: with `Poisson` offspring,
# summing over the generation's cases gives a `Poisson` total scaled by the
# same draw, so marginalising over a `Gamma` multiplier gives a negative
# binomial total whose shape is the multiplier's own, rather than one that
# grows with the number of cases, as summing independent per-case negative
# binomials would.

"""
    GenerationCounts

The state of a [`BranchingProcess`](@ref) simulated by generation
(`simulate_by_generation`), holding case counts rather than individuals:

- `counts`: the current generation's case count, an `Int` for a single-type
  model or a `Vector{Int}` with one entry per type for a multi-type one.
- `cumulative_cases`: the total number of cases so far.
- `population_size`: the model's own (`NoPopulation()` if unlimited).
- `generation`: the last generation simulated.

Continue it with `simulate(model, state; max_generations)`.
"""
struct GenerationCounts{C <: Union{Int, Vector{Int}}, P <: Union{Int, NoPopulation}, R <: AbstractRNG}
    counts::C
    cumulative_cases::Int
    population_size::P
    generation::Int
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

# `n_initial` as the starting counts: an `Int` for a single-type model, a
# vector of one count per type for a multi-type one.
function _initial_generation_counts(model::BranchingProcess, n_initial)
    nt = n_types(model)
    if nt == 1
        n_initial isa Integer || throw(
            ArgumentError(
                "simulate_by_generation needs n_initial as a single Int for a single-type model"
            )
        )
        n_initial >= 0 || throw(ArgumentError("n_initial must be non-negative"))
        return Int(n_initial)
    end
    n_initial isa AbstractVector{<:Integer} || throw(
        ArgumentError(
            "simulate_by_generation needs n_initial as a vector of $nt counts (one per " *
                "type) for a multi-type model"
        )
    )
    length(n_initial) == nt ||
        throw(ArgumentError("n_initial has $(length(n_initial)) entries for $nt types"))
    all(>=(0), n_initial) || throw(ArgumentError("n_initial must be non-negative"))
    return collect(Int, n_initial)
end

"""
    simulate_by_generation(model::BranchingProcess; n_initial = 1,
                           max_generations = 100, rng = Random.default_rng(),
                           multiplier = nothing)

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

Returns a [`GenerationCounts`](@ref); continue it with
`simulate(model, state; max_generations)`.

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
```
"""
function simulate_by_generation(
        model::BranchingProcess;
        n_initial::Union{Integer, AbstractVector{<:Integer}} = 1,
        max_generations::Integer = _DEFAULT_MAX_GENERATIONS,
        rng::AbstractRNG = Random.default_rng(),
        multiplier = nothing
    )
    counts = _initial_generation_counts(model, n_initial)
    pop = population_size(model)
    cumulative = counts isa Int ? counts : sum(counts)
    pop isa NoPopulation || cumulative <= pop || throw(
        ArgumentError("n_initial ($cumulative) must not exceed population_size ($pop)")
    )
    state = GenerationCounts(counts, cumulative, pop, 0, rng)
    return simulate(model, state; max_generations, multiplier)
end

"""
    simulate(model::BranchingProcess, state::GenerationCounts; max_generations,
             rng::AbstractRNG = state.rng, multiplier = nothing)

Continue `state`, a [`GenerationCounts`](@ref) returned by
[`simulate_by_generation`](@ref) or by this method itself, up to a later
`max_generations`. `multiplier` is as in `simulate_by_generation`, and is
read fresh on every call, so a later continuation can turn it on, off, or
change it. `state` is returned unchanged if its current generation has
already gone extinct (every count 0): there is nothing left for a later
`max_generations` to continue.

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
    _extinct(state.counts) && return state
    max_generations > state.generation || throw(
        ArgumentError(
            "max_generations ($max_generations) must be past the state's own " *
                "($(state.generation)) to continue it"
        )
    )
    offspring = _generation_offspring(model)
    counts, cumulative = state.counts, state.cumulative_cases
    pop = state.population_size
    gen = state.generation
    while gen < max_generations && !_extinct(counts)
        m = multiplier === nothing ? nothing : rand(rng, multiplier)
        potential = _generation_potential(rng, offspring, counts, m)
        counts, cumulative = _thin_counts(rng, potential, pop, cumulative)
        gen += 1
    end
    return GenerationCounts(counts, cumulative, pop, gen, rng)
end

# The sum of `current` cases' offspring (single-type), each drawn
# independently from `offspring`, scaled by the generation's shared
# multiplier `m` first when given.
function _generation_potential(rng::AbstractRNG, offspring::Distribution, current::Int, m)
    current == 0 && return 0
    law = m === nothing ? offspring : _scale_mean(offspring, m)
    return sum(rand(rng, law) for _ in 1:current)
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
        total = sum(rand(rng, law) for _ in 1:current[j])
        total == 0 && continue
        potential .+= rand(rng, Multinomial(total, offspring.alloc_probs[:, j]))
    end
    return potential
end

# `BranchingProcess` always pairs a `Distribution` offspring with a
# single-type (`Int`) count and a `MultiTypeOffspring` with per-type
# (`Vector{Int}`) counts, so these two never fire; they exist only so the
# dispatch on `offspring::Union{Distribution, MultiTypeOffspring}` and
# `current::Union{Int, Vector{Int}}` in `simulate` is exhaustive, with a clear
# error rather than a `MethodError` if the two were ever mismatched.
function _generation_potential(::AbstractRNG, ::Distribution, ::Vector{Int}, m)
    throw(ArgumentError("a single-type offspring law needs a single-type (Int) case count"))
end
function _generation_potential(::AbstractRNG, ::MultiTypeOffspring, ::Int, m)
    throw(
        ArgumentError("a multi-type offspring law needs per-type (Vector{Int}) case counts")
    )
end

# Thin `n` potential contacts one at a time against the population's
# remaining susceptibles, exactly as `_offspring_builtin_risk_blocks` decides
# each contact on the individual-based race: kept independently with
# probability `(pop_size - cumulative) / pop_size`, a kept contact raising
# `cumulative` before the next is decided. Returns the number kept and the
# resulting `cumulative`.
function _thin(rng::AbstractRNG, n::Int, pop_size::Int, cumulative::Int)
    kept = 0
    for _ in 1:n
        cumulative >= pop_size && break
        if rand(rng) <= (pop_size - cumulative) / pop_size
            kept += 1
            cumulative += 1
        end
    end
    return kept, cumulative
end
_thin(rng::AbstractRNG, n::Int, ::NoPopulation, cumulative::Int) = (n, cumulative + n)

_thin_counts(rng, potential::Int, pop, cumulative) = _thin(rng, potential, pop, cumulative)
function _thin_counts(rng, potential::Vector{Int}, pop, cumulative)
    kept = similar(potential)
    for i in eachindex(potential)
        kept[i], cumulative = _thin(rng, potential[i], pop, cumulative)
    end
    return kept, cumulative
end

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
