# Julia for R users

This page covers the Julia syntax the EpiBranch documentation uses, each with
its nearest R equivalent. It is not a Julia course; the
[Julia manual](https://docs.julialang.org/en/v1/manual/getting-started/) and
its [noteworthy differences from R](https://docs.julialang.org/en/v1/manual/noteworthy-differences/#Noteworthy-differences-from-R)
cover more.

## Loading packages: `using`

```@example juliar
using EpiBranch
using Distributions
using StableRNGs
```

`using EpiBranch` is `library(EpiBranch)`. Distributions provides the
probability distributions (`Poisson`, `Gamma`, `LogNormal`, ...), and StableRNGs
a random number generator that gives the same numbers on every computer and
Julia version.

## Random seeds

```@example juliar
rng = StableRNG(3)
```

R has one global seed, set with `set.seed(3)`. In Julia you create a random
number generator and pass it to each function that draws random numbers, here
as `rng = rng`. Two analyses with their own generators do not affect each
other.

## Keyword arguments after `;`

```@example juliar
model = BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
outbreak = simulate(model; max_cases = 500, rng = rng)
```

Arguments after the `;` are named (keyword) arguments, like
`simulate(model, max_cases = 500)` in R. In a call you may also separate them
with a comma; the documentation uses `;` to show where they start. Unlike R,
Julia does not match partial argument names.

## Fields and indexing: `.` and `[1]`

```@example juliar
first_case = outbreak.individuals[1]
first_case.infection_time
```

`outbreak.individuals` is like `outbreak$individuals` in R. Indexing starts at
1, as in R.

## Strings with values: `"$(x)"`

```@example juliar
println("Cases: $(outbreak.cumulative_cases)")
```

`$(...)` inside a string inserts a value, like `paste0("Cases: ", x)` or
`glue("Cases: {x}")` in R. `println` prints a line, like `cat(..., "\n")`.

## Symbols: `:name`

```@example juliar
ind_age = Dict(:age => 34)
ind_age[:age]
```

A symbol such as `:age` is a name, used for things like column names and
options, much as you would use the string `"age"` in R. A `Dict` is a lookup
table from names to values, like a named list. EpiBranch stores a
person's characteristics under symbols, for example `ind.state[:age]`.

## Ranges: `a:b`

```@example juliar
sizes = 50:100
(first(sizes), last(sizes), length(sizes))
```

`50:100` is every whole number from 50 to 100, as in R. In Julia it is stored
as just its two ends, and `collect(50:100)` gives the full vector. A step goes
in the middle: `0:7:28` is R's `seq(0, 28, by = 7)`.

## Anonymous functions: `x -> ...`

```@example juliar
prob_asymptomatic = (rng, ind) -> 0.3
```

`(rng, ind) -> 0.3` is a function without a name, like `function(rng, ind) 0.3`
or `\(rng, ind) 0.3` in R. EpiBranch accepts such a function wherever a value
can differ between people. It is given the random number generator and the
person (`ind`), and returns the value for that person.

## The conditional operator: `c ? a : b`

```@example juliar
by_age = (rng, ind) -> ind.state[:age] < 18 ? 0.6 : 0.2
```

`c ? a : b` is `if (c) a else b` in R: here 0.6 for children and 0.2 for
adults.

## Passing a function: `count(f, x)` and `do` blocks

```@example juliar
count(is_infected, outbreak.individuals)
```

Many Julia functions take a function as their first argument. This counts the
people for whom `is_infected` is true, like
`sum(sapply(individuals, is_infected))` in R.

A `do` block writes that first argument as a longer function below the call:

```@example juliar
count(outbreak.individuals) do ind
    is_infected(ind) && ind.generation >= 1
end
```

This counts secondary cases (infected people who are not the index case). It
is the same as `count(ind -> is_infected(ind) && ind.generation >= 1,
outbreak.individuals)`.

## Applying to every element: `f.(x)`

```@example juliar
R_values = [0.8, 1.5, 2.5]
extinction_probability.(R_values, 0.16)
```

A dot after a function name applies it to each element: this gives one
extinction probability per value of R. R does this automatically for
vectorised functions; in Julia you add the dot. Operators take the dot first:
`R_values .* 2`.

## Comprehensions: `[... for ...]`

```@example juliar
[extinction_probability(2.5, k) for k in (0.1, 0.5, 1.0)]
```

The result is a vector made by evaluating the expression for each `k`, like
`sapply(c(0.1, 0.5, 1.0), function(k) extinction_probability(2.5, k))` in R.

## Types: `::`

```@example juliar
typeof(NegBin(2.5, 0.16))
```

Every Julia value has a type, and `x::Type` says that `x` must have that type.
`NegBin(2.5, 0.16)` returns a `NegativeBinomial` from the Distributions package.
You will mostly see `::` in function definitions, as in the next section.

## Functions ending in `!`

```@example juliar
compute_trace_level!(outbreak)
```

`compute_trace_level!(outbreak)` records, for each traced person, how many
tracing steps it took to reach them from the case where tracing started, and
stores it in `outbreak` itself. By convention a function whose name ends in `!` changes one
of its arguments. R functions usually return a modified copy and leave the
original alone.

## Adding a method to an EpiBranch function

A Julia function can have several versions, called methods, and Julia picks
the one that matches the types of the arguments. This is close to S4 methods
in R. You give EpiBranch new behaviour by defining a type and a method for it:

```@example juliar
struct IndexCasesOnly <: TraceEligibility end

EpiBranch.is_eligible(::IndexCasesOnly, infector, contact, state) =
    infector.generation == 0

ct = ContactTracing(IndexCasesOnly(), 0.5, Exponential(1.5))
```

`struct IndexCasesOnly <: TraceEligibility end` creates a new kind of tracing
rule. The method of `EpiBranch.is_eligible` for it says that only contacts of
index cases are traced. The `EpiBranch.` prefix adds the method to the package's
own function; without it you would define a separate function of the same
name. In the definition, `::IndexCasesOnly` with no
argument name means the method applies whenever that argument is of this type.
[Extending EpiBranch](tutorials/extending.md) explains which functions you can
add methods to.
