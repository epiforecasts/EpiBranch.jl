# Notes for contributors

This page is for people changing EpiBranch itself. It sets out the principles
the package is meant to satisfy and the rules for adding behaviour, against
which changes are reviewed. The ideas behind the model are in
[Design](design.md); how to write new pieces from outside the package is in
[Extending EpiBranch](@ref).

## Design principles

These are the principles EpiBranch is meant to satisfy. They should
be revisited when adding anything substantial, and the architecture
should be reviewed against them periodically.

### 1. Simple but rigorous

Express only what we need. A new mechanism earns its place only when an
analysis we actually do can't be done without it. Prefer closed-form
likelihoods over simulation when both are available and equivalent. One
verb (`loglikelihood`, `simulate`) does the dispatch, without
specialised wrapper functions per data type or model variant.

### 2. Self-explanatory

Type names, function names, and signatures should match the intuition
the epidemiology gives for what they do. If a user has to read
source to understand what a public name means, the name is wrong. The
mathematical names (`Borel`, `GammaBorel`) are fair when they match
the literature the user comes from; the operational names should
match how the epidemiology describes what's happening.

### 3. Cleanly separable concerns

Process model, observation model, data, inference, simulation, and
output each own one thing. Their interfaces are explicit. A new
alternative (a network model, multi-stream observation, time-varying
reporting, aggregated counts) slots in by implementing the relevant
interface rather than by editing core code. Each concern is replaceable
independently.

### 4. Extensible from outside

A user can add their own transmission model, observation model,
intervention, or data type as a separate package or script, reusing
all framework infrastructure. The contracts they have to satisfy are
documented and small. Adding a custom piece does not require editing
EpiBranch.

### 5. Documented with examples

Principle 4 is empty without 5. Every public extension point has a
worked example. Tutorials are checked at build time so the prose
stays consistent with the code.

## Extension by dispatch

New behaviour is added by defining a new type and a method, not by growing
options on an existing struct. A user wanting a variant should be able to
write a small struct plus one or two methods, with no edits to the package
source and no copy-pasting of existing function bodies. A `Union` field
whose members trigger different branches, a `Symbol` that switches
behaviour inside a function, or a `Bool` that selects a policy are all
signals that a seam is in the wrong place and should become a dispatched-on
type.

This holds on every axis:

- **Transmission models**: spatial, network, and immunity dynamics enter
  through new model subtypes or reusable wrapper types, not flags on the
  branching process.
- **Interventions** are orchestrators of smaller dispatched pieces. An
  intervention struct is a thin shell wiring together independently
  dispatched components (eligibility, rate, delay, effect), each a
  type with a method. The intervention body holds no hardcoded policy
  branching. Composition then works at two levels: between interventions
  (the stack the model carries) and within each intervention (its pieces).
- **Output, observation, and outcome rules** follow the same shape:
  mortality, hospitalisation, reporting, stopping conditions, and line-list
  columns are typed objects with methods, not closed sets of fields.
- **Engine loops** should ask the composed layers and never decide for them. A
  stepping loop or continuous-time race that decides what a named intervention
  does to whom, reads a state key an intervention owns (see
  [Individual state and reserved keys](@ref)), or keeps its own record of what
  has already been done to whom has taken a policy decision into core, where
  nothing a user writes can reach it. The varying part belongs behind the hook
  the layer already implements, and the loop's own bookkeeping should be about
  running the simulation.

  A loop does sometimes need a fact about an intervention: whether it can be
  honoured at all, whether its effects are exact enough for a fast path. The
  shape for that is a documented trait with a conservative default, which an
  intervention opts out of for itself, as
  [`infection_likelihood_compatible`](@ref EpiBranch.infection_likelihood_compatible)
  is asked of a composed component and
  [`watched_records`](@ref EpiBranch.watched_records) of a pair kernel, which
  says which host records its hazards depend on in place of the race guessing
  that only an intervention can move one. A method on a concrete type is then
  that type's author declaring something about it, available to anyone who
  writes a type, where a method the engine keeps on its own built-ins is
  reachable only from inside the package.

  This axis is the one the engine has not finished moving onto. Several loops
  still dispatch on a built-in intervention type or read a key a layer owns,
  which is why the rule above is written as the target rather than as a
  description of the current code.

The test of correctness for any component: can a plausible new variant be
added without editing the component's source? If not, the component is
doing too much, and the varying part should be lifted into a dispatched-on
trait. The concrete contracts (which methods each axis requires, with
worked examples) are in [Writing an intervention](tutorials/writing-interventions.md),
[New transmission structures](tutorials/new-structures.md) and the
[Extension reference](tutorials/extending-reference.md).

## Why individual state is an open dictionary

Each individual has a small typed core that the engine reads, plus an open
dictionary for everything else. The core holds its place in the transmission
tree, its infection time, and the two universal modifiers (susceptibility and
infectiousness). The dictionary is the deliberate extension point:
interventions, attribute builders, clinical transitions, and observation
models each own a small set of keys, and the engine never inspects them.

A typed core would couple the struct to every intervention's state shape and
would break the dose-label namespacing that lets multi-dose vaccination
schedules coexist. The dictionary keeps the individual independent of which
pieces a user composes. The keys the package itself reserves, and the
convention for naming keys added from other packages, are listed in
[Individual state and reserved keys](@ref).

## Why the engine mutates state in place

The engine works in place, one generation at a time. Copying the whole state
every generation would be far too expensive for an unbounded tree, so the
engine mutates on purpose.

## Analytical and simulation paths must agree

Any extension with both an analytical chain-size distribution and a simulation
path should have a regression test confirming they agree; the test suite
provides a helper for this (see [Checking simulation against the closed
form](@ref)).
