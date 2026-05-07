# Design principles

These are the principles EpiBranch is meant to satisfy. They should
be revisited when adding anything substantial, and the architecture
should be reviewed against them periodically.

## 1. Simple but rigorous

Express what we need, no more. New machinery only earns its place
when an analysis we actually do can't be done without it. Closed-form
likelihoods over simulation when both are available and equivalent.
One verb (`loglikelihood`, `simulate`) does the dispatch — no
specialised wrapper functions per data type or model variant.

## 2. Self-explanatory

Type names, function names, and signatures should match an
epidemiologist's intuition for what they do. If a user has to read
source to understand what a public name means, the name is wrong. The
mathematical names (`Borel`, `GammaBorel`) are fair when they match
the literature the user comes from; the operational names should
match how an epidemiologist would describe what's happening.

## 3. Cleanly separable concerns

Process model, observation model, data, inference, simulation, and
output each own one thing. Their interfaces are explicit. New
alternatives — network models, multi-stream observation, time-varying
reporting, aggregated counts — slot in by implementing the relevant
interface, not by editing core code. Each concern is replaceable
independently.

## 4. Extensible from outside

A user can add their own transmission model, observation model,
intervention, or data type as a separate package or script, reusing
all framework infrastructure. The contracts they have to satisfy are
documented and small. Adding a custom piece does not require editing
EpiBranch.

## 5. Documented with examples

Principle 4 is empty without 5. Every public extension point has a
worked example. Tutorials are checked at build time so the prose
stays consistent with the code.
