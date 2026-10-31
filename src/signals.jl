# ── Signals ─────────────────────────────────────────────────────────
# A signal reads a population quantity off a simulated state over the
# reporting interval a step covers, the same quantity every engine that
# produces a `SimulationState` can compute. `SignalData` (in
# `analytical/data_types.jl`) pairs one with a density to score it against
# observed data; `loglikelihood(::SignalData, ...)` (in
# `analytical/signal_likelihood.jl`) scores several together against the
# same simulated path, since separate `~` statements would each simulate
# their own outbreak.

"""
Parent type of every signal: a read of a population quantity off a simulated
[`SimulationState`](@ref), over the half-open interval `(lo, hi]` a step of a
[`SignalData`](@ref) covers. A signal `sig` is called as `sig(state,
interval)`, with `interval` a `(lo, hi)` pair of times.

Built in: [`TransitionCount`](@ref) (how many individuals newly reached a
state within the interval) and [`Prevalence`](@ref) (how many individuals are
in a state at the interval's end). Both read the time an individual reached a
named state the way a [`Transition`](@ref)'s `from` does: `:infection` for the
time of infection, or any other symbol `s` for the time recorded under
`Symbol(s, :_time)`, such as `:onset` for `:onset_time`.
"""
abstract type Signal end

"""
    TransitionCount(at)

How many individuals newly reached the state named `at` within the interval a
[`SignalData`](@ref) step covers: those whose `at` time falls in `(lo, hi]`.
`at` is `:infection` for new infections, or any other state an attribute or a
progression step such as [`Transition`](@ref) records, for example `:onset`.

# Examples

```julia
new_onsets = TransitionCount(:onset)
new_onsets(state, (6.0, 7.0))  # cases with onset on day 7
```
"""
struct TransitionCount{F} <: Signal
    at::F
end

function (sig::TransitionCount)(state::SimulationState, interval::Tuple{<:Real, <:Real})
    lo, hi = interval
    return count(state.individuals) do ind
        is_infected(ind) || return false
        t = _state_time(ind, sig.at)
        return isfinite(t) && lo < t <= hi
    end
end

"""
    Prevalence(enter, leave = :outcome)

How many individuals are in a state at the end of the interval a
[`SignalData`](@ref) step covers: those who had reached `enter` by then and
had not yet reached `leave`. `leave` defaults to `:outcome`, the case's
outcome (recovery, death, or whatever ends it), so `Prevalence(:infectious)`
counts cases still infectious at that time.

# Examples

```julia
infectious = Prevalence(:infectious)
infectious(state, (6.0, 7.0))  # infectious cases at the end of day 7

# currently isolated, read from the pair isolation sets and clears
isolated = Prevalence(:isolation_time, :isolation_release_time)
```
"""
struct Prevalence{F, G} <: Signal
    enter::F
    leave::G
end
Prevalence(enter) = Prevalence(enter, :outcome)

function (sig::Prevalence)(state::SimulationState, interval::Tuple{<:Real, <:Real})
    _, hi = interval
    return count(state.individuals) do ind
        is_infected(ind) || return false
        enter_t = _state_time(ind, sig.enter)
        (isfinite(enter_t) && enter_t <= hi) || return false
        leave_t = _state_time(ind, sig.leave)
        return !(isfinite(leave_t) && leave_t <= hi)
    end
end
