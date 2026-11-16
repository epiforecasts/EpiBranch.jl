"""
    within_host(trajectory; key = :trajectory)

An attribute builder that gives each case its within-host trajectory: a
measured quantity over the days since that case was infected, such as pathogen
load, an antibody titre or infectiousness measured by culture. `trajectory` is a
function of the random number generator and the individual,
`(rng, ind) -> value`, returning a function `value(t)` of the days `t` since
infection. It is called once per case, so the value can differ between cases
and depend on anything already recorded on them, such as age.

The package includes no pathogen: the caller supplies the trajectory, whether
an acute rise and clearance or a chronic plateau. Define it
once and let everything that depends on it read it. [`Testing`](@ref) samples
it at the time a test is taken, and a [`HazardKernel`](@ref) can turn it into
infectiousness through [`trajectory_value`](@ref EpiBranch.trajectory_value).
The curve is stored in `ind.state[:trajectory]` (or under `key`).

The unit is the caller's own (copies per millilitre, log10 copies, a titre, a
relative scale); the assay in [`Testing`](@ref) reads it on the same scale.

# Examples

```julia
# an acute course: load rises to a peak, then clears
function acute(rng, ind)
    peak_time = rand(rng, Gamma(3.0, 1.5))
    peak = rand(rng, LogNormal(2.0, 0.3))
    clearance = rand(rng, Gamma(4.0, 0.5))
    return t -> t < peak_time ? peak * t / peak_time :
        peak * exp(-clearance * (t - peak_time))
end

attributes = within_host(acute)
```
"""
function within_host(trajectory; key::Symbol = :trajectory)
    return (rng, ind) -> (ind.state[key] = trajectory(rng, ind); nothing)
end

"""
    trajectory_value(ind, t; key = :trajectory) -> Float64

The value of the trajectory of `ind` at `t` days after its infection, read from
the trajectory [`within_host`](@ref) stored on it. Use it inside a kernel,
a link or an assay function that follows the case's own trajectory.
"""
function trajectory_value(ind::Individual, t::Real; key::Symbol = :trajectory)
    curve = get(ind.state, key, nothing)
    curve === nothing && throw(
        ArgumentError(
            "no within-host trajectory on the case under `:$key`; " *
                "add `within_host(...)` to the model's `attributes`"
        )
    )
    return float(curve(t))
end

"""
Parent type of an assay: how a sample's measured value becomes a test result.
[`Assay`](@ref) is the general implementation. Define `measure` for a new type
to read the value differently.
"""
abstract type AbstractAssay end

"""
    measure(assay::AbstractAssay, rng, value) -> (reading, positive)

The reading and the result of testing a sample at trajectory value `value`:
`reading` a number (`NaN` when the sample gives none) and `positive` a `Bool`.
Draws from `rng` only if the assay is stochastic.
"""
function measure end

"""
    Assay(; value, detect)

An assay that maps the value in a sample to a reading and to a result.

- `value`: a function `value -> reading`, on the scale of the assay (for example
  a log-linear calibration from load to Ct, or a titre dilution). Return `NaN`
  for a value too low to give a reading.
- `detect`: when the sample is positive. A number is a limit of detection: the
  result is positive when the value reaches it. A function
  `(rng, value) -> Bool` gives a result that can be probabilistic, for example a
  detection probability that rises with the value.

Neither has a default, because both belong to the assay and the pathogen being
tested for.

# Examples

```julia
# load in log10 copies per mL, Ct falling 3.3 per log10, detected from 2 log10
Assay(value = load -> 40.0 - 3.3 * load, detect = 2.0)
```
"""
struct Assay{V, D} <: AbstractAssay
    value::V
    detect::D
end
Assay(; value, detect) = Assay(value, detect)

_detected(limit::Real, rng, value) = value >= limit
_detected(rule, rng, value) = rule(rng, value)::Bool

function measure(a::Assay, rng, value)
    return (float(a.value(value)), _detected(a.detect, rng, value))
end

"""
    Testing(; assay, delay, from = :onset_time, probability = 1.0,
            name = :test, key = :trajectory)

A test taken from a case: a sample is drawn `delay` days after `from`, and the
case's trajectory at that time, read from its within-host trajectory
([`within_host`](@ref)), goes through `assay` to give a reading and a result.
A case with a test called `name` records:

- `:<name>_tested` (`true`/`false`) and `:<name>_time` (`Inf` if not tested);
- `:<name>_value`, the reading of the assay (`NaN` if not tested);
- `:<name>_result`, whether the test was positive.

With the default name these are `:test_tested`, `:test_time`, `:test_value` and
`:test_result`. They appear in [`linelist`](@ref) as `test_tested`, `date_test`,
`test_value` and `test_result` (`pcr_tested`, `date_pcr`, `pcr_value` and
`pcr_result` for `name = :pcr`).

- `assay`: an [`Assay`](@ref) or another [`AbstractAssay`](@ref).
- `delay`, `from`, `probability`: as for [`Reporting`](@ref). The sampling rule
  is these three: a delay from onset, from another recorded time such as
  tracing or admission, or from a function of the case that gives a screening
  schedule, and the chance a case is sampled at all. Cases that never reached
  `from` are not tested. A test timed for after the case has died or recovered
  is reset to "not tested".
- `name`: the name of the test. A case can have several tests, each a
  `Testing` step with its own name (a PCR test and a serology test, say). Steps
  with the same name write the same keys.
- `key`: where the trajectory is stored on the case.

A case is not tested for an infection it was never infected with: the sample
time is measured from the infection that is current when the step resolves.

Use the step as `test_sensitivity` of [`Isolation`](@ref) to isolate the cases
with a positive test.

# Examples

```julia
assay = Assay(value = load -> 40.0 - 3.3 * load, detect = 2.0)
spec = ModelSpec(
    BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5));
    attributes = [clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
                  within_host(acute)],
    progression = [Testing(assay = assay, delay = Gamma(2.0, 1.0), name = :pcr)],
)
```
"""
struct Testing{A <: AbstractAssay, D, P, F} <: AbstractClinicalTransition
    assay::A
    delay::D
    probability::P
    from::F
    name::Symbol
    key::Symbol
end

function Testing(;
        assay::AbstractAssay, delay, from = :onset_time,
        probability = 1.0, name::Symbol = :test, key::Symbol = :trajectory
    )
    return Testing(assay, delay, probability, from, name, key)
end

# The keys a test called `name` writes.
function _test_keys(t::Testing)
    return (
        tested = Symbol(t.name, :_tested), time = Symbol(t.name, :_time),
        value = Symbol(t.name, :_value), result = Symbol(t.name, :_result),
    )
end

# Only writes the test's own keys; never touches susceptibility or
# infectiousness.
offspring_law_unchanged(::Testing) = true

required_fields(t::Testing) = _from_required(t.from)

function initialise_individual!(t::Testing, individual, state)
    k = _test_keys(t)
    individual.state[k.tested] = false
    individual.state[k.time] = Inf
    individual.state[k.value] = NaN
    individual.state[k.result] = false
    return nothing
end

function resolve_individual!(t::Testing, individual, state)
    k = _test_keys(t)
    anchor = _state_time(individual, t.from)
    time = transition_time(
        state.rng, individual, anchor, t.delay;
        probability = t.probability, purpose = k.tested
    )
    time === nothing && return nothing
    value = trajectory_value(individual, time - individual.infection_time; key = t.key)
    assay_rng = stream(state.rng, individual.id, (k.tested, :assay))
    reading, positive = measure(t.assay, assay_rng, value)
    individual.state[k.tested] = true
    individual.state[k.time] = time
    individual.state[k.value] = reading
    individual.state[k.result] = positive
    return nothing
end

function censor_after_outcome!(t::Testing, individual)
    k = _test_keys(t)
    _censor_event!(individual, k.tested, k.time, t.from)
    if !individual.state[k.tested]::Bool
        individual.state[k.value] = NaN
        individual.state[k.result] = false
    end
    return nothing
end

# `Testing` as the `test_sensitivity` of an `Isolation` (see `isolation.jl`): the
# case's own test has not been taken when the case is created, so the result is
# read from it once the progression has resolved.
_draw_test_positive(::Testing, rng, individual) = false

function _read_test_positive!(t::Testing, iso, individual, state)
    k = _test_keys(t)
    st = individual.state
    st[:test_positive] = is_eligible_for_isolation(iso.eligibility, individual, state) &&
        get(st, k.tested, false)::Bool &&
        get(st, k.result, false)::Bool
    return nothing
end
