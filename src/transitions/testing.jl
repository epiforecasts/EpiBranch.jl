"""
    within_host(trajectory; key = :trajectory)

An attribute builder that gives each case its within-host trajectory: the
pathogen load over the days since that case was infected. `trajectory` is a
function of the random number generator and the individual,
`(rng, ind) -> load`, returning a function `load(t)` of the days `t` since
infection. It is called once per case, so the load can differ between cases
and depend on anything already recorded on them, such as age.

The package includes no pathogen. The caller supplies the trajectory, whether
an acute rise and clearance or a chronic plateau, and everything that depends
on load reads it from there. [`Testing`](@ref) samples
it at the time a test is taken, and a [`HazardKernel`](@ref) can turn it into
infectiousness through [`trajectory_load`](@ref EpiBranch.trajectory_load).
The curve is stored in `ind.state[:trajectory]` (or under `key`).

The unit of load is the caller's own (copies per millilitre, log10 copies, a
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
    trajectory_load(ind, t; key = :trajectory) -> Float64

The pathogen load of `ind` at `t` days after its infection, read from the
trajectory [`within_host`](@ref) stored on it. Use it inside a kernel,
a link or an assay function that follows the case's own load.
"""
function trajectory_load(ind::Individual, t::Real; key::Symbol = :trajectory)
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
Parent type of an assay: how a sample's pathogen load becomes a test result.
[`Assay`](@ref) is the general implementation. Define `measure` for a new type
to read load differently.
"""
abstract type AbstractAssay end

"""
    measure(assay::AbstractAssay, rng, load) -> (ct, positive)

The Ct value and the result of testing a sample of pathogen `load`:
`ct` a number (`NaN` when the sample gives none) and `positive` a `Bool`.
Draws from `rng` only if the assay is stochastic.
"""
function measure end

"""
    Assay(; ct, detect)

An assay that maps the load in a sample to a Ct value and to a result.

- `ct`: a function `load -> Ct value`, on the scale of the assay (for example a
  log-linear calibration from load to Ct). Return `NaN` for a load too low to
  give a Ct value.
- `detect`: when the sample is positive. A number is a limit of detection: the
  result is positive when the load reaches it. A function `(rng, load) -> Bool`
  gives a result that can be probabilistic, for example a detection probability
  that rises with load.

Neither has a default, as both depend on the assay and the pathogen tested for.

# Examples

```julia
# load in log10 copies per mL, Ct falling 3.3 per log10, detected from 2 log10
Assay(ct = load -> 40.0 - 3.3 * load, detect = 2.0)
```
"""
struct Assay{C, D} <: AbstractAssay
    ct::C
    detect::D
end
Assay(; ct, detect) = Assay(ct, detect)

_detected(limit::Real, rng, load) = load >= limit
_detected(rule, rng, load) = rule(rng, load)::Bool

function measure(a::Assay, rng, load)
    return (float(a.ct(load)), _detected(a.detect, rng, load))
end

"""
    Testing(; assay, delay, from = :onset_time, probability = 1.0, key = :trajectory)

A test taken from a case: a sample is drawn `delay` days after `from`, and the
case's pathogen load at that time, read from its within-host trajectory
([`within_host`](@ref)), goes through `assay` to give a Ct value and a result.
A case with a test records:

- `:tested` (`true`/`false`) and `:test_time` (`Inf` if not tested);
- `:test_ct`, the Ct value (`NaN` if not tested);
- `:test_result`, whether the test was positive.

These appear in [`linelist`](@ref) as `tested`, `date_test`, `test_ct` and
`test_result`.

- `assay`: an [`Assay`](@ref) or another [`AbstractAssay`](@ref).
- `delay`, `from`, `probability`: as for [`Reporting`](@ref). The sampling rule
  is these three: a delay from onset, from another recorded time such as
  tracing or admission, or from a function of the case that gives a screening
  schedule, and the chance a case is sampled at all. Cases that never reached
  `from` are not tested. A test timed for after the case has died or recovered
  is reset to "not tested".
- `key`: where the trajectory is stored on the case.

A case is not tested for an infection it was never infected with: the sample
time is measured from the infection that is current when the step resolves.

Use the step as `test_sensitivity` of [`Isolation`](@ref) to isolate the cases
with a positive test.

# Examples

```julia
assay = Assay(ct = load -> 40.0 - 3.3 * load, detect = 2.0)
spec = ModelSpec(
    BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5));
    attributes = [clinical_presentation(incubation_period = LogNormal(1.6, 0.5)),
                  within_host(acute)],
    progression = [Testing(assay = assay, delay = Gamma(2.0, 1.0))],
)
```
"""
struct Testing{A <: AbstractAssay, D, P, F} <: AbstractClinicalTransition
    assay::A
    delay::D
    probability::P
    from::F
    key::Symbol
end

function Testing(;
        assay::AbstractAssay, delay, from = :onset_time,
        probability = 1.0, key::Symbol = :trajectory
    )
    return Testing(assay, delay, probability, from, key)
end

# Only writes the test's own keys; never touches susceptibility or
# infectiousness.
offspring_law_unchanged(::Testing) = true

required_fields(t::Testing) = _from_required(t.from)

function initialise_individual!(::Testing, individual, state)
    individual.state[:tested] = false
    individual.state[:test_time] = Inf
    individual.state[:test_ct] = NaN
    individual.state[:test_result] = false
    return nothing
end

function resolve_individual!(t::Testing, individual, state)
    anchor = _state_time(individual, t.from)
    time = transition_time(
        state.rng, individual, anchor, t.delay;
        probability = t.probability, purpose = :tested
    )
    time === nothing && return nothing
    load = trajectory_load(individual, time - individual.infection_time; key = t.key)
    assay_rng = stream(state.rng, individual.id, (:tested, :assay))
    ct, positive = measure(t.assay, assay_rng, load)
    individual.state[:tested] = true
    individual.state[:test_time] = time
    individual.state[:test_ct] = ct
    individual.state[:test_result] = positive
    return nothing
end

function censor_after_outcome!(t::Testing, individual)
    _censor_event!(individual, :tested, :test_time, t.from)
    if !individual.state[:tested]::Bool
        individual.state[:test_ct] = NaN
        individual.state[:test_result] = false
    end
    return nothing
end

# `Testing` as the `test_sensitivity` of an `Isolation` (see `isolation.jl`): the
# case's own test has not been taken when the case is created, so the result is
# read from it once the progression has resolved.
_draw_test_positive(::Testing, rng, individual) = false

function _read_test_positive!(::Testing, iso, individual, state)
    st = individual.state
    st[:test_positive] = is_eligible_for_isolation(iso.eligibility, individual, state) &&
        get(st, :tested, false)::Bool &&
        get(st, :test_result, false)::Bool
    return nothing
end
