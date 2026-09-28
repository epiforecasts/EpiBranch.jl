"""
    TrialVaccination(; efficacy, severity_efficacy = 0.0, delay_to_immunity = 0.0,
        waning = nothing, mode = LeakyMode(), dose_label = :default)

Vaccinate everyone randomised to the `:vaccine` arm at enrolment (time 0); the
`:control` arm receives nothing, which stands for a placebo or delayed
vaccination beyond the end of follow-up. The keywords are those of
`VaccineEffect` and mean the same as on any other EpiBranch vaccination.

The dose is recorded when each individual is created, so it is in place before
any exposure on every transmission model, including the continuous-time ones
(`HomogeneousProcess`, households, networks). Requires an arm on every
individual, set by a randomisation such as [`IndividualRandomisation`](@ref).

# Examples

```julia
TrialVaccination(efficacy = 0.6, delay_to_immunity = 10.0)
```
"""
struct TrialVaccination{V <: VaccineEffect} <: AbstractVaccination
    effect::V
end

function TrialVaccination(; effect...)
    _check_effect_keywords(TrialVaccination, (), effect)
    return TrialVaccination(VaccineEffect(; effect...))
end

vaccine_effect(v::TrialVaccination) = getfield(v, :effect)
Base.getproperty(v::TrialVaccination, name::Symbol) = _effect_getproperty(v, name)
Base.propertynames(v::TrialVaccination, ::Bool = false) = _effect_propertynames(v)
Base.show(io::IO, v::TrialVaccination) = _show_keywords(io, v)

required_fields(::TrialVaccination) = [ARM_KEY]

function initialise_individual!(v::TrialVaccination, ind, state)
    # The shared vaccination initialiser marks the individual unvaccinated.
    invoke(initialise_individual!, Tuple{AbstractVaccination, Any, Any}, v, ind, state)
    arm(ind) === :vaccine && _record_vaccination!(v, ind, 0.0, state.rng)
    return nothing
end
