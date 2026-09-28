# Key under which a participant's arm is stored on `Individual.state`.
const ARM_KEY = :epitrial_arm

"""
    IndividualRandomisation(; vaccine_fraction = 0.5)

Attributes element that randomises each individual independently: to the
`:vaccine` arm with probability `vaccine_fraction`, to `:control` otherwise.
The arm is stored under `:epitrial_arm` and read with [`arm`](@ref). Pass it as
(or within) the `attributes` of the `ModelSpec`, ahead of anything that reads
the arm.

# Examples

```julia
spec = ModelSpec(process;
    attributes = IndividualRandomisation(),
    interventions = [TrialVaccination(efficacy = 0.6)])
```
"""
struct IndividualRandomisation
    vaccine_fraction::Float64
    function IndividualRandomisation(vaccine_fraction::Real)
        0 <= vaccine_fraction <= 1 ||
            throw(ArgumentError("vaccine_fraction must lie in [0, 1]"))
        return new(Float64(vaccine_fraction))
    end
end

function IndividualRandomisation(; vaccine_fraction = 0.5)
    IndividualRandomisation(vaccine_fraction)
end

function (r::IndividualRandomisation)(rng, ind)
    ind.state[ARM_KEY] = rand(rng) < r.vaccine_fraction ? :vaccine : :control
    return nothing
end

"""
    arm(ind) -> Symbol

The trial arm (`:vaccine` or `:control`) an individual was randomised to.
Throws an `ArgumentError` if the individual carries no arm, which means no
randomisation was included in the model's `attributes`.
"""
function arm(ind)
    a = get(ind.state, ARM_KEY, nothing)
    a === nothing && throw(ArgumentError(
        "individual $(ind.id) has no trial arm; include a randomisation such as " *
        "`IndividualRandomisation()` in the model's `attributes`"))
    return a
end
