module EpiBranchCensoredDistributionsExt

# `serial_interval` turns a model's generation time and incubation period
# into the distribution of a serial interval, for CensoredDistributions.jl's
# `double_interval_censored` to score linked-pair data against.
# CensoredDistributions.jl is a direct dependency. Loading
# ConvolvedDistributions.jl, a weak dependency that this extension calls
# directly, activates the method. That keeps the package optional for
# users who only simulate.

using EpiBranch
using ConvolvedDistributions: convolved, difference
using Distributions: Distribution

const _SIMULATED_ROUTE = " If the kernel follows a within-host trajectory shared with " *
    "onset, serial_interval does not handle that yet. For now, fit a distribution " *
    "to realised_serial_intervals(states) of simulated states."

function EpiBranch.serial_interval(model, infector::Individual, infectee::Individual)
    gt = EpiBranch._single_kernel(model)
    gt isa Distribution || throw(
        ArgumentError(
            "serial_interval needs a fixed generation-time distribution, not a " *
                "per-case function (got $(typeof(gt))); it has no general way to " *
                "resolve a convolution for every possible case. Build the " *
                "BranchingProcess with a distribution, e.g. " *
                "BranchingProcess(offspring, LogNormal(1.6, 0.5))." * _SIMULATED_ROUTE
        )
    )
    cp = EpiBranch._onset_law(EpiBranch.attributes(model))
    cp === nothing && throw(
        ArgumentError(
            "serial_interval needs the incubation period, from " *
                "attributes = clinical_presentation(...)."
        )
    )
    cp.incubation_period isa Union{Real, Distribution} || throw(
        ArgumentError(
            "serial_interval reads the incubation period directly and needs a " *
                "number or a distribution, not a function of the individual " *
                "(got $(typeof(cp.incubation_period)))."
        )
    )
    inc_infector = EpiBranch.resolve(cp.incubation_period, infector)
    inc_infectee = EpiBranch.resolve(cp.incubation_period, infectee)
    return convolved(gt, difference(inc_infectee, inc_infector))
end

end # module
