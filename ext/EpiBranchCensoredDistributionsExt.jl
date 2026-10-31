module EpiBranchCensoredDistributionsExt

# `serial_interval` turns a model's generation time and incubation period
# into the distribution of a serial interval, for CensoredDistributions.jl's
# `double_interval_censored` to score linked-pair data against. Loading
# CensoredDistributions.jl (which depends on ConvolvedDistributions.jl,
# declared here as its own weak dependency so this extension can call it
# directly) is what turns this on, so both packages stay optional for users
# who only simulate.

using EpiBranch
using ConvolvedDistributions: convolved, difference
using Distributions: Distribution

function EpiBranch.serial_interval(model, infector::Individual, infectee::Individual)
    gt = EpiBranch._single_kernel(model)
    gt isa Distribution || throw(
        ArgumentError(
            "serial_interval needs a fixed generation-time distribution, not a " *
                "per-case function (got $(typeof(gt))); it has no general way to " *
                "resolve a convolution for every possible case. Build the " *
                "BranchingProcess with a distribution, e.g. " *
                "BranchingProcess(offspring, LogNormal(1.6, 0.5))."
        )
    )
    cp = EpiBranch._clinical_presentation(EpiBranch.attributes(model))
    cp === nothing && throw(
        ArgumentError(
            "serial_interval needs the incubation period, from " *
                "attributes = clinical_presentation(...)"
        )
    )
    inc_infector = EpiBranch.resolve(cp.incubation_period, infector)
    inc_infectee = EpiBranch.resolve(cp.incubation_period, infectee)
    return convolved(gt, difference(inc_infectee, inc_infector))
end

end # module
