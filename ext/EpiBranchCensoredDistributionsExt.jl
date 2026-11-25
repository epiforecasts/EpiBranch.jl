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
using Distributions: Distribution, MixtureModel, cdf, truncated
using Random: AbstractRNG, default_rng

function EpiBranch.serial_interval(
        model, infector::Individual, infectee::Individual;
        n_trajectories::Integer = 200, rng::AbstractRNG = default_rng()
    )
    attrs = EpiBranch.attributes(model)
    if EpiBranch._onset_law(attrs) === nothing && !(attrs isa EpiBranch.NoAttributes)
        return _trajectory_serial_interval(model, attrs, n_trajectories, rng)
    end
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

# Mixture over `n` draws of the attributes: each draw's kernel conditioned on
# a contact, shifted by minus its onset, weighted by the chance of a contact.
# The onset of the infectee comes from the next draw, an independent case.
function _trajectory_serial_interval(model, attrs, n::Integer, rng)
    n >= 2 || throw(ArgumentError("n_trajectories must be at least 2, got $n"))
    EpiBranch.single_type_offspring(model) isa Distribution || throw(
        ArgumentError(
            "serial_interval needs a fixed offspring distribution, not a function " *
                "of the individual"
        )
    )
    window_kernel = EpiBranch._single_kernel(model)
    kernels, onsets = Any[], Float64[]
    for k in 1:n
        ind = Individual(id = k)
        EpiBranch._apply_attributes!(attrs, rng, ind)
        EpiBranch._set_onset_from_incubation!(ind)
        onset = EpiBranch.onset_time(ind)
        isnan(onset) && continue
        push!(onsets, onset)
        push!(kernels, EpiBranch.get_generation_time(window_kernel, ind))
    end
    isempty(onsets) && throw(
        ArgumentError(
            "serial_interval found no draw of the attributes with an onset time; " *
                "the attributes must record `:incubation_period`"
        )
    )
    m = length(onsets)
    weights = [cdf(kern, Inf) for kern in kernels]
    keep = findall(>(0), weights)
    isempty(keep) && throw(
        ArgumentError("no draw of the attributes has a contact that happens")
    )
    components = map(keep) do i
        kern = kernels[i]
        base = kern isa HazardKernel ? truncated(kern, 0.0, Inf) : kern
        return base + (onsets[mod1(i + 1, m)] - onsets[i])
    end
    return MixtureModel(components, weights[keep] ./ sum(weights[keep]))
end

end # module
