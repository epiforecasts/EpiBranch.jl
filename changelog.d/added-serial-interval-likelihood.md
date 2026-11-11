`serial_interval(model, infector, infectee)` gives the distribution of the
serial interval between a linked pair, as the generation time plus the
infectee's incubation period minus the infector's, for fitting timing data
such as serial intervals from contact-traced pairs. Pass it to
CensoredDistributions.jl's `double_interval_censored` for the likelihood
under daily reporting and right truncation. Available once
CensoredDistributions.jl and ConvolvedDistributions.jl are loaded, through a
new package extension.
