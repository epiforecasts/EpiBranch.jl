The chain-size closed form now accounts for the probability that a case reaches
the `from` state of the reporting step as well as its reporting probability.
This covers asymptomatic cases when reporting is from onset. The
end-of-outbreak check uses the same probability. Chain-size calculations fall
back to simulation when outcomes can censor reports or the reporting state
cannot be determined. The end-of-outbreak check rejects these models.
