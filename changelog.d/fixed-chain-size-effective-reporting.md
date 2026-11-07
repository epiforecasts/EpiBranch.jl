The chain-size closed form now uses the probability that a case reaches the
reporting step's `from` state as well as its reporting probability, accounting
for asymptomatic cases when reporting from onset. The end-of-outbreak check
uses the same probability. Chain-size calculations fall back to simulation
when outcomes can censor reports or the reporting anchor cannot be read;
the end-of-outbreak check rejects these models.
