The chain-length likelihood and `chain_length_distribution(model)` now
reject a `Reporting` step with probability below 1 because the observed
chain length is undefined when cases go undetected. Previously both
calculated lengths over all cases without accounting for incomplete reporting.
