The chain-length likelihood and `chain_length_distribution(model)` now
reject a `Reporting` step with probability below 1, because undetected
cases leave the observed chain length undefined. Previously both used
lengths over all cases and silently ignored incomplete reporting.
