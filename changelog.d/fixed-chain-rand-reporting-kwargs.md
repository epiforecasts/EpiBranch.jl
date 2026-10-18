`rand` on a `chain_size_distribution` wrapper now draws the reported cluster
size, applying the same reporting and observation `logpdf` scores against,
rather than the unobserved total number infected; a cluster with no reported
case is redrawn. `rand` on both `chain_size_distribution` and
`chain_length_distribution` also no longer forwards a stored likelihood-only
keyword such as `n_sim` to `simulate`, which it does not accept.
