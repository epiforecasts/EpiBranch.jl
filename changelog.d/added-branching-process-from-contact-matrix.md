`BranchingProcess(contacts, q, D, dist_fn, gt; kwargs...)` builds a multi-type
process directly from a contact matrix, a transmission probability `q` (a
single number, or a vector giving per-infectee-type susceptibility), and an
infectious period `D`, doing the transpose and scaling into a
next-generation matrix itself.
