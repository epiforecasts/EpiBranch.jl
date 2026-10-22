`BranchingProcess(contacts, q, D, dist_fn, gt; kwargs...)` builds a multi-type
process directly from a contact matrix, a transmission probability `q` (a
single number, or a vector giving per-infectee-type susceptibility), and an
infectious period `D`, transposing and scaling the contact matrix into a
next-generation matrix itself.
