A benchmark suite under `benchmark/`, covering simulation (network races with a
shared and a live pair kernel, a live kernel under a policy, households) and
evaluation (`compile_contact_pairs`, `pairwise_surv_loglik`,
`extinction_probability`, the chain-size log-likelihood). `task benchmark` runs
it, and `task benchmark-r` runs the standalone R-comparison scripts in
`benchmarks/`. A pull request touching `src/`, `lib/` or the suite itself is
benchmarked against its base revision, and the comparison is posted as a single
comment.
