A benchmark suite under `benchmark/`, covering simulation (network races with a
shared and a live pair kernel, a live kernel under a policy, households) and
evaluation (`compile_contact_pairs`, `pairwise_surv_loglik`,
`extinction_probability`, the chain-size log-likelihood). `task benchmark` runs
it. A pull request touching `src/`, `lib/` or the suite itself is benchmarked
against its base revision, and the comparison is posted as a single comment.

The older scripts in `benchmarks/`, which compare against R, keep their own task
under the name `benchmark-r`.
