The benchmark suite under `benchmark/` gains a `BranchingProcess` case, with
and without isolation and contact tracing, run on the same continuous-time
race as every other entry (network, household or homogeneous); the suite
previously had none that drove the race through `BranchingProcess`, as the
tutorials and `docs/src/benchmarks.md` do.
