The benchmark suite under `benchmark/` covers the generation engine: a
`BranchingProcess` simulation, with and without isolation and contact
tracing. Every other entry drives a continuous-time race (network,
household or homogeneous), so the suite previously had no case that
exercised the generation-based engine the tutorials and
`docs/src/benchmarks.md` run.
