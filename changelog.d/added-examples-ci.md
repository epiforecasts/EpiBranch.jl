The scripts under `examples/` now run as part of the test suite, with a small
number of simulations, so a change to the public API they call can no longer
break them unnoticed. `examples/ringbp_clone.jl` reads its simulation count from
`EPIBRANCH_EXAMPLE_NSIM`.
