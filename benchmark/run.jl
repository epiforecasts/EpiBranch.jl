#!/usr/bin/env julia
# Taken from EpiAwarePackageTools' templates. No sync manages it yet, so edit
# it here and carry the change over if this repository adopts the kit.
#
# Run this checkout's benchmark suite and save the results to JSON, via the
# shared EpiAwarePackageTools benchmark harness. The package owns `benchmarks.jl`
# (which defines `SUITE`); this runner is standard.
#
#   julia --project=benchmark benchmark/run.jl [out.json]

using EpiAwarePackageTools.Benchmarks: run_suite

out_file = get(ARGS, 1, "results.json")

include(joinpath(@__DIR__, "benchmarks.jl"))  # defines `SUITE`

run_suite(SUITE; out_file = out_file, seconds = 1, verbose = true)
println("Saved benchmark results to ", out_file)
