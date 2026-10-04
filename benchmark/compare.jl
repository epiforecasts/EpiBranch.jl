#!/usr/bin/env julia
# Taken from EpiAwarePackageTools' templates. No sync manages it yet, so edit
# it here and carry the change over if this repository adopts the kit.
#
# Compare two benchmark result files and write a Markdown PR comment, via the
# shared EpiAwarePackageTools benchmark harness. `BACKEND_ORDER` orders the
# automatic-differentiation rows the harness folds into a matrix; this suite has
# none, so that section of the comment is empty and the order is kept for when
# it does.
#
#   julia --project=benchmark benchmark/compare.jl pr.json base.json out.md

using EpiAwarePackageTools.Benchmarks: compare_comment

const BACKEND_ORDER = [
    "ForwardDiff", "ReverseDiff (tape)", "Mooncake reverse",
    "Mooncake forward", "Enzyme reverse", "Enzyme forward",
]

pr_file, base_file, out_file = ARGS[1], ARGS[2], ARGS[3]

comment = compare_comment(pr_file, base_file; backend_order = BACKEND_ORDER)
write(out_file, comment)
println("Wrote benchmark comparison to ", out_file)
