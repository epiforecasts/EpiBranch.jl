#!/usr/bin/env julia
# Fail when a benchmark has slowed down by more than a set factor against the
# base revision, so a large regression shows as a failed check rather than only
# as a row in the comparison comment. Each file holds the fastest round per
# benchmark, as the Benchmark workflow writes it.
#
#   julia benchmark/regressions.jl pr.json base.json [threshold]
#
# The threshold is a ratio of minimum times (default 1.25, i.e. 25% slower).
# Pull requests that leave performance alone stay within about 7% of their
# base on the hosted runners, so a smaller threshold would fail on noise.

using BenchmarkTools
using BenchmarkTools: leaves

pr_file, base_file = ARGS[1], ARGS[2]
threshold = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.25

minimum_times(file) = Dict(
    join(string.(key), " / ") => time(minimum(trial))
        for (key, trial) in leaves(BenchmarkTools.load(file)[1])
)

pr, base = minimum_times(pr_file), minimum_times(base_file)
slower = sort!(
    [(name, pr[name] / base[name]) for name in keys(pr) if haskey(base, name)];
    by = last, rev = true
)
regressions = filter(r -> last(r) > threshold, slower)

for (name, ratio) in regressions
    println("::error::$(name) is $(round(ratio; digits = 2))x slower than the base")
end
summary = get(ENV, "GITHUB_STEP_SUMMARY", "")
if !isempty(summary)
    open(summary, "a") do io
        println(io, "### Benchmark regressions\n")
        if isempty(regressions)
            println(io, "No benchmark is more than $(threshold)x slower than the base.")
        else
            println(io, "Slower than the base by more than $(threshold)x:\n")
            println(io, "| Benchmark | Ratio |\n|---|---|")
            for (name, ratio) in regressions
                println(io, "| ", name, " | ", round(ratio; digits = 2), "x |")
            end
        end
    end
end
println(
    length(regressions), " of ", length(slower), " benchmarks more than ",
    threshold, "x slower than the base"
)
exit(isempty(regressions) ? 0 : 1)
