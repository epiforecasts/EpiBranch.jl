# Parameter sweeps across processes

A sweep over a grid of parameters, such as isolation delays crossed with reproduction numbers, is many independent runs of `containment_probability` or `simulate`. This page shows how to spread a sweep over separate Julia processes with the standard library's `Distributed`, and why you might prefer that to `parallel = true`.

The code blocks on this page are not run when the documentation is built, because they start worker processes.

## Processes or threads

`parallel = true` spreads the runs of one call over the threads of one Julia process. Garbage collection in Julia pauses every thread in the process. A run that allocates heavily therefore spends much of its wall time with all threads stopped, and more threads do not shorten that time. Separate processes each have their own heap and collect their own garbage, so one worker's pause does not stop the others.

Processes beat threads when a sweep has many runs that allocate heavily: large outbreaks, many interventions, line lists or recorders. For a small number of cheap runs, threads are simpler and start faster, because a process has to load EpiBranch before it does any work. The two combine: each worker can start with several threads and pass `parallel = true`, though one process per core is the usual first choice.

## A sweep over a grid

The function that runs one grid point goes on every worker with `@everywhere`, so that `pmap` can call it wherever it sends the point. It takes the point and a seed, and builds its own random number generator from that seed.

```julia
using Distributed

addprocs(4; exeflags = "--project")   # four workers, using the current project

@everywhere using EpiBranch, Distributions, Random

@everywhere function run_point(point, seed)
    (; R, delay) = point
    model = ModelSpec(
        BranchingProcess(NegBin(R, 0.5), LogNormal(1.6, 0.5));
        interventions = [
            Isolation(onset_to_isolation_delay = Exponential(delay), duration = 7.0),
        ],
        attributes = clinical_presentation(incubation_period = LogNormal(1.6, 0.5))
    )
    p = containment_probability(model, 1000; max_cases = 500, rng = Xoshiro(seed))
    return (; R, delay, containment = p)
end

grid = [(; R, delay) for R in [1.5, 2.5] for delay in [1.0, 3.0, 5.0, 7.0]]
seeds = 1000 .+ eachindex(grid)   # one seed per grid point

results = pmap(run_point, grid, seeds)
```

`results` is a vector of named tuples in the order of `grid`, which `DataFrame(results)` turns into a table.

## Seeding

Give every grid point its own seed, as above, rather than sharing one generator or seeding by worker. `pmap` hands points to whichever worker is free, and the worker that runs a point changes from run to run and with the number of workers. With a seed tied to the point, it gives the same result wherever it runs. The sweep then gives the same answer with 1 worker, 4 workers or `map(run_point, grid, seeds)` in a single process, and rerunning one point on its own reproduces its row.

Points should have different seeds, but they need not be unrelated to each other: `1000 .+ eachindex(grid)` is enough, because `Xoshiro(seed)` expands each seed into a well-mixed state. To change the whole sweep, change the offset.

## Collecting results

`pmap` returns the values that `run_point` returns. Return the summary you want rather than the `SimulationState`s: a number or a small tuple keeps the data sent between processes small. `containment_probability(model, n)` already does this for the containment probability. To keep more from each run, summarise on the worker, with `chain_statistics` or `weekly_incidence` for instance, and return that table.

`scenario_sweep` loops over a grid inside one process and does not distribute it. For a large grid, write the grid and the function that runs a point yourself, as above.

## On a cluster

The same code runs under a cluster manager. With [SlurmClusterManager.jl](https://github.com/JuliaParallel/SlurmClusterManager.jl), replace `addprocs(4; ...)` with

```julia
using Distributed, SlurmClusterManager

addprocs(SlurmManager())
```

in a script submitted with `sbatch`, and the rest is unchanged. The workers need the same project environment as the script, on a filesystem they all see, with the packages precompiled before the job starts so that the workers do not all compile at once.
