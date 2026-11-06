# EpiNetwork.jl

Transmission over a fixed contact network, built on
[EpiBranch.jl](https://github.com/epiforecasts/EpiBranch.jl).

`NetworkProcess` transmits along the edges of a fixed graph: each node is a
person, and along each edge the time from an infector becoming infectious to
its next infectious contact with a neighbour is drawn from a contact-interval
kernel (Kenah 2011). Transmission happens only while the infector is still
infectious, so shortening that window — through recovery or isolation —
curtails onward spread; each node can be infected at most once. It is a
structure-driven model simulated by the **Sellke construction** in continuous
time, like `HouseholdProcess` in EpiHouseholds. The latent period, infectious
period, onset and testing are flexible `Transition`s on the shared
natural-history timeline, so a network line list is an EpiBranch line list.

```julia
using EpiBranch, EpiNetwork, Distributions, Graphs, StableRNGs

# A small-world network of 400 people: mostly local contacts, so that the
# contacts of a person tend to be in contact with each other, with a few
# long-range links.
g = watts_strogatz(400, 6, 0.1; rng = StableRNG(1))

model = ModelSpec(NetworkProcess(g, Exponential(3.0));
    progression = [Transition(:recovered; from = :infection, delay = 7.0, terminal = true)])
state = simulate(model; n_initial = 1, rng = StableRNG(3))
```

`NetworkProcess` was originally part of EpiBranch.jl and was split into this
subpackage so the core package stays focused on the branching-process models;
the structured network model lives here alongside future network and household
extensions.
