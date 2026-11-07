`copy(state)` on a `SimulationState` gives an independent copy that shares each
person with the original until one of them looks at that person, and only then
takes a private copy of them. A particle filter that copies every particle's
state at each resampling step then pays for the people a continuation reaches
rather than for the whole population: in a homogeneous population of 50,000 a
`copy` takes a few milliseconds where a `deepcopy` takes about 0.2 s. The random
number generator and the pending events of an unfinished race are copied
outright, and a run continued from a copy agrees with one continued from a
`deepcopy` of the same state.
