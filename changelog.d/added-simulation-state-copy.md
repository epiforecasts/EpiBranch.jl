`copy(state)` on a `SimulationState` gives an independent copy. Each person is
shared with the original until accessed through either state, when that state
takes a private copy of the person. For a particle filter that copies each
particle state at resampling, the cost of copying people depends on the number
accessed during continuation. In a homogeneous population of 50,000, a `copy`
takes a few milliseconds compared with about 0.2 s for a `deepcopy`. The random
number generator and pending events of an unfinished race are copied in full.
A run continued from a copy agrees with one continued from a `deepcopy` of the
same state.
