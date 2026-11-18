A `Transition` and an `Infectiousness` window now resolve the state key of
their `from` name once, when built, instead of building a new symbol for every
case. The simulated outcome is unchanged. A branching process with a
multi-step progression runs faster.
