A `Transition` and an `Infectiousness` window now resolve the state key of
their `from` name once, when built, rather than building a new symbol for every
case. The simulated outcome is unchanged; a branching process with a
multi-step progression runs faster.
