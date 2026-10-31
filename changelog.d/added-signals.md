`TransitionCount` and `Prevalence` read a population quantity (new cases, or
cases currently in a state) off a simulated `SimulationState` over a
reporting step. `SignalData` pairs several of these `Signal`s, each with a
measurement density of your own, as several observed series; `loglikelihood`
simulates the model once and scores every series against that one path,
since separate `~` statements would each simulate their own outbreak.
