`HomogeneousProcess`, `MixingProcess`, `NetworkProcess`, `RoutedNetwork` and
`HouseholdProcess` used to ignore `max_cases`, `max_generations` and every
stopping rule but `MaxTime`, running to extinction or `max_time` regardless of
what else was asked for. Their continuous-time race now polls `should_stop`
after each settled case for a rule not already captured by `time_bound` alone;
an explicit `MaxCases`, or a rule of your own, ends such a run as it does on
the generation-based engine. `MaxGenerations` needs its own treatment, since a
race settles cases in time order rather than generation order: the new
`EpiBranch.generation_bound(rule)` trait, alongside `time_bound`, drops a
settled case's own proposals once its generation reaches the bound, leaving
every case beyond that depth undrawn, while everything within it still runs
to its own extinction or `max_time`. A `HouseholdProcess` puts every household
on one shared clock whenever such a rule is set, as it already does for an
intervention that reads population-wide state, so the cap lands on the
chronologically earliest cases rather than on whichever household's own race
happened to run first.

The `max_cases` and `max_generations` keyword *defaults* still stay off on
these five models, so an existing run over a large fixed population is not
silently capped; give an explicit value, or a `stopping_rules` entry, to end
such a run early.
