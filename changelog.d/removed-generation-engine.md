The generation-based simulation engine is removed. Every model now runs on the
continuous-time race. A model of your own written against `generate_offspring`
runs as a branching process with one infectiousness window. One written
against `contacts_of` proposes each exposure the model names at the time it
gives, and needs no other method: `collect_exposures` and `gather_by_target`
are removed. Cases are settled one at a time in order of infection time. A
person exposed by several cases is infected by the earliest exposure that gets
through, and a blocked exposure is not tried again. A ring of contact tracing
grows past a person on such a model only once they settle, unless the model
declares `EpiBranch.reads_settled_state(::YourModel) = false`, in which case
`contacts_of` is also asked about a traced person who has not settled, as
though they were infected at time 0, and its answer is kept for when they do. These models now call
every intervention hook the built-in ones do, `on_infection_settled!` and
`infectious_removal_time` included, and their seeded runs give different
results.

Also removed are the deprecated `apply_post_transmission!` batch hook,
`settles_generation_batch` and `continuous_actions`. The candidates an
intervention's actions are offered to are chosen by
`EpiBranch.action_candidates` instead. The warning about hooks the
continuous-time race does not call is gone with the second engine. In its
place, `simulate` warns when contact tracing, or an action that needs the
contacts of a case, is given a model that names no contacts, such as
`HomogeneousProcess`.
