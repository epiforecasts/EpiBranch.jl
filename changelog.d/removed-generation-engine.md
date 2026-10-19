The generation-based simulation engine is gone, and every model now runs on
the continuous-time race. A model of your own written against
`generate_offspring` runs as a branching process with one infectiousness
window, and one written against `contacts_of` as a route that proposes each
exposure the model names at the time it gives. Such a model now needs only
`contacts_of`: `collect_exposures` and `gather_by_target` are removed. Cases
are settled one at a time in order of infection time, so a person exposed by
several cases is infected by the earliest exposure that gets through, and an
exposure that is blocked is not tried again. Every intervention hook is called
on these models as on the built-in ones, including `on_infection_settled!`
and `infectious_removal_time`, and seeded runs of them give different results.

Removed with it: the deprecated `apply_post_transmission!` batch hook,
`settles_generation_batch`, `continuous_actions` (its one remaining use, which
candidates an intervention's actions are offered to, is
`EpiBranch.action_candidates`), and the warning that an intervention had
hooks the continuous-time race does not call, since there is no longer
another engine to call them. What remains is a warning when contact tracing,
or actions that need a case's contacts, is given a model that names none,
such as `HomogeneousProcess`.
