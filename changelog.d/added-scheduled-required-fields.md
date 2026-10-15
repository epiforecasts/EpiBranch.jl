A `Scheduled` trigger can declare the population characteristics it reads
with `EpiBranch.required_fields`, alongside the transitions it already
declares with `required_transitions`, so a model missing one of them stops
with a clear error instead of leaving the schedule's gate shut with no
message.
