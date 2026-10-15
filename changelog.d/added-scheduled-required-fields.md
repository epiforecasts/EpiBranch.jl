A `Scheduled` trigger can declare the population characteristics it reads
with `EpiBranch.required_fields`, alongside the transitions it already
declares with `required_transitions`. A model missing either one now stops
with a clear error, rather than leaving the schedule's gate shut with no
message.
