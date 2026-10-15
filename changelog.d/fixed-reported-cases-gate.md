A `Scheduled` intervention with `start_after = ReportedCases(n)` no longer
makes the time a run takes grow with the square of the number of individuals.
A run of 40,000 individuals that took about 130 s takes under a tenth of a
second.
