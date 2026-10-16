`household_infections` and `network_infections` now throw when a simulated
removal recorded a releasing stretch through `record_removal!` under a key no
component names through `removal_gap_host_times`. Such a stretch previously
dropped out of the infection layer with no warning, so the likelihood fitted
on the whole exposure while the simulation had blocked part of it. A
component that means to leave a key out on purpose, such as a `Scheduled`
that can lapse, now says so through the new `removal_gap_host_times_withdrawn`,
which silences the error for that key.
