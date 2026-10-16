`RingVaccination` now moves a traced contact's dose earlier when a
re-trace finds an earlier `:trace_time`. Infectors settle in order of
infection, not of isolation, so a contact traced late from one infector
can later be traced earlier from another whose isolation came first; the
dose used to stay at the first trace found even once a genuinely earlier
one turned up.
