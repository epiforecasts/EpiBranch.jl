A function `generation_time`, `(ind) -> distribution`, is no longer called
for a case whose drawn number of contacts turns out to be zero. Such a
function can read something a contactless case never reaches, such as an
infectious period; it is now asked for a distribution only once a case is
known to have at least one contact to time.
