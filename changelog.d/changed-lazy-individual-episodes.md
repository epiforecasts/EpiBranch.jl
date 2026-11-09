An `Individual` no longer allocates an `episodes` vector when it is created. The
vector is made by the first `close_episode!`, and `ind.episodes` returns an empty
one for a person with none, so a `push!` straight onto it is lost:
`close_episode!` is the only supported way to archive an episode.
