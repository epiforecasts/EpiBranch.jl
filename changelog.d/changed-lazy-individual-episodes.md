An `Individual` no longer allocates an `episodes` vector when it is created. The
first `close_episode!` makes the vector. For a person with no episodes,
`ind.episodes` returns a fresh empty vector, and a `push!` onto it is lost.
Use `close_episode!` to archive an episode.
