`exclusive_probabilities` named its shared draw's state key with `gensym`,
which produced a key like `##exclusive_draw#N`. `linelist` hides internal
state by the leading underscore on its name and did not recognise this one,
so the draw leaked through as a column with a session-dependent name. The
key now comes from a counter under a `_`-prefixed name, which `linelist`
already drops.
