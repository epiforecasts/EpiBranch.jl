`exclusive_probabilities` named its shared draw's state key with `gensym`,
giving it a `##exclusive_draw#N` name. `linelist` hides internal state by the
leading underscore on its name and did not recognise this one, so the draw
leaked through as a column with a session-dependent name. The key is now
built from a counter under a `_`-prefixed name instead, which `linelist`
already drops.
