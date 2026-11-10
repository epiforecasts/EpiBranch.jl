`progression_loglik` and the built-in `transition_loglik` methods take a
`followup_end` keyword. A step that has not happened by then is scored as
censored, with its survival probability in place of a density. Imputed
infection times near the end of follow-up are therefore no longer scored as if
their onset or report could still be observed. Observed delays are not yet
corrected for right truncation, as the docstring and the households and
networks tutorials note.
