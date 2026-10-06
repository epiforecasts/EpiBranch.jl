**Breaking.** `Quarantine` takes a required `duration` keyword, which 0.1.0 had
no equivalent of, its `Quarantine` being a singleton. There is no default, since
an indefinite quarantine is a choice to make rather than one to inherit: pass
`Inf` to keep 0.1.0's behaviour, or a finite value, distribution or
`(rng, ind)` callable to give the contact a release time. `ContactTracing`
requires its `action` on every constructor for the same reason, and
`quarantine_on_trace` is deprecated in favour of it.
