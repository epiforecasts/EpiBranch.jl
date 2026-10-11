**Breaking.** `Isolation` takes a required `duration` keyword, which 0.1.0 had
no equivalent of. There is no default, since indefinite isolation is a choice
to make rather than one to inherit: pass `Inf` for a removal that never
releases, or a finite value, distribution or `(rng, ind)` callable to give it a
release time. `Quarantine` takes the same keyword under the same name.
