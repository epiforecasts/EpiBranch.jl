A `Transition(:susceptible_again; from = :recovered, ...)` timed from a
`terminal = true` recovery was reset to `false` on every case because its
time lies after the outcome. Nobody waned and reinfection never happened.
`Transition`, `Reporting` and `Hospitalisation` now keep an event timed from
the outcome state itself and still censor events timed from an earlier state.
