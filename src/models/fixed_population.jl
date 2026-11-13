"""
    FixedPopulationProcess <: TransmissionModel

Supertype for a process over a fixed, enumerated population whose race runs on
`sellke_race!` with one infectiousness window per host: `HomogeneousProcess`,
`MixingProcess`, `NetworkProcess` and `HouseholdProcess` subtype it, and so can
a structure written outside the package. It supplies the defaults those four
share:

- [`honours_termination_controls`](@ref EpiBranch.honours_termination_controls)
  is `false`, so the `max_cases` and `max_generations` keyword defaults are not
  applied and a run goes to extinction or `max_time`;
- [`validate_process_windows`](@ref EpiBranch.validate_process_windows) warns
  when the window's `until` states miss a terminal state of the progression;
- [`infection_layer_columns`](@ref EpiBranch.infection_layer_columns) reads the
  window from [`process_window`](@ref EpiBranch.process_window).

A subtype with fields `from` (a `Symbol` or `nothing`) and `until` (a `Tuple` of
states) needs nothing else. One that names them differently, or builds its
window another way, extends `process_window`.
"""
abstract type FixedPopulationProcess <: TransmissionModel end

honours_termination_controls(::FixedPopulationProcess) = false

"""
    process_window(process::FixedPopulationProcess) -> (from, until)

The state at which a host's infectious window opens (`nothing` to derive it from
the progression) and the tuple of states that close it. The default reads the
fields `from` and `until` of `process`.
"""
process_window(process::FixedPopulationProcess) = (process.from, process.until)

function validate_process_windows(process::FixedPopulationProcess, progression)
    from, until = process_window(process)
    return warn_uncovered_terminal_states(until, progression; from)
end
