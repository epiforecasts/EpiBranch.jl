Three places where core tested a concrete type or a field's presence instead
of dispatching now read a trait instead, per `docs/src/design.md`'s
extension-by-dispatch rule:

- A structure-driven (Sellke) run's end time now reads a new `time_bound(rule)`
  on each stopping rule (default `Inf`, `MaxTime` overriding it) rather than
  matching `MaxTime` by type, so a user-defined stopping rule that overrides
  `time_bound` can end such a run. The warning about ignored termination
  controls reads a second trait, `honoured_without_should_stop(rule)` (default
  `false`; `Extinction` and `MaxTime` override it), and names the rules it
  reports, distinguishing one that did nothing from one whose time bound was
  applied while its other conditions were not.
- `terminal_certainty(transition)` replaces reading a terminal transition's
  `probability` field by `hasproperty`, which misjudged a custom transition
  either way: certain when it happened to have no `probability` field,
  however conditional it was, and judged by an unrelated field when it did.
  `Recovery`, `Death` and `Transition` declare it; the default is `missing`
  (unknown), as `_terminal_target` already did for the `until`-coverage check.
- `supports_waning(mode::AbstractEffectMode)` replaces `VaccineEffect`'s
  constructor testing `mode isa AllOrNothingMode`. Default `true`,
  `AllOrNothingMode` overriding it to `false`; a third-party effect mode that
  is likewise all-or-nothing can now reject `waning` the same way.
