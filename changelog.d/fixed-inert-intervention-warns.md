`ModelSpec` now warns when a composed intervention has no method of its own,
at the right arity, for any hook the engine calls. A hook written with the
wrong number of arguments, or defined without the `EpiBranch.` prefix needed
to add a method rather than shadow it, previously fell back to the package's
no-op default with no sign that the intervention was doing nothing.
