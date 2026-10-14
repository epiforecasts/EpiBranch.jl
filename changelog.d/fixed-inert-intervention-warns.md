`ModelSpec` now warns when a composed intervention's hooks will not be
called. A hook written with the wrong number of arguments, or defined without
the `EpiBranch.` prefix needed to add a method rather than shadow it,
previously fell back to the package's no-op default with no sign that
anything was missing. The warning names the hook at fault in both cases, and
an intervention with no hook method of any kind is reported on its own. An
intervention that acts through a seam of its own says so by defining
`EpiBranch.has_effect`, which silences that last warning.
