`docs/src/contributing.md` states the engine-loop half of the
extension-by-dispatch rule: a loop should ask the composed layers and never
decide for them. A loop that decides what a named intervention does to whom,
reads a state key an intervention owns, or keeps its own record of what has
been done to whom has taken a policy decision into core, where nothing a user
writes can reach it. Where a loop needs a fact about an intervention, the
shape is a documented trait with a conservative default the intervention's own
author opts out of. The rule is the target: several loops do not meet it yet,
and the document says so.
