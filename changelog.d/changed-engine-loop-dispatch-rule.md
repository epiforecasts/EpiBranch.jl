`docs/src/design.md` states the engine-loop half of the extension-by-dispatch
rule: a loop asks the composed layers and never decides for them. A loop that
decides what a named intervention does to whom, reads a state key an
intervention owns, or keeps its own record of what has been done to whom has
taken a policy decision into core, where nothing a user writes can reach it.
Where a loop needs a fact about an intervention, it asks a documented trait
with a conservative default that the intervention's own author opts out of.
