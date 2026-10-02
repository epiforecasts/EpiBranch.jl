`docs/src/design.md` states the engine-loop half of the extension-by-dispatch
rule: a loop asks the composed layers and never decides for them. A loop that
names a concrete intervention type, reads a state key an intervention owns, or
keeps its own record of what has been done to whom has taken a policy decision
into core, where nothing a user writes can reach it.
