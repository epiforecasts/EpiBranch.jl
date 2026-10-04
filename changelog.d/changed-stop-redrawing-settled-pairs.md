On a race between two standing nodes (a household clique, a network edge), a
block that is certain and declared permanent now ends the pair instead of
redrawing towards a foregone conclusion: the race stops proposing along that
edge, which is what lets an unbounded window terminate. A source declares
permanence with `EpiBranch.standing_block`, because the `Risk` it returns
cannot say whether the block will still be in force at the next proposal, and
an aborted infector's onward risk declares it already. The pair remains in each
other's standing contacts for tracing and ring construction, which read that
relationship rather than the proposals; an output logging every contact event
still needs the draws this skips.
