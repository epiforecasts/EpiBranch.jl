A continuous-time (Sellke) race stops proposing a pair's contacts once a
certain, non-fading block has settled it for good (`EpiBranch.standing_block`),
which leaves the pair's standing relationship untouched but drops the later
contact events between it and its infector. A `ContactRecorder` attached to a
`ModelSpec`'s `recorder` is asked, every time such a block would end a pair's
draws, whether they still matter (`EpiBranch.records_contacts`); the default
`NoContactRecorder` answers no to every pair, so a run with none attached is
unaffected. Resuming the draws puts the pair back under the
rejection-continuation guard, so attaching a recorder narrows which models can
run, exactly as a model with no standing block never reached.
