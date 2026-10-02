The household likelihood can condition each household on its earliest infection
instead of its recruited index: `condition_on = EarliestInfected()` on
`loglikelihood(data, model)` and `compile_household_pairs(data)`, against the
default `RecruitedIndex()`. A recruited index need not be the first household
member infected, and augmenting an earlier infection onto a household-mate
otherwise makes the draw impossible. The conditioned host is resolved on every
call, so recompile the layout each evaluation. A rule of your own subtypes
`ConditionOn` and supplies one `condition_mask` method.
