`_sellke_race!` no longer boxes a case's transmission routes while settling
the case, on a model whose windows are not all the same type (a different
offspring law, kernel or `until`). Settling a case used to cost memory and
time in proportion to how many routes there were in total rather than to
how many of them were reached, which could multiply a run's allocation by
tens even where only one route out of many differed.
