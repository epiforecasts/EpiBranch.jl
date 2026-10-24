`_sellke_race!` no longer boxes a case's transmission routes while settling
the case, on a model whose windows are not all the same type (a different
offspring law, kernel or `until`). Settling a case used to cost memory and
time in proportion to how many routes there were in total rather than how
many of them were reached. That could multiply the memory a run allocates by
tens even where only one route out of many differed.
