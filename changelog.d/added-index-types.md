`simulate` takes a new `index_types` keyword for a multi-type
`BranchingProcess`: a vector fixes each index case's type directly (and sets
how many there are), or a `Distribution` draws each one independently, in
place of the previous uniform random choice among the model's types.
