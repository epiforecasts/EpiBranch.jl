`MultiTypeOffspring` and `BranchingProcess` built from an offspring matrix now
also accept one offspring law per type, as a vector of distributions, instead
of a function of the column sum. This lets types' offspring laws differ in
more than their mean, for example one heavy-tailed and one not, while
`reproduction_number` and `extinction_probability` still read it exactly.
