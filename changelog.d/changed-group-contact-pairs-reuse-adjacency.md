`compile_contact_pairs` for a membership vector now builds each group's contact
lists and reuses the adjacency method, instead of keeping a second copy of the
row-building code. It no longer allocates a bucket for every integer between the
smallest and largest label, so sparse labels cost nothing extra. Rows within a
susceptible are ordered by infector, which changes log-likelihood sums only by
rounding.
