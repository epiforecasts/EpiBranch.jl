`NetworkProcess` no longer copies the adjacency list when it is already a
`Vector{Vector{Int}}`, so building it on a large network no longer pays the
allocation and garbage-collection cost of copying every neighbour vector.
Since the adjacency list is then shared with the caller, it must not be
changed after the `NetworkProcess` is built.
