`NetworkProcess` no longer copies the adjacency list when it is already a
`Vector{Vector{Int}}`, avoiding the allocation and garbage-collection cost of
copying every neighbour vector when building it on a large network. The
adjacency list is then shared with the caller and must not be changed after
the `NetworkProcess` is built.
