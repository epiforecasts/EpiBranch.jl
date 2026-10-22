`MultiTypeOffspring`'s docstring now warns that a contact matrix from a
survey must be transposed (`permutedims`) before it can serve as the
next-generation matrix `M`. The multi-type tutorial now builds its worked
example from an asymmetric contact matrix throughout, comparing the
reproduction numbers from passing the matrix as given against those from
its transpose.
