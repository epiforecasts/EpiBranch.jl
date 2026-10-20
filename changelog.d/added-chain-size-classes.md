`ChainSizeClasses` holds chain sizes recorded only in size classes, such as
the number of isolated cases, of chains with 2 to 9 cases, and of chains with
10 or more. `loglikelihood` sums each class's `counts[i] * log P(lower[i] ≤
X ≤ upper[i])` under any offspring distribution or chain-size law, rather
than making up exact sizes to stand in for the classes.
