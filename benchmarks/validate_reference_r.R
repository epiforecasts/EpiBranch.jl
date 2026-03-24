#!/usr/bin/env Rscript
# Generate reference values from R epichains for Julia validation

library(epichains)
cat("=== R reference values for EpiBranch.jl validation ===\n\n")

# 1. Borel density
cat("1. Borel density at mu=1\n")
cat("   dborel(1:5, 1) =", dborel(1:5, 1), "\n\n")

# 2. Chain size simulation (Poisson, subcritical)
cat("2. Chain sizes, Poisson(0.9), n=20, threshold=10\n")
set.seed(32)
sizes <- simulate_chain_stats(n_chains=20, statistic="size",
    offspring_dist=rpois, stat_threshold=10, lambda=0.9)
cat("   Sizes:", as.numeric(sizes), "\n")
cat("   Mean:", round(mean(sizes), 2), "\n\n")

# 3. Chain size likelihood
cat("4. Chain size log-likelihood\n")
chain_sizes <- c(4, 7, 1, 2, 7, 2, 3, 1, 5, 6, 1, 10, 5, 10, 6, 8, 8, 6, 7, 10)
ll <- likelihood(chains=chain_sizes, statistic="size",
    offspring_dist=rpois, lambda=0.5)
cat("   LL (Poisson, lambda=0.5):", ll, "\n\n")

# 4. Chain length likelihood
cat("11. Chain length log-likelihood\n")
lengths <- c(0, 1, 0, 2, 1, 0, 0, 3, 0, 1)
ll_len <- likelihood(chains=lengths, statistic="length",
    offspring_dist=rpois, lambda=0.5)
cat("   LL (Poisson, lambda=0.5):", ll_len, "\n\n")

# 5. Extinction probability via simulation
cat("6. Extinction probability: NegBin(R=1.5, k=0.5)\n")
set.seed(42)
sims <- simulate_chain_stats(n_chains=5000, statistic="size",
    offspring_dist=rnbinom, stat_threshold=5000, mu=1.5, size=0.5)
q_sim <- mean(is.finite(sims))
cat("   Simulated P(extinct):", round(q_sim, 4), "\n\n")

# 6. Superspreading (from superspreading package if available)
cat("10. Superspreading proportion (manual calculation)\n")
# NegBin(R=2.5, k=0.16): proportion from top 20%
# Using the Lorenz curve of Gamma(k, 1)
g <- qgamma(0.8, shape=0.16, rate=1)
lorenz_bottom <- pgamma(g, shape=0.16+1, rate=1)
prop_top20 <- 1 - lorenz_bottom
cat("   Top 20% cause", round(prop_top20 * 100, 1), "% of transmission\n\n")

cat("=== Done ===\n")
