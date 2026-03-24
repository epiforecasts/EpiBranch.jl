#!/usr/bin/env Rscript
# Reference values from R superspreading package

library(superspreading)
cat("=== R superspreading reference values ===\n\n")

# 1. probability_extinct
cat("1. probability_extinct(R=1.5, k=0.1, num_init_infect=10)\n")
cat("   ", probability_extinct(R=1.5, k=0.1, num_init_infect=10), "\n\n")

# 2. probability_epidemic
cat("2. probability_epidemic(R=1.5, k=0.1, num_init_infect=10)\n")
cat("   ", probability_epidemic(R=1.5, k=0.1, num_init_infect=10), "\n\n")

# 3. probability_contain with pop_control
cat("3. probability_contain(R=1.5, k=0.5, num_init_infect=1, pop_control=0.1)\n")
cat("   ", probability_contain(R=1.5, k=0.5, num_init_infect=1, pop_control=0.1), "\n\n")

# 4. probability_contain with ind_control
cat("4. probability_contain(R=1.5, k=0.5, num_init_infect=1, ind_control=0.1)\n")
cat("   ", probability_contain(R=1.5, k=0.5, num_init_infect=1, ind_control=0.1), "\n\n")

# 5. probability_contain with both controls
cat("5. probability_contain(R=1.5, k=0.5, num_init_infect=1, ind_control=0.1, pop_control=0.1)\n")
cat("   ", probability_contain(R=1.5, k=0.5, num_init_infect=1, ind_control=0.1, pop_control=0.1), "\n\n")

# 6. probability_contain with multiple introductions
cat("6. probability_contain(R=1.5, k=0.5, num_init_infect=5, pop_control=0.1)\n")
cat("   ", probability_contain(R=1.5, k=0.5, num_init_infect=5, pop_control=0.1), "\n\n")

# 7. proportion_cluster_size
cat("7. proportion_cluster_size(R=2, k=0.1, cluster_size=10)\n")
res <- proportion_cluster_size(R=2, k=0.1, cluster_size=10)
cat("   ", as.numeric(res[, 3]), "\n\n")

# 8. proportion_transmission (80/20 rule)
cat("8. proportion_transmission(R=2, k=0.5, prop_transmission=0.8)\n")
res <- proportion_transmission(R=2, k=0.5, prop_transmission=0.8)
cat("   ", as.numeric(res[, 3]), "\n\n")

# 9. calc_network_R (NATSAL data)
cat("9. calc_network_R(mean=14.1, sd=69.6, dur=1, prob=1, age=c(16,74))\n")
res <- calc_network_R(mean_num_contact=14.1, sd_num_contact=69.6,
                       infect_duration=1, prob_transmission=1, age_range=c(16,74))
cat("   R =", res["R"], ", R_net =", res["R_net"], "\n\n")

# 10. probability_epidemic varying k (from vignette)
cat("10. probability_epidemic varying k (R=1.5)\n")
for (k in c(1, 0.5, 0.1)) {
  p <- probability_epidemic(R=1.5, k=k, num_init_infect=1)
  cat("    k=", k, ": ", p, "\n")
}
cat("\n")

# 11. probability_epidemic varying R (from vignette)
cat("11. probability_epidemic varying R (k=1)\n")
for (R in c(0.5, 1.0, 1.5, 5.0)) {
  p <- probability_epidemic(R=R, k=1, num_init_infect=1)
  cat("    R=", R, ": ", p, "\n")
}
cat("\n")

# 12. Extinction probability across R for multiple k (Lloyd-Smith Fig 2B)
cat("12. probability_extinct grid (from epidemic_risk vignette)\n")
for (k in c(0.01, 0.1, 0.5, 1, 4)) {
  p <- probability_extinct(R=3, k=k, num_init_infect=1)
  cat("    R=3, k=", k, ": ", round(p, 4), "\n")
}
cat("\n")

# 13. Containment under pop control (Lloyd-Smith Fig 3C)
cat("13. probability_contain grid (R=3, varying k and control)\n")
for (k in c(0.1, 0.5, 1)) {
  for (ctrl in c(0.25, 0.5, 0.75)) {
    p <- probability_contain(R=3, k=k, num_init_infect=1, pop_control=ctrl)
    cat("    k=", k, ", ctrl=", ctrl, ": ", round(p, 4), "\n")
  }
}
cat("\n")

cat("=== Done ===\n")
