# ─────────────────────────────────────────────────────────────────────
# Cross-package extension contract.
#
# Empty generic declarations for every function that more than one
# slot-in package extends. Declaring them here lets each slot-in
# depend only on `EpiBranchCore` (rather than on the package that
# happens to define the first method).
#
# Concrete methods live in:
#   - `EpiBranchProcess`: `simulate`, `simulate_batch`, `step!`,
#     `make_contact!`, `draw_offspring` (default and BranchingProcess
#     methods).
#   - `EpiBranchObservation`: `simulate(::Observed{...})`.
#   - `EpiBranchAnalytics`: `chain_size_distribution` for offspring specs.
# ─────────────────────────────────────────────────────────────────────

# Engine seam
function simulate end
function simulate_batch end
function step! end
function make_contact! end
function draw_offspring end

# Analytics seam
function chain_size_distribution end
