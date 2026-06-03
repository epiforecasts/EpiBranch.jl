# RFC: Thompson 2019 EOO under under-reporting (#33)

The current code (`src/analytical/end_of_outbreak_probability.jl`) implements the closed form for full reporting (`ρ = 1`) and throws a friendly error for `ρ < 1`, pointing at this issue.

This RFC captures the plan for the under-reporting case so the implementation work can pick it up later.

## Maths

Per the issue body. The direct-offspring approximation:

```
π(τ) ≈ (1 + ρ·R·S(τ)/k)^(-k)
```

treats unobserved cases as silent. It's tight near `ρ = 1` and biases low at low `ρ` because it ignores hazard from undetected descendants.

The exact Thompson, Morgan & Jansen (2019, [doi:10.1098/rstb.2018.0431](https://doi.org/10.1098/rstb.2018.0431)) form folds the unobserved sub-tree propagation into a recursive integral equation. For NegBin offspring with dispersion `k` and effective per-case report rate `ρ`, the per-case extinction probability `η(τ)` satisfies a fixed point of the form:

```
η(τ) = ∫ f(η(τ-t); R, k, ρ) g(t) dt
```

with `g(t)` the generation time density. A numerical solver iterates the recursion to convergence.

## Implementation plan

1. **Solver.** A `_per_case_extinction_probability(ρ, R, k, gt, τ; tol, maxiter)` function in `src/analytical/end_of_outbreak_probability.jl` that runs the fixed-point iteration. Use Picard iteration first; fall back to Anderson acceleration if convergence is too slow on the parameter ranges Endo-style data covers.

2. **Quadrature.** The convolution against `g(t)` needs adaptive Gauss-Kronrod (QuadGK is already a dep) or a dense grid with trapezoidal rule. Start with QuadGK; benchmark later if it's the hot path.

3. **AD through the fixed point.** Three options:
   - **`ImplicitDifferentiation.jl`**: declarative, handles the fixed-point gradient cleanly via the implicit function theorem. Right shape for this.
   - **Hand-rolled `ChainRules`**: more work but no extra dep.
   - **Run AD through the iterations**: easiest, expensive at convergence (ForwardDiff dual-prop through hundreds of iters).
   
   First pass: ImplicitDifferentiation if it adds <50ms to startup, otherwise hand-rolled rules.

4. **Tests.**
   - `ρ → 1` limit: result matches the existing closed form within `tol`.
   - Forward simulation: run `simulate_batch` under the same parameters, take the fraction of clusters dead by `τ`, compare to the analytical value with appropriate CI.
   - AD smoke test: gradients of EOO w.r.t. R and k are finite and continuous through `ρ < 1`.

5. **Wire-up.**
   - Replace the `throw(...)` branch in `end_of_outbreak_probability(::Observed{<:BranchingProcess, <:PerCaseObservation, <:Snapshot}, τ)` with a call to the solver.
   - Keep the direct-offspring approximation behind an opt-in (e.g. `approximation = :direct_offspring`) for speed-sensitive sweeps where the bias is acceptable.

6. **Docs.**
   - Section in `docs/src/tutorials/real_time.md` (or wherever EOO lives) explaining the two paths and when to choose each.
   - Citation to Thompson 2019.

## Sequencing

Per the issue: "this comes before timed-reports inference (which would inherit the same fixed-point machinery if the renewal-equation likelihood ever needs the same correction)."

So: do the EOO under-reporting first, then revisit timed-reports inference with the solver already available.

## What's hard

- **Convergence on near-critical (`R ≈ 1`) cases**: the recursion's iteration map has spectral radius close to 1, so Picard converges slowly. Anderson acceleration helps but adds parameters to tune.
- **AD differentiability**: the iteration is non-smooth in `R` near critical; ImplicitDifferentiation through the fixed point handles smooth perturbations but assumes a unique fixed point. Worth a sanity-check that the iteration is contractive across the parameter range we sample over.
- **Numerical floor**: at low `ρ` and small `τ`, `η(τ)` can be very close to 1 and the recursion's signal-to-noise drops. Probably need a `log1p`-style reformulation for stability.

## Scope of this PR

This PR is the plan, not the implementation. The implementation lands in a follow-up. The current `throw(...)` stub stays as the user-facing signal until then.

Relates to #33.
