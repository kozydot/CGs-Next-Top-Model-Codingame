## what this does

* reads a training line and a test line (x y pairs) from stdin.
* fits four model families to the training data by minimizing sum-of-squared errors (sse).
* evaluates each fitted model on the test set and prints the model with lowest test sse.

## main ideas

* we search for best parameters using randomized sampling + short local hill-climb.
* keep a tiny top-k of best random seeds, then refine them with small perturbations.
* use early cutoff during sse evaluation to reject bad candidates fast.
* deterministic fast rng seeded from input so runs are reproducible.

## code structure

* parsing: fast split-and-parse into vec of pairs.
* data container: store xs, ys and precompute x^2 to speed up parabolic evals.
* sse evaluators: inline, simple loops with early cutoff to abort when candidate is already worse than current best.
* optimizer: sample a modest number of random parameter vectors, keep a tiny sorted top-k, then do 3–5 refinement rounds with shrinking perturbation scales.
* models:

  * linear: `a*x + b`
  * parabola: `a*x^2 + b*x + c`
  * sine: `a*sin(b*x + c) + d`
  * exponential: `abs(a)^(x+b) + c` (with guards for invalid/overflow cases)

## performance used

* precompute x^2 and store arrays to avoid tuple indexing overhead.
* xorshift-style fast rng and reusing buffers to avoid frequent allocations.
* early sse cutoff: quit evaluating a candidate as soon as partial sse exceeds cutoff.
* small top-k (6) so we only refine a few seeds.
* tuned sample counts per model (sine needs more samples than linear, etc.).

## pitfalls handled

* exponential overflows: we treat huge powf results as invalid and reject candidate.
* zero-base with negative exponents: flagged invalid and rejected.
* nan / inf checks everywhere to avoid silent failures.
