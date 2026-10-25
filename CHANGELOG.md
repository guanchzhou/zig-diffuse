# Changelog

## 0.1.2 - 2026-10-25

- Added `sparse.Csr.validate`, which checks a matrix's structure in $O(n + \text{nnz})$. `Csr.mul`, `Csr.trace`, `Csr.entries`, `sparse.combine`, `graph.degrees`, `graph.normalizedAdjacency`, `graph.laplacian`, `expm.action`, `rank.heat` and `rank.manifold` call it once and return `error.InvalidCsr` for a malformed caller-built `Csr`; `expm.action` and `rank.manifold` validate once, not on every product. A standalone `Csr.mul` now costs about twice as much as before.
- `Csr.mul`, `Csr.trace`, `graph.degrees` and `rank.blend` now return error unions; callers need `try`.
- Public entry points return errors instead of asserting on caller input: `error.NodeOutOfRange` and `error.InvalidWeight` (negative or NaN) for edges and entries, `error.TooManyNonzeros` and `error.TooManyNodes` beyond the `u32` index range, `error.DimensionMismatch` in `sparse.combine`, `error.LengthMismatch` for mismatched vectors and a zero or non-dividing `graph.knn` dimension, and `error.InvalidParameter` for `a` outside [0, 1) in `rank.manifold`.
- `rank.manifold` returns `error.NotFinite` as soon as the residual or step size is infinite or NaN instead of running to `max_iterations`.
- `graph.knn` keeps each node's top k in a size-k heap instead of sorting every candidate, with identical output.
- CLI `stdin` reads are capped at 256 MiB.

## 0.1.1 - 2026-10-20

- Linear $O(n + \text{nnz})$ sorted row merge in `sparse.combine` and direct CSR construction in `graph.normalizedAdjacency` and `graph.laplacian`.
- Vectorised dot products in `graph.knn` and `rank`, fused inner Taylor loop in `expm.action`, and guarded `graph.knn` against subnormal norm product underflow.
- Bounded CLI `stdin` reads and added GitHub Actions CI workflow.

## 0.1.0 - 2026-10-06

- First release: `sparse`, `graph`, `expm` and `rank`, checked against
  independent dense NumPy and SciPy computations on 160 cases.
- The `zig-diffuse` command runs heat diffusion, advection-diffusion,
  k-nearest neighbours, manifold ranking and the matrix-exponential action
  on JSON from stdin.
- Fuzz targets for conservation, positivity, the semigroup property and the
  neighbour-graph shape.
