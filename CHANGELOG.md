# Changelog

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
