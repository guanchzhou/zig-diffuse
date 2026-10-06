# Changelog

## 0.1.0 - 2026-10-06

- First release: `sparse`, `graph`, `expm` and `rank`, checked against
  independent dense NumPy and SciPy computations on 160 cases.
- The `zig-diffuse` command runs heat diffusion, advection-diffusion,
  k-nearest neighbours, manifold ranking and the matrix-exponential action
  on JSON from stdin.
- Fuzz targets for conservation, positivity, the semigroup property and the
  neighbour-graph shape.
