# zig-diffuse

Re-ranking by diffusion on sparse graphs, in pure Zig 0.17. Scores from a
retriever spread along the links between notes or along a nearest-neighbour
graph, so a note that the query matches poorly can gain from neighbours that
it matches well.

- **Heat diffusion**: (1 - beta) s + beta exp(-tL) s with the symmetric
  normalised Laplacian.
- **Advection-diffusion**: the same with gamma L_adv + (1 - gamma) L_sym, where
  the advection operator moves score along link directions and conserves
  its total.
- **Manifold ranking**: F = (I - aS)^(-1) y on a cosine k-nearest-neighbour
  graph, solved by conjugate gradients.
- **Matrix-exponential action**: exp(tA) b for any sparse A without forming
  exp(tA), by the method of Al-Mohy and Higham.
- No dependencies and no C. Checked against independent dense NumPy and
  SciPy computations. MIT licensed.

## Install

```sh
zig fetch --save git+https://github.com/guanchzhou/zig-diffuse
```

```zig
const diffuse = b.dependency("zig_diffuse", .{}).module("diffuse");
exe.root_module.addImport("diffuse", diffuse);
```

## Library

| Module | What it does | Reference |
| --- | --- | --- |
| `sparse` | Square CSR matrices: build from entries, multiply, combine | |
| `graph` | Symmetric weights, normalised adjacency and Laplacian, advection, k-nearest neighbours | Chung 1997; Chapman and Mesbahi 2011 |
| `expm` | exp(tA) b by truncated Taylor series with scaling | Al-Mohy and Higham 2011 |
| `rank` | Heat diffusion, advection-diffusion, manifold ranking, blending | Kondor and Lafferty 2002; Chung 2007; Zhou et al. 2004 |

Conventions:

- Undirected weights take the largest weight of any edge between two nodes,
  in either direction. Self loops are dropped.
- Isolated nodes have a zero Laplacian row, so diffusion leaves their
  scores unchanged.
- The advection operator is L_adv = D_out - A^T for edges i -> j. Under
  dx/dt = -L_adv x a node sends its score along its outgoing edges, and the
  sum of all scores stays the same.
- `knn` keeps only positive cosine similarities. Ties go to the lower index.
- `expm` controls the error relative to the norm of the result, as SciPy's
  `expm_multiply` does. A small entry next to a large one can have a larger
  relative error.

```zig
const w = try diffuse.graph.symmetric(gpa, n, links);
defer w.deinit(gpa);
const l = try diffuse.graph.laplacian(gpa, w);
defer l.deinit(gpa);
_ = try diffuse.rank.heat(gpa, l, 2, 0.5, scores, out);
```

[`examples/notes.zig`](examples/notes.zig) runs heat diffusion and pure
advection on six linked notes. It shows the mechanics: score moves from a
strongly matched note to its neighbours, and advection keeps the total. Whether
that improves retrieval has to be measured on real queries.

## Command line

`zig build` installs `zig-out/bin/zig-diffuse`. Each command reads one JSON
object on stdin and writes one JSON object on stdout. Edges are `[from, to]`
or `[from, to, weight]`.

```sh
$ echo '{"n":3,"edges":[[0,1],[1,2]],"scores":[1,0,0],"t":1,"gamma":1,"beta":1}' | zig-diffuse advect
{"scores":[0.3678794411714423,0.3678794411714422,0.26424111765711544],"m":20,"s":1,"matvecs":17}
```

That is the exact solution e^-1, t e^-t and 1 - 2e^-1 for a unit flow along
the path 0 -> 1 -> 2. `zig-diffuse --help` lists every command. Invalid input
exits with 1; an unknown or missing command exits with 2.

## Development

```sh
zig build test              # unit, golden, fuzz-corpus, example, and command-line tests
zig build test-cli          # command-line tests only
zig build test --fuzz       # fuzz the invariants (needs .zig-cache/tmp to exist)
zig build docs              # API documentation in zig-out/docs
uv run --with numpy --with scipy python test/reference.py   # regenerate test/golden.json
```

`test/reference.py` uses a different method from the library wherever it
can:

- heat diffusion by eigendecomposition;
- advection-diffusion and general matrices by SciPy's Padé `expm`;
- manifold ranking by a dense direct solve over brute-force neighbours.

`test/golden.zig` replays 160 cases and requires
||got - want||_inf <= 1e-10 max(1, ||want||_inf), with neighbour lists equal.
On a 2,000-node graph with 6,000 links, heat diffusion agrees with SciPy's
`expm_multiply` to 3e-16.

The fuzz targets check properties on random graphs:

- advection conserves the total;
- heat diffusion conserves the sum of sqrt(degree) times score;
- diffusion keeps non-negative scores non-negative;
- exp((t1 + t2) A) b = exp(t1 A) exp(t2 A) b;
- nearest-neighbour graphs have no self loops and at most k edges per node;
- manifold scores never fall below their seeds.

The first `--fuzz` run on a fresh cache can abort while the fuzzer sets up
its coverage files; a second run starts normally.

## Credits

- A. H. Al-Mohy and N. J. Higham, "Computing the action of the matrix
  exponential, with an application to exponential integrators", *SIAM J.
  Sci. Comput.* 33(2), 488-511 (2011).
- R. I. Kondor and J. D. Lafferty, "Diffusion kernels on graphs and other
  discrete input spaces", *ICML* (2002).
- F. Chung, *Spectral Graph Theory*, AMS (1997), and "The heat kernel as the
  pagerank of a graph", *PNAS* 104(50), 19735-19740 (2007).
- D. Zhou, J. Weston, A. Gretton, O. Bousquet and B. Schölkopf, "Ranking on
  data manifolds", *NIPS* 16 (2004).
- A. Chapman and M. Mesbahi, "Advection on graphs", *50th IEEE CDC-ECC*,
  1461-1466 (2011).

## License

MIT. See [LICENSE](LICENSE).
