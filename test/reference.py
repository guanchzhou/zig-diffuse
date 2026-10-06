"""Independent reference outputs for zig-diffuse, written to test/golden.json.

Everything here is dense NumPy and SciPy and shares no code with the Zig implementation. Where
possible a different method is used:
- expm: scipy.linalg.expm (scaling and squaring with Pade approximants), not a Taylor series;
- heat: eigendecomposition of the symmetric Laplacian, exp(-tL) = V exp(-t Lambda) V^T;
- advect: scipy.linalg.expm of the dense combined operator;
- manifold: brute-force cosine k-nearest neighbours and numpy.linalg.solve.
Graph conventions follow src/graph.zig: undirected weights take the largest weight of any edge
between two nodes, self loops are dropped, isolated nodes have a zero Laplacian row.

Run: uv run --with numpy --with scipy python test/reference.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

RNG = np.random.default_rng(20261006)
OUT = Path(__file__).resolve().parent / "golden.json"
CASES = 40


def random_edges(n):
    edges = []
    m = int(RNG.integers(0, 3 * n + 1))
    for _ in range(m):
        i, j = (int(x) for x in RNG.integers(0, n, size=2))
        w = float(RNG.choice([1.0, round(float(RNG.uniform(0.05, 3)), 6)]))
        edges.append([i, j, w])
    return edges


def symmetric(n, edges):
    w = np.zeros((n, n))
    for i, j, x in edges:
        if i != j:
            w[i, j] = max(w[i, j], x)
            w[j, i] = max(w[j, i], x)
    return w


def laplacian(w):
    d = w.sum(axis=1)
    inv = np.where(d > 0, 1 / np.sqrt(np.where(d > 0, d, 1)), 0)
    s = inv[:, None] * w * inv[None, :]
    return np.diag((d > 0).astype(float)) - s, s


def advection(n, edges):
    a = np.zeros((n, n))
    for i, j, x in edges:
        if i != j:
            a[i, j] = max(a[i, j], x)
    return np.diag(a.sum(axis=1)) - a.T


def knn(vectors, k):
    n = len(vectors)
    norms = np.linalg.norm(vectors, axis=1)
    edges = []
    for i in range(n):
        if norms[i] == 0:
            continue
        cands = []
        for j in range(n):
            if j == i or norms[j] == 0:
                continue
            c = float(vectors[i] @ vectors[j] / (norms[i] * norms[j]))
            if c > 0:
                cands.append((-c, j, c))
        cands.sort()
        edges += [[i, j, c] for _, j, c in cands[:k]]
    return edges


def blend(s, f, beta):
    lo, hi = s.min(), s.max()
    flo, fhi = f.min(), f.max()
    scaled = lo + (f - flo) / (fhi - flo) * (hi - lo) if fhi > flo else np.full_like(f, lo)
    return (1 - beta) * s + beta * scaled


def case_expm():
    n = int(RNG.integers(1, 26))
    a = np.zeros((n, n))
    mask = RNG.random((n, n)) < RNG.uniform(0.05, 0.6)
    a[mask] = RNG.normal(size=mask.sum())
    a *= RNG.choice([0.1, 1.0, 5.0])
    t = float(RNG.choice([-2.0, -0.5, 0.25, 1.0, 3.0]))
    b = RNG.normal(size=n)
    entries = [[int(i), int(j), float(a[i, j])] for i, j in zip(*np.nonzero(a))]
    return {"n": n, "entries": entries, "t": t, "b": b.tolist(), "x": (expm(t * a) @ b).tolist()}


def case_heat():
    n = int(RNG.integers(1, 31))
    edges = random_edges(n)
    lap, _ = laplacian(symmetric(n, edges))
    vals, vecs = np.linalg.eigh(lap)
    s = RNG.uniform(0, 1, size=n)
    t = float(RNG.choice([0.5, 1.0, 2.0, 4.0]))
    beta = float(RNG.choice([0.1, 0.3, 0.5]))
    diffused = vecs @ (np.exp(-t * vals) * (vecs.T @ s))
    out = (1 - beta) * s + beta * diffused
    return {"n": n, "edges": edges, "s": s.tolist(), "t": t, "beta": beta, "out": out.tolist()}


def case_advect():
    n = int(RNG.integers(1, 31))
    edges = random_edges(n)
    lap, _ = laplacian(symmetric(n, edges))
    gamma = float(RNG.choice([0.25, 0.5, 0.75]))
    op = gamma * advection(n, edges) + (1 - gamma) * lap
    s = RNG.uniform(0, 1, size=n)
    t = float(RNG.choice([0.5, 1.0, 2.0]))
    beta = float(RNG.choice([0.1, 0.3]))
    out = (1 - beta) * s + beta * (expm(-t * op) @ s)
    return {"n": n, "edges": edges, "s": s.tolist(), "t": t, "gamma": gamma, "beta": beta, "out": out.tolist()}


def case_manifold():
    n = int(RNG.integers(2, 41))
    d = int(RNG.integers(2, 9))
    vectors = RNG.normal(size=(n, d))
    if n > 3 and RNG.random() < 0.3:
        vectors[int(RNG.integers(0, n))] = 0
    k = int(RNG.choice([1, 3, 10, 20]))
    edges = knn(vectors, k)
    _, s_mat = laplacian(symmetric(n, edges))
    a = float(RNG.choice([0.5, 0.8, 0.95]))
    y = np.where(RNG.random(n) < 0.5, RNG.uniform(0, 1, size=n), 0.0)
    f = np.linalg.solve(np.eye(n) - a * s_mat, y)
    s = RNG.uniform(0, 1, size=n)
    beta = float(RNG.choice([0.3, 0.5]))
    return {
        "vectors": vectors.ravel().tolist(),
        "d": d,
        "k": k,
        "edges": edges,
        "a": a,
        "y": y.tolist(),
        "f": f.tolist(),
        "s": s.tolist(),
        "beta": beta,
        "blended": blend(s, f, beta).tolist(),
    }


def main():
    golden = {
        "expm": [case_expm() for _ in range(CASES)],
        "heat": [case_heat() for _ in range(CASES)],
        "advect": [case_advect() for _ in range(CASES)],
        "manifold": [case_manifold() for _ in range(CASES)],
    }
    OUT.write_text(json.dumps(golden, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({sum(len(v) for v in golden.values())} cases)")


if __name__ == "__main__":
    main()
