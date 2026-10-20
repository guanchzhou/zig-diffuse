//! Graph operators built from edge lists.
//!
//! - `symmetric`: undirected weights W, w_ij = w_ji = the largest weight of any edge between i
//!   and j in either direction. Self loops are dropped.
//! - `normalizedAdjacency`: S = D^(-1/2) W D^(-1/2), with D the weighted degrees.
//! - `laplacian`: the symmetric normalised Laplacian L = I - S on nodes with a positive degree,
//!   and 0 on isolated nodes (Chung's convention), so diffusion leaves an isolated node alone.
//! - `advection`: L_adv = D_out - A^T for directed edges i -> j with weights a_ij (Chapman and
//!   Mesbahi 2011). Under dx/dt = -L_adv x each node sends mass along its outgoing edges in
//!   proportion to their weights, and the total sum of x is conserved.
//! - `knn`: a k-nearest-neighbour graph over vectors by cosine similarity.

const std = @import("std");
const sparse = @import("sparse.zig");
const Csr = sparse.Csr;
const Entry = sparse.Entry;

pub const Edge = struct { from: u32, to: u32, weight: f64 = 1 };

pub fn symmetric(gpa: std.mem.Allocator, n: usize, edges: []const Edge) !Csr {
    var list: std.ArrayList(Entry) = .empty;
    defer list.deinit(gpa);
    for (edges) |e| {
        std.debug.assert(e.from < n and e.to < n and e.weight >= 0);
        if (e.from == e.to) continue;
        try list.append(gpa, .{ .row = e.from, .col = e.to, .val = e.weight });
        try list.append(gpa, .{ .row = e.to, .col = e.from, .val = e.weight });
    }
    return sparse.fromEntries(gpa, n, list.items, .max);
}

/// Row sums of W.
pub fn degrees(w: Csr, out: []f64) void {
    std.debug.assert(out.len == w.n);
    for (0..w.n) |r| {
        var d: f64 = 0;
        for (w.row_ptr[r]..w.row_ptr[r + 1]) |k| d += w.val[k];
        out[r] = d;
    }
}

pub fn normalizedAdjacency(gpa: std.mem.Allocator, w: Csr) !Csr {
    const d = try gpa.alloc(f64, w.n);
    defer gpa.free(d);
    degrees(w, d);

    const row_ptr = try gpa.dupe(u32, w.row_ptr);
    errdefer gpa.free(row_ptr);
    const col = try gpa.dupe(u32, w.col);
    errdefer gpa.free(col);
    const val = try gpa.alloc(f64, w.val.len);

    for (0..w.n) |r| {
        const dr = d[r];
        for (w.row_ptr[r]..w.row_ptr[r + 1]) |k| {
            const dd = dr * d[w.col[k]];
            val[k] = if (dd > 0) w.val[k] / @sqrt(dd) else 0;
        }
    }
    return .{ .n = w.n, .row_ptr = row_ptr, .col = col, .val = val };
}

pub fn laplacian(gpa: std.mem.Allocator, w: Csr) !Csr {
    const n = w.n;
    const d = try gpa.alloc(f64, n);
    defer gpa.free(d);
    degrees(w, d);

    const row_ptr = try gpa.alloc(u32, n + 1);
    errdefer gpa.free(row_ptr);
    row_ptr[0] = 0;

    var total_nnz: usize = 0;
    for (0..n) |r| {
        const start = w.row_ptr[r];
        const end = w.row_ptr[r + 1];
        var has_diag = false;
        for (start..end) |k| {
            if (w.col[k] == r) {
                has_diag = true;
                break;
            }
        }
        total_nnz += (end - start) + @intFromBool(d[r] > 0 and !has_diag);
        row_ptr[r + 1] = @intCast(total_nnz);
    }

    const col = try gpa.alloc(u32, total_nnz);
    errdefer gpa.free(col);
    const val = try gpa.alloc(f64, total_nnz);

    var out_k: usize = 0;
    for (0..n) |r| {
        const dr = d[r];
        const r_u32: u32 = @intCast(r);
        var inserted_diag = !(dr > 0);
        for (w.row_ptr[r]..w.row_ptr[r + 1]) |k| {
            const c = w.col[k];
            if (!inserted_diag and c > r_u32) {
                col[out_k] = r_u32;
                val[out_k] = 1.0;
                out_k += 1;
                inserted_diag = true;
            }
            const dd = dr * d[c];
            const s_val = if (dd > 0) w.val[k] / @sqrt(dd) else 0;
            col[out_k] = c;
            if (c == r_u32) {
                val[out_k] = (if (dr > 0) @as(f64, 1.0) else @as(f64, 0.0)) - s_val;
                inserted_diag = true;
            } else {
                val[out_k] = -s_val;
            }
            out_k += 1;
        }
        if (!inserted_diag) {
            col[out_k] = r_u32;
            val[out_k] = 1.0;
            out_k += 1;
        }
    }
    return .{ .n = n, .row_ptr = row_ptr, .col = col, .val = val };
}

/// Repeated edges i -> j keep the largest weight. Self loops are dropped.
pub fn advection(gpa: std.mem.Allocator, n: usize, edges: []const Edge) !Csr {
    var direct: std.ArrayList(Entry) = .empty;
    defer direct.deinit(gpa);
    for (edges) |e| {
        std.debug.assert(e.from < n and e.to < n and e.weight >= 0);
        if (e.from != e.to) try direct.append(gpa, .{ .row = e.from, .col = e.to, .val = e.weight });
    }
    const a = try sparse.fromEntries(gpa, n, direct.items, .max);
    defer a.deinit(gpa);
    var list: std.ArrayList(Entry) = .empty;
    defer list.deinit(gpa);
    for (0..n) |i| {
        for (a.row_ptr[i]..a.row_ptr[i + 1]) |k| {
            try list.append(gpa, .{ .row = @intCast(i), .col = @intCast(i), .val = a.val[k] });
            try list.append(gpa, .{ .row = a.col[k], .col = @intCast(i), .val = -a.val[k] });
        }
    }
    return sparse.fromEntries(gpa, n, list.items, .sum);
}

fn vecDot(a: []const f64, b: []const f64) f64 {
    const lanes = 4;
    const Vec = @Vector(lanes, f64);
    var acc: Vec = @splat(0.0);
    var i: usize = 0;
    while (i + lanes <= a.len) : (i += lanes) {
        const va: Vec = a[i..][0..lanes].*;
        const vb: Vec = b[i..][0..lanes].*;
        acc = @mulAdd(Vec, va, vb, acc);
    }
    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// For each of the n vectors (row-major, d values each), an edge to each of its k most
/// cosine-similar other vectors with a positive similarity, weighted by that similarity. Ties go
/// to the lower index. Zero vectors get no edges. Pass the result to `symmetric` for the
/// undirected graph. Caller owns the returned slice.
pub fn knn(gpa: std.mem.Allocator, vectors: []const f64, d: usize, k: usize) ![]Edge {
    std.debug.assert(d > 0 and vectors.len % d == 0);
    const n = vectors.len / d;
    if (n == 0 or k == 0) return gpa.alloc(Edge, 0);
    const norms = try gpa.alloc(f64, n);
    defer gpa.free(norms);
    for (0..n) |i| {
        const row = vectors[i * d ..][0..d];
        norms[i] = @sqrt(vecDot(row, row));
    }
    const Cand = struct { j: u32, cos: f64 };
    const cands = try gpa.alloc(Cand, n);
    defer gpa.free(cands);
    var out: std.ArrayList(Edge) = .empty;
    errdefer out.deinit(gpa);
    for (0..n) |i| {
        if (norms[i] == 0) continue;
        const row_i = vectors[i * d ..][0..d];
        var m: usize = 0;
        for (0..n) |j| {
            if (j == i or norms[j] == 0) continue;
            const denom = norms[i] * norms[j];
            if (denom <= 0) continue;
            const dot = vecDot(row_i, vectors[j * d ..][0..d]);
            const c = dot / denom;
            if (c > 0 and std.math.isFinite(c)) {
                cands[m] = .{ .j = @intCast(j), .cos = c };
                m += 1;
            }
        }
        std.mem.sort(Cand, cands[0..m], {}, struct {
            fn f(_: void, x: Cand, y: Cand) bool {
                return x.cos > y.cos or (x.cos == y.cos and x.j < y.j);
            }
        }.f);
        for (cands[0..@min(k, m)]) |c| try out.append(gpa, .{ .from = @intCast(i), .to = c.j, .weight = c.cos });
    }
    return out.toOwnedSlice(gpa);
}

test "laplacian of a path, with an isolated node" {
    const gpa = std.testing.allocator;
    const w = try symmetric(gpa, 4, &.{ .{ .from = 0, .to = 1 }, .{ .from = 1, .to = 2 }, .{ .from = 2, .to = 1, .weight = 0.5 } });
    defer w.deinit(gpa);
    const l = try laplacian(gpa, w);
    defer l.deinit(gpa);
    var y: [4]f64 = undefined;
    // The kernel of L is spanned by D^(1/2) 1 on the connected part.
    l.mul(&.{ 1, @sqrt(2.0), 1, 7 }, &y);
    for (y) |v| try std.testing.expectApproxEqAbs(@as(f64, 0), v, 1e-15);
}

test "advection conserves mass" {
    const gpa = std.testing.allocator;
    const l = try advection(gpa, 3, &.{ .{ .from = 0, .to = 1, .weight = 2 }, .{ .from = 0, .to = 2 }, .{ .from = 1, .to = 2 } });
    defer l.deinit(gpa);
    var y: [3]f64 = undefined;
    l.mul(&.{ 1, 2, 3 }, &y);
    try std.testing.expectEqualSlices(f64, &.{ 3, 0, -3 }, &y);
}

test "knn keeps the k most similar, ties to the lower index" {
    const gpa = std.testing.allocator;
    const e = try knn(gpa, &.{ 1, 0, 1, 0, 1, 0, 0, 1, 0, 0 }, 2, 1);
    defer gpa.free(e);
    try std.testing.expectEqual(@as(usize, 3), e.len);
    try std.testing.expectEqual(@as(u32, 1), e[0].to);
    try std.testing.expectEqual(@as(u32, 0), e[1].to);
    try std.testing.expectEqual(@as(u32, 0), e[2].to);
}
