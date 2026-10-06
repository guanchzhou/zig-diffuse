//! Re-ranking by diffusion over a graph.
//!
//! - `heat`: s' = (1 - beta) s + beta exp(-t L) s for any operator L: the symmetric normalised
//!   Laplacian gives the heat kernel (Kondor and Lafferty 2002; Chung 2007), and
//!   gamma L_adv + (1 - gamma) L_sym gives advection-diffusion.
//! - `manifold`: F = (I - a S)^(-1) y (Zhou et al. 2004), solved by conjugate gradients.
//! - `blend`: (1 - beta) s + beta F with F rescaled to the range of s.

const std = @import("std");
const sparse = @import("sparse.zig");
const expm = @import("expm.zig");
const Csr = sparse.Csr;

pub fn heat(gpa: std.mem.Allocator, l: Csr, t: f64, beta: f64, s: []const f64, out: []f64) expm.Error!expm.Stats {
    const diffused = try gpa.alloc(f64, s.len);
    defer gpa.free(diffused);
    const st = try expm.action(gpa, l, -t, s, diffused);
    for (out, s, diffused) |*o, x, y| o.* = (1 - beta) * x + beta * y;
    return st;
}

pub const ManifoldOptions = struct {
    /// Stop when ||y - (I - aS) F||_2 <= tol * ||y||_2.
    tol: f64 = 1e-12,
    max_iterations: usize = 10_000,
};

/// `s` must be symmetric with spectral radius at most 1, as `graph.normalizedAdjacency` is, and
/// 0 <= a < 1, so that I - aS is positive definite. Returns the number of iterations.
pub fn manifold(gpa: std.mem.Allocator, s: Csr, a: f64, y: []const f64, out: []f64, opts: ManifoldOptions) !usize {
    std.debug.assert(a >= 0 and a < 1 and y.len == s.n and out.len == s.n);
    const n = s.n;
    @memset(out, 0);
    const r = try gpa.dupe(f64, y);
    defer gpa.free(r);
    const p = try gpa.dupe(f64, y);
    defer gpa.free(p);
    const ap = try gpa.alloc(f64, n);
    defer gpa.free(ap);

    var rr = dot(r, r);
    const stop = opts.tol * opts.tol * rr;
    if (rr == 0) return 0;
    for (1..opts.max_iterations + 1) |it| {
        s.mul(p, ap);
        for (ap, p) |*v, pv| v.* = pv - a * v.*;
        const alpha = rr / dot(p, ap);
        for (out, p) |*x, pv| x.* += alpha * pv;
        for (r, ap) |*rv, v| rv.* -= alpha * v;
        const next = dot(r, r);
        if (next <= stop) return it;
        const beta = next / rr;
        rr = next;
        for (p, r) |*pv, rv| pv.* = rv + beta * pv.*;
    }
    return error.NoConvergence;
}

fn dot(x: []const f64, y: []const f64) f64 {
    var sum: f64 = 0;
    for (x, y) |a, b| sum += a * b;
    return sum;
}

/// out = (1 - beta) s + beta F', where F' maps F linearly onto [min s, max s]. A constant F maps
/// to min s.
pub fn blend(s: []const f64, f: []const f64, beta: f64, out: []f64) void {
    std.debug.assert(s.len == f.len and out.len == s.len);
    if (s.len == 0) return;
    const s_lo = std.mem.min(f64, s);
    const s_hi = std.mem.max(f64, s);
    const f_lo = std.mem.min(f64, f);
    const f_hi = std.mem.max(f64, f);
    for (out, s, f) |*o, x, v| {
        const scaled = if (f_hi > f_lo) s_lo + (v - f_lo) / (f_hi - f_lo) * (s_hi - s_lo) else s_lo;
        o.* = (1 - beta) * x + beta * scaled;
    }
}

test "manifold ranking solves the linear system" {
    const gpa = std.testing.allocator;
    const graph = @import("graph.zig");
    const w = try graph.symmetric(gpa, 3, &.{ .{ .from = 0, .to = 1 }, .{ .from = 1, .to = 2 } });
    defer w.deinit(gpa);
    const s = try graph.normalizedAdjacency(gpa, w);
    defer s.deinit(gpa);
    const y = [_]f64{ 1, 0, 0 };
    var f: [3]f64 = undefined;
    _ = try manifold(gpa, s, 0.9, &y, &f, .{});
    var sf: [3]f64 = undefined;
    s.mul(&f, &sf);
    for (f, sf, y) |fv, sv, yv| try std.testing.expectApproxEqAbs(yv, fv - 0.9 * sv, 1e-12);
    // numpy.linalg.solve on the dense system.
    for ([_]f64{ 3.13157895, 3.34945317, 2.13157895 }, f) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-8);
}

test "blend rescales onto the range of s" {
    var out: [3]f64 = undefined;
    blend(&.{ 0.2, 0.4, 0.6 }, &.{ 10, 0, 5 }, 1, &out);
    try std.testing.expectEqualSlices(f64, &.{ 0.6, 0.2, 0.4 }, &out);
}
