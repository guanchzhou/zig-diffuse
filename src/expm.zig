//! The action of the matrix exponential, exp(tA) b, without forming exp(tA) (Al-Mohy and
//! Higham 2011, Algorithm 3.2). A is shifted by mu = trace(A) / n, and the truncated Taylor
//! series of exp(tA/s) is applied s times, stopping each series early once two consecutive
//! terms are negligible.
//!
//! The degree m and the scaling s minimise m * s subject to s >= ||t(A - mu I)||_1 / theta_m,
//! with theta_m the double-precision table of the paper. The exact 1-norm is used throughout;
//! the paper's sharper estimate for large norms can save matrix-vector products but never
//! changes the error bound.

const std = @import("std");
const Csr = @import("sparse.zig").Csr;

const theta = [_]struct { m: u32, t: f64 }{
    .{ .m = 1, .t = 2.29e-16 }, .{ .m = 2, .t = 2.58e-8 },  .{ .m = 3, .t = 1.39e-5 },  .{ .m = 4, .t = 3.40e-4 },
    .{ .m = 5, .t = 2.40e-3 },  .{ .m = 6, .t = 9.07e-3 },  .{ .m = 7, .t = 2.38e-2 },  .{ .m = 8, .t = 5.00e-2 },
    .{ .m = 9, .t = 8.96e-2 },  .{ .m = 10, .t = 1.44e-1 }, .{ .m = 11, .t = 2.14e-1 }, .{ .m = 12, .t = 3.00e-1 },
    .{ .m = 13, .t = 4.00e-1 }, .{ .m = 14, .t = 5.14e-1 }, .{ .m = 15, .t = 6.41e-1 }, .{ .m = 16, .t = 7.81e-1 },
    .{ .m = 17, .t = 9.31e-1 }, .{ .m = 18, .t = 1.09 },    .{ .m = 19, .t = 1.26 },    .{ .m = 20, .t = 1.44 },
    .{ .m = 21, .t = 1.62 },    .{ .m = 22, .t = 1.82 },    .{ .m = 23, .t = 2.01 },    .{ .m = 24, .t = 2.22 },
    .{ .m = 25, .t = 2.43 },    .{ .m = 26, .t = 2.64 },    .{ .m = 27, .t = 2.86 },    .{ .m = 28, .t = 3.08 },
    .{ .m = 29, .t = 3.31 },    .{ .m = 30, .t = 3.54 },    .{ .m = 35, .t = 4.7 },     .{ .m = 40, .t = 6.0 },
    .{ .m = 45, .t = 7.2 },     .{ .m = 50, .t = 8.5 },     .{ .m = 55, .t = 9.9 },
};

const tol = std.math.pow(f64, 2, -53);

/// Matrix-vector products are bounded by m * s, so a norm this large is rejected rather than run.
const max_steps = 1 << 24;

pub const Stats = struct { m: u32, s: u64, matvecs: u64 };

pub const Error = error{ OutOfMemory, NormTooLarge };

fn infNorm(x: []const f64) f64 {
    var m: f64 = 0;
    for (x) |v| m = @max(m, @abs(v));
    return m;
}

/// ||A - mu I||_1, the largest absolute column sum.
fn shiftedNorm1(gpa: std.mem.Allocator, a: Csr, mu: f64) !f64 {
    const col = try gpa.alloc(f64, a.n);
    defer gpa.free(col);
    const diag = try gpa.alloc(f64, a.n);
    defer gpa.free(diag);
    @memset(col, 0);
    @memset(diag, 0);
    for (0..a.n) |r| {
        for (a.row_ptr[r]..a.row_ptr[r + 1]) |k| {
            if (a.col[k] == r) diag[r] += a.val[k] else col[a.col[k]] += @abs(a.val[k]);
        }
    }
    var norm: f64 = 0;
    for (col, diag) |c, d| norm = @max(norm, c + @abs(d - mu));
    return norm;
}

/// out = exp(tA) b. `out` may not alias `b`.
pub fn action(gpa: std.mem.Allocator, a: Csr, t: f64, b: []const f64, out: []f64) Error!Stats {
    const n = a.n;
    std.debug.assert(b.len == n and out.len == n);
    @memcpy(out, b);
    if (n == 0) return .{ .m = 0, .s = 1, .matvecs = 0 };
    const mu = a.trace() / @as(f64, @floatFromInt(n));
    const norm = @abs(t) * try shiftedNorm1(gpa, a, mu);
    if (norm == 0) {
        const e = @exp(t * mu);
        for (out) |*v| v.* *= e;
        return .{ .m = 0, .s = 1, .matvecs = 0 };
    }

    var best_m: u32 = 0;
    var best_s: u64 = 0;
    for (theta) |th| {
        const sf = @ceil(norm / th.t);
        if (!(sf * @as(f64, @floatFromInt(th.m)) <= max_steps)) continue;
        const s: u64 = @intFromFloat(sf);
        if (best_m == 0 or th.m * s < best_m * best_s) {
            best_m = th.m;
            best_s = s;
        }
    }
    if (best_m == 0) return error.NormTooLarge;

    const term = try gpa.alloc(f64, n);
    defer gpa.free(term);
    const tmp = try gpa.alloc(f64, n);
    defer gpa.free(tmp);
    @memcpy(term, b);
    const sf: f64 = @floatFromInt(best_s);
    const eta = @exp(t * mu / sf);
    var matvecs: u64 = 0;
    for (0..best_s) |_| {
        var c1 = infNorm(term);
        for (0..best_m) |j| {
            a.mul(term, tmp);
            matvecs += 1;
            const coeff = t / (sf * @as(f64, @floatFromInt(j + 1)));
            var c2: f64 = 0;
            var out_norm: f64 = 0;
            for (term, tmp, out) |*v, av, *f| {
                const nv = coeff * (av - mu * v.*);
                v.* = nv;
                c2 = @max(c2, @abs(nv));
                const nf = f.* + nv;
                f.* = nf;
                out_norm = @max(out_norm, @abs(nf));
            }
            if (c1 + c2 <= tol * out_norm) break;
            c1 = c2;
        }
        for (out, term) |*f, *v| {
            const scaled = f.* * eta;
            f.* = scaled;
            v.* = scaled;
        }
    }
    return .{ .m = best_m, .s = best_s, .matvecs = matvecs };
}

test "diagonal matrix" {
    const gpa = std.testing.allocator;
    const sparse = @import("sparse.zig");
    const a = try sparse.fromEntries(gpa, 3, &.{
        .{ .row = 0, .col = 0, .val = -1 },
        .{ .row = 1, .col = 1, .val = 2 },
        .{ .row = 2, .col = 2, .val = 0.5 },
    }, .sum);
    defer a.deinit(gpa);
    var out: [3]f64 = undefined;
    _ = try action(gpa, a, 3, &.{ 1, 1, 2 }, &out);
    // The error bound is relative to the norm of the result, not to each entry.
    const want = [_]f64{ @exp(-3.0), @exp(6.0), 2 * @exp(1.5) };
    for (want, out) |w, o| try std.testing.expectApproxEqAbs(w, o, 1e-14 * @exp(6.0));
}

test "rotation generator" {
    const gpa = std.testing.allocator;
    const sparse = @import("sparse.zig");
    const a = try sparse.fromEntries(gpa, 2, &.{ .{ .row = 0, .col = 1, .val = -1 }, .{ .row = 1, .col = 0, .val = 1 } }, .sum);
    defer a.deinit(gpa);
    var out: [2]f64 = undefined;
    const st = try action(gpa, a, 30, &.{ 1, 0 }, &out);
    try std.testing.expect(st.s > 1);
    try std.testing.expectApproxEqAbs(@cos(30.0), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@sin(30.0), out[1], 1e-12);
}

test "zero matrix and t = 0" {
    const gpa = std.testing.allocator;
    const sparse = @import("sparse.zig");
    const a = try sparse.fromEntries(gpa, 2, &.{}, .sum);
    defer a.deinit(gpa);
    var out: [2]f64 = undefined;
    const st = try action(gpa, a, 5, &.{ 3, 4 }, &out);
    try std.testing.expectEqual(@as(u64, 0), st.matvecs);
    try std.testing.expectEqualSlices(f64, &.{ 3, 4 }, &out);
}
