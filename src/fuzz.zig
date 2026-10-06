//! Fuzz targets checking properties that must hold for every input. `zig build test` runs every
//! input in `src/fuzz-corpus/`; `zig build test --fuzz` explores further. Copy any input that
//! finds a bug back into the corpus.

const std = @import("std");
const sparse = @import("sparse.zig");
const graph = @import("graph.zig");
const expm = @import("expm.zig");
const rank = @import("rank.zig");
const Smith = std.testing.Smith;

const corpus = [_][]const u8{
    @embedFile("fuzz-corpus/00"),
    @embedFile("fuzz-corpus/01"),
    @embedFile("fuzz-corpus/02"),
    @embedFile("fuzz-corpus/03"),
    @embedFile("fuzz-corpus/04"),
    @embedFile("fuzz-corpus/05"),
};

const max_n = 16;
const max_edges = 48;

fn upTo(smith: *Smith, lo: usize, hi: usize) usize {
    return smith.valueRangeAtMost(u32, @intCast(lo), @intCast(hi));
}

fn unitFloat(smith: *Smith) f64 {
    return @as(f64, @floatFromInt(smith.value(u16))) / 65535.0;
}

fn randomEdges(smith: *Smith, n: usize, buf: []graph.Edge) []graph.Edge {
    const m = upTo(smith, 0, buf.len);
    for (buf[0..m]) |*e| e.* = .{
        .from = @intCast(upTo(smith, 0, n - 1)),
        .to = @intCast(upTo(smith, 0, n - 1)),
        .weight = if (smith.boolWeighted(1, 1)) 1 else 0.01 + 3 * unitFloat(smith),
    };
    return buf[0..m];
}

fn sum(x: []const f64) f64 {
    var s: f64 = 0;
    for (x) |v| s += v;
    return s;
}

fn diffusion(_: void, smith: *Smith) anyerror!void {
    const gpa = std.testing.allocator;
    const n = upTo(smith, 1, max_n);
    var buf: [max_edges]graph.Edge = undefined;
    const edges = randomEdges(smith, n, &buf);
    var x: [max_n]f64 = undefined;
    for (x[0..n]) |*v| v.* = unitFloat(smith);
    const t = 4 * unitFloat(smith);
    var out: [max_n]f64 = undefined;

    const l_adv = try graph.advection(gpa, n, edges);
    defer l_adv.deinit(gpa);
    _ = try expm.action(gpa, l_adv, -t, x[0..n], out[0..n]);
    try std.testing.expectApproxEqAbs(sum(x[0..n]), sum(out[0..n]), 1e-10 * @as(f64, @floatFromInt(n)));
    for (out[0..n]) |v| try std.testing.expect(v >= -1e-12);

    const w = try graph.symmetric(gpa, n, edges);
    defer w.deinit(gpa);
    const l_sym = try graph.laplacian(gpa, w);
    defer l_sym.deinit(gpa);
    _ = try expm.action(gpa, l_sym, -t, x[0..n], out[0..n]);
    var d: [max_n]f64 = undefined;
    graph.degrees(w, d[0..n]);
    var before: f64 = 0;
    var after: f64 = 0;
    for (d[0..n], x[0..n], out[0..n]) |dv, xv, ov| {
        before += @sqrt(dv) * xv;
        after += @sqrt(dv) * ov;
    }
    try std.testing.expectApproxEqAbs(before, after, 1e-9 * @max(1, @abs(before)));
    for (out[0..n]) |v| try std.testing.expect(v >= -1e-12);

    const gamma = unitFloat(smith);
    const op = try sparse.combine(gpa, gamma, l_adv, 1 - gamma, l_sym);
    defer op.deinit(gpa);
    _ = try expm.action(gpa, op, -t, x[0..n], out[0..n]);
    for (out[0..n]) |v| try std.testing.expect(v >= -1e-12);
}

test "fuzz: diffusion conserves its invariant and keeps scores non-negative" {
    try std.testing.fuzz({}, diffusion, .{ .corpus = &corpus });
}

fn semigroup(_: void, smith: *Smith) anyerror!void {
    const gpa = std.testing.allocator;
    const n = upTo(smith, 1, max_n);
    var list: [max_edges]sparse.Entry = undefined;
    const m = upTo(smith, 0, max_edges);
    for (list[0..m]) |*e| e.* = .{
        .row = @intCast(upTo(smith, 0, n - 1)),
        .col = @intCast(upTo(smith, 0, n - 1)),
        .val = 2 * unitFloat(smith) - 1,
    };
    const a = try sparse.fromEntries(gpa, n, list[0..m], .sum);
    defer a.deinit(gpa);
    var b: [max_n]f64 = undefined;
    for (b[0..n]) |*v| v.* = 2 * unitFloat(smith) - 1;
    const t1 = 2 * unitFloat(smith);
    const t2 = 2 * unitFloat(smith);
    var once: [max_n]f64 = undefined;
    var half: [max_n]f64 = undefined;
    var twice: [max_n]f64 = undefined;
    _ = try expm.action(gpa, a, t1 + t2, b[0..n], once[0..n]);
    _ = try expm.action(gpa, a, t2, b[0..n], half[0..n]);
    _ = try expm.action(gpa, a, t1, half[0..n], twice[0..n]);
    var scale: f64 = 1;
    for (once[0..n]) |v| scale = @max(scale, @abs(v));
    for (once[0..n], twice[0..n]) |x, y| try std.testing.expectApproxEqAbs(x, y, 1e-9 * scale);
}

test "fuzz: exp((t1 + t2) A) b = exp(t1 A) exp(t2 A) b" {
    try std.testing.fuzz({}, semigroup, .{ .corpus = &corpus });
}

fn manifoldInvariants(_: void, smith: *Smith) anyerror!void {
    const gpa = std.testing.allocator;
    const n = upTo(smith, 1, max_n);
    const d = upTo(smith, 1, 6);
    const k = upTo(smith, 1, n);
    var v: [max_n * 6]f64 = undefined;
    for (v[0 .. n * d]) |*x| x.* = if (smith.boolWeighted(1, 7)) 0 else 2 * unitFloat(smith) - 1;
    const edges = try graph.knn(gpa, v[0 .. n * d], d, k);
    defer gpa.free(edges);
    var per_node: [max_n]usize = @splat(0);
    for (edges, 0..) |e, i| {
        try std.testing.expect(e.from != e.to and e.weight > 0 and e.weight <= 1 + 1e-12);
        if (i > 0) try std.testing.expect(edges[i - 1].from <= e.from);
        per_node[e.from] += 1;
    }
    for (per_node[0..n]) |c| try std.testing.expect(c <= k);

    const w = try graph.symmetric(gpa, n, edges);
    defer w.deinit(gpa);
    const s = try graph.normalizedAdjacency(gpa, w);
    defer s.deinit(gpa);
    var y: [max_n]f64 = undefined;
    for (y[0..n]) |*x| x.* = if (smith.boolWeighted(1, 1)) unitFloat(smith) else 0;
    const a = 0.95 * unitFloat(smith);
    var f: [max_n]f64 = undefined;
    _ = try rank.manifold(gpa, s, a, y[0..n], f[0..n], .{});
    for (f[0..n], y[0..n]) |fv, yv| try std.testing.expect(fv >= yv - 1e-9);
}

test "fuzz: knn graph shape and manifold scores never fall below the seeds" {
    try std.testing.fuzz({}, manifoldInvariants, .{ .corpus = &corpus });
}
