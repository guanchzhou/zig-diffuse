//! Replays test/golden.json, the outputs of the independent reference in test/reference.py.
//! k-nearest-neighbour edges must match exactly; vectors must satisfy
//! ||got - want||_inf <= 1e-10 * max(1, ||want||_inf).

const std = @import("std");
const diffuse = @import("diffuse");
const sparse = diffuse.sparse;
const graph = diffuse.graph;

const Golden = struct {
    expm: []const struct { n: usize, entries: []const [3]f64, t: f64, b: []const f64, x: []const f64 },
    heat: []const struct { n: usize, edges: []const [3]f64, s: []const f64, t: f64, beta: f64, out: []const f64 },
    advect: []const struct { n: usize, edges: []const [3]f64, s: []const f64, t: f64, gamma: f64, beta: f64, out: []const f64 },
    manifold: []const struct {
        vectors: []const f64,
        d: usize,
        k: usize,
        edges: []const [3]f64,
        a: f64,
        y: []const f64,
        f: []const f64,
        s: []const f64,
        beta: f64,
        blended: []const f64,
    },
};

fn load(arena: std.mem.Allocator) !Golden {
    return std.json.parseFromSliceLeaky(Golden, arena, @embedFile("golden.json"), .{});
}

fn expectClose(want: []const f64, got: []const f64) !void {
    try std.testing.expectEqual(want.len, got.len);
    var scale: f64 = 1;
    for (want) |w| scale = @max(scale, @abs(w));
    for (want, got) |w, g| {
        if (!(@abs(w - g) <= 1e-10 * scale)) {
            std.debug.print("want {d}, got {d} (scale {d})\n", .{ w, g, scale });
            return error.TestExpectedApproxEq;
        }
    }
}

fn toEdges(arena: std.mem.Allocator, rows: []const [3]f64) ![]graph.Edge {
    const out = try arena.alloc(graph.Edge, rows.len);
    for (rows, out) |r, *e| e.* = .{ .from = @intFromFloat(r[0]), .to = @intFromFloat(r[1]), .weight = r[2] };
    return out;
}

test "expm matches scipy.linalg.expm" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const g = try load(arena);
    for (g.expm) |c| {
        const list = try arena.alloc(sparse.Entry, c.entries.len);
        for (c.entries, list) |r, *e| e.* = .{ .row = @intFromFloat(r[0]), .col = @intFromFloat(r[1]), .val = r[2] };
        const a = try sparse.fromEntries(arena, c.n, list, .sum);
        const x = try arena.alloc(f64, c.n);
        _ = try diffuse.expm.action(arena, a, c.t, c.b, x);
        try expectClose(c.x, x);
    }
}

test "heat matches the eigendecomposition" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const g = try load(arena);
    for (g.heat) |c| {
        const w = try graph.symmetric(arena, c.n, try toEdges(arena, c.edges));
        const l = try graph.laplacian(arena, w);
        const out = try arena.alloc(f64, c.n);
        _ = try diffuse.rank.heat(arena, l, c.t, c.beta, c.s, out);
        try expectClose(c.out, out);
    }
}

test "advection-diffusion matches scipy.linalg.expm" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const g = try load(arena);
    for (g.advect) |c| {
        const edges = try toEdges(arena, c.edges);
        const l_sym = try graph.laplacian(arena, try graph.symmetric(arena, c.n, edges));
        const l_adv = try graph.advection(arena, c.n, edges);
        const op = try sparse.combine(arena, c.gamma, l_adv, 1 - c.gamma, l_sym);
        const out = try arena.alloc(f64, c.n);
        _ = try diffuse.rank.heat(arena, op, c.t, c.beta, c.s, out);
        try expectClose(c.out, out);
    }
}

test "manifold ranking matches brute-force knn and a direct solve" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const g = try load(arena);
    for (g.manifold) |c| {
        const edges = try graph.knn(arena, c.vectors, c.d, c.k);
        const want = try toEdges(arena, c.edges);
        try std.testing.expectEqual(want.len, edges.len);
        for (want, edges) |we, ge| {
            try std.testing.expectEqual(we.from, ge.from);
            try std.testing.expectEqual(we.to, ge.to);
            try std.testing.expectApproxEqAbs(we.weight, ge.weight, 1e-12);
        }
        const n = c.y.len;
        const s = try graph.normalizedAdjacency(arena, try graph.symmetric(arena, n, edges));
        const f = try arena.alloc(f64, n);
        _ = try diffuse.rank.manifold(arena, s, c.a, c.y, f, .{});
        try expectClose(c.f, f);
        const blended = try arena.alloc(f64, n);
        try diffuse.rank.blend(c.s, f, c.beta, blended);
        try expectClose(c.blended, blended);
    }
}
