const std = @import("std");
const diffuse = @import("diffuse");
const build_options = @import("build_options");
const graph = diffuse.graph;
const sparse = diffuse.sparse;

const usage =
    \\usage: zig-diffuse COMMAND < input.json
    \\
    \\Edges are [from, to] or [from, to, weight]; weights default to 1.
    \\
    \\  heat      {"n":N,"edges":[..],"scores":[..],"t":T,"beta":B}
    \\            -> {"scores":[..],"m":M,"s":S,"matvecs":K}
    \\  advect    {"n":N,"edges":[..],"scores":[..],"t":T,"gamma":G,"beta":B}
    \\            -> {"scores":[..],"m":M,"s":S,"matvecs":K}
    \\  knn       {"vectors":[[..]..],"k":K}
    \\            -> {"edges":[[from,to,cosine]..]}
    \\  manifold  {"vectors":[[..]..],"k":K | "n":N,"edges":[..], "a":A,"y":[..],"scores":[..]?,"beta":B?}
    \\            -> {"f":[..],"iterations":I,"scores":[..]?}
    \\  expm      {"n":N,"entries":[[row,col,value]..],"t":T,"b":[..]}
    \\            -> {"x":[..],"m":M,"s":S,"matvecs":K}
    \\  version
    \\
;

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("zig-diffuse: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

fn parse(comptime T: type, arena: std.mem.Allocator, text: []const u8) T {
    return std.json.parseFromSliceLeaky(T, arena, text, .{ .ignore_unknown_fields = true }) catch |e|
        fail("invalid input: {s}", .{@errorName(e)});
}

fn emit(out: *std.Io.Writer, v: anytype) !void {
    try std.json.Stringify.value(v, .{}, out);
    try out.writeAll("\n");
}

fn index(x: f64, n: usize, what: []const u8, i: usize) u32 {
    if (!(x >= 0 and x < @as(f64, @floatFromInt(n)) and @floor(x) == x))
        fail("{s} {d}: {d} is not a node index below {d}", .{ what, i, x, n });
    return @intFromFloat(x);
}

fn edgesOf(arena: std.mem.Allocator, rows: []const []const f64, n: usize) ![]graph.Edge {
    const out = try arena.alloc(graph.Edge, rows.len);
    for (rows, out, 0..) |r, *e, i| {
        if (r.len != 2 and r.len != 3) fail("edge {d} has {d} values, expected 2 or 3", .{ i, r.len });
        const w = if (r.len == 3) r[2] else 1;
        if (!(w >= 0) or std.math.isInf(w)) fail("edge {d} has weight {d}, expected a finite value >= 0", .{ i, w });
        e.* = .{ .from = index(r[0], n, "edge", i), .to = index(r[1], n, "edge", i), .weight = w };
    }
    return out;
}

fn flatten(arena: std.mem.Allocator, rows: []const []const f64) !struct { data: []f64, d: usize } {
    if (rows.len == 0) fail("vectors needs at least one row", .{});
    const d = rows[0].len;
    if (d == 0) fail("vectors must have at least one column", .{});
    const data = try arena.alloc(f64, rows.len * d);
    for (rows, 0..) |r, i| {
        if (r.len != d) fail("vector {d} has {d} values, expected {d}", .{ i, r.len, d });
        @memcpy(data[i * d ..][0..d], r);
    }
    return .{ .data = data, .d = d };
}

fn checkLen(name: []const u8, got: usize, n: usize) void {
    if (got != n) fail("{s} has {d} values, expected {d}", .{ name, got, n });
}

fn finite(name: []const u8, x: f64) void {
    if (!std.math.isFinite(x)) fail("{s} must be finite", .{name});
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    var buf: [64 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &buf);
    const out = &stdout.interface;
    defer out.flush() catch {};
    if (args.len != 2) {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
    const cmd: []const u8 = args[1];
    if (std.mem.eql(u8, cmd, "version") or std.mem.eql(u8, cmd, "--version")) {
        try out.print("zig-diffuse {s}\n", .{build_options.version});
        return;
    }
    if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        try out.writeAll(usage);
        return;
    }
    const commands = [_][]const u8{ "heat", "advect", "knn", "manifold", "expm" };
    for (commands) |c| {
        if (std.mem.eql(u8, cmd, c)) break;
    } else {
        std.debug.print("zig-diffuse: unknown command: {s}\n{s}", .{ cmd, usage });
        std.process.exit(2);
    }

    var in_buf: [64 * 1024]u8 = undefined;
    var stdin = std.Io.File.stdin().readerStreaming(init.io, &in_buf);
    const text = stdin.interface.allocRemaining(arena, .limited(256 * 1024 * 1024)) catch |e| switch (e) {
        error.ReadFailed => return stdin.err.?,
        else => |x| return x,
    };

    if (std.mem.eql(u8, cmd, "heat") or std.mem.eql(u8, cmd, "advect")) {
        const in = parse(struct {
            n: usize,
            edges: []const []const f64,
            scores: []const f64,
            t: f64,
            beta: f64 = 1,
            gamma: f64 = 0,
        }, arena, text);
        checkLen("scores", in.scores.len, in.n);
        finite("t", in.t);
        finite("beta", in.beta);
        const edges = try edgesOf(arena, in.edges, in.n);
        const l_sym = try graph.laplacian(arena, try graph.symmetric(arena, in.n, edges));
        const op = if (std.mem.eql(u8, cmd, "heat")) l_sym else blk: {
            if (!(in.gamma >= 0 and in.gamma <= 1)) fail("gamma must be in [0, 1]", .{});
            break :blk try sparse.combine(arena, in.gamma, try graph.advection(arena, in.n, edges), 1 - in.gamma, l_sym);
        };
        const scores = try arena.alloc(f64, in.n);
        const st = diffuse.rank.heat(arena, op, in.t, in.beta, in.scores, scores) catch |e| fail("{s}", .{@errorName(e)});
        try emit(out, .{ .scores = scores, .m = st.m, .s = st.s, .matvecs = st.matvecs });
    } else if (std.mem.eql(u8, cmd, "knn")) {
        const in = parse(struct { vectors: []const []const f64, k: usize }, arena, text);
        const v = try flatten(arena, in.vectors);
        const edges = try graph.knn(arena, v.data, v.d, in.k);
        const rows = try arena.alloc([3]f64, edges.len);
        for (edges, rows) |e, *r| r.* = .{ @floatFromInt(e.from), @floatFromInt(e.to), e.weight };
        try emit(out, .{ .edges = rows });
    } else if (std.mem.eql(u8, cmd, "manifold")) {
        const in = parse(struct {
            vectors: ?[]const []const f64 = null,
            k: usize = 10,
            n: ?usize = null,
            edges: ?[]const []const f64 = null,
            a: f64,
            y: []const f64,
            scores: ?[]const f64 = null,
            beta: f64 = 0.5,
        }, arena, text);
        if (!(in.a >= 0 and in.a < 1)) fail("a must be in [0, 1)", .{});
        var n: usize = undefined;
        const edges = if (in.vectors) |rows| blk: {
            const v = try flatten(arena, rows);
            n = rows.len;
            break :blk try graph.knn(arena, v.data, v.d, in.k);
        } else blk: {
            n = in.n orelse fail("manifold needs either vectors or n and edges", .{});
            break :blk try edgesOf(arena, in.edges orelse fail("manifold needs either vectors or n and edges", .{}), n);
        };
        checkLen("y", in.y.len, n);
        const s = try graph.normalizedAdjacency(arena, try graph.symmetric(arena, n, edges));
        const f = try arena.alloc(f64, n);
        const iterations = diffuse.rank.manifold(arena, s, in.a, in.y, f, .{}) catch |e| fail("{s}", .{@errorName(e)});
        if (in.scores) |sc| {
            checkLen("scores", sc.len, n);
            const blended = try arena.alloc(f64, n);
            try diffuse.rank.blend(sc, f, in.beta, blended);
            try emit(out, .{ .f = f, .iterations = iterations, .scores = blended });
        } else try emit(out, .{ .f = f, .iterations = iterations });
    } else if (std.mem.eql(u8, cmd, "expm")) {
        const in = parse(struct { n: usize, entries: []const []const f64, t: f64, b: []const f64 }, arena, text);
        checkLen("b", in.b.len, in.n);
        finite("t", in.t);
        const list = try arena.alloc(sparse.Entry, in.entries.len);
        for (in.entries, list, 0..) |r, *e, i| {
            if (r.len != 3) fail("entry {d} has {d} values, expected 3", .{ i, r.len });
            finite("entry value", r[2]);
            e.* = .{ .row = index(r[0], in.n, "entry", i), .col = index(r[1], in.n, "entry", i), .val = r[2] };
        }
        const a = try sparse.fromEntries(arena, in.n, list, .sum);
        const x = try arena.alloc(f64, in.n);
        const st = diffuse.expm.action(arena, a, in.t, in.b, x) catch |e| fail("{s}", .{@errorName(e)});
        try emit(out, .{ .x = x, .m = st.m, .s = st.s, .matvecs = st.matvecs });
    } else unreachable;
}
