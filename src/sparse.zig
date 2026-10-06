//! Square sparse matrices in compressed sparse row form.

const std = @import("std");

pub const Entry = struct { row: u32, col: u32, val: f64 };

/// How `fromEntries` combines entries with the same row and column.
pub const Merge = enum { sum, max };

pub const Csr = struct {
    n: usize,
    row_ptr: []u32,
    col: []u32,
    val: []f64,

    pub fn deinit(a: Csr, gpa: std.mem.Allocator) void {
        gpa.free(a.row_ptr);
        gpa.free(a.col);
        gpa.free(a.val);
    }

    /// y = A x.
    pub fn mul(a: Csr, x: []const f64, y: []f64) void {
        std.debug.assert(x.len == a.n and y.len == a.n);
        for (0..a.n) |r| {
            var sum: f64 = 0;
            for (a.row_ptr[r]..a.row_ptr[r + 1]) |k| sum += a.val[k] * x[a.col[k]];
            y[r] = sum;
        }
    }

    pub fn trace(a: Csr) f64 {
        var t: f64 = 0;
        for (0..a.n) |r| {
            for (a.row_ptr[r]..a.row_ptr[r + 1]) |k| {
                if (a.col[k] == r) t += a.val[k];
            }
        }
        return t;
    }

    pub fn entries(a: Csr, gpa: std.mem.Allocator) ![]Entry {
        const out = try gpa.alloc(Entry, a.val.len);
        for (0..a.n) |r| {
            for (a.row_ptr[r]..a.row_ptr[r + 1]) |k| out[k] = .{ .row = @intCast(r), .col = a.col[k], .val = a.val[k] };
        }
        return out;
    }
};

fn before(_: void, x: Entry, y: Entry) bool {
    return x.row < y.row or (x.row == y.row and x.col < y.col);
}

/// Builds an n x n matrix. Entries may come in any order; entries at the same position are
/// combined by `merge`.
pub fn fromEntries(gpa: std.mem.Allocator, n: usize, list: []const Entry, merge: Merge) !Csr {
    const sorted = try gpa.dupe(Entry, list);
    defer gpa.free(sorted);
    std.mem.sort(Entry, sorted, {}, before);

    var unique: usize = 0;
    for (sorted, 0..) |e, i| {
        std.debug.assert(e.row < n and e.col < n);
        if (i == 0 or before({}, sorted[i - 1], e)) unique += 1;
    }
    const row_ptr = try gpa.alloc(u32, n + 1);
    errdefer gpa.free(row_ptr);
    const col = try gpa.alloc(u32, unique);
    errdefer gpa.free(col);
    const val = try gpa.alloc(f64, unique);
    @memset(row_ptr, 0);

    var k: usize = 0;
    for (sorted, 0..) |e, i| {
        if (i > 0 and !before({}, sorted[i - 1], e)) {
            val[k - 1] = switch (merge) {
                .sum => val[k - 1] + e.val,
                .max => @max(val[k - 1], e.val),
            };
            continue;
        }
        col[k] = e.col;
        val[k] = e.val;
        row_ptr[e.row + 1] += 1;
        k += 1;
    }
    for (1..n + 1) |r| row_ptr[r] += row_ptr[r - 1];
    return .{ .n = n, .row_ptr = row_ptr, .col = col, .val = val };
}

/// alpha * A + beta * B.
pub fn combine(gpa: std.mem.Allocator, alpha: f64, a: Csr, beta: f64, b: Csr) !Csr {
    std.debug.assert(a.n == b.n);
    const list = try gpa.alloc(Entry, a.val.len + b.val.len);
    defer gpa.free(list);
    const ea = try a.entries(gpa);
    defer gpa.free(ea);
    const eb = try b.entries(gpa);
    defer gpa.free(eb);
    for (ea, list[0..ea.len]) |e, *o| o.* = .{ .row = e.row, .col = e.col, .val = alpha * e.val };
    for (eb, list[ea.len..]) |e, *o| o.* = .{ .row = e.row, .col = e.col, .val = beta * e.val };
    return fromEntries(gpa, a.n, list, .sum);
}

test "fromEntries merges duplicates and multiplies" {
    const gpa = std.testing.allocator;
    const a = try fromEntries(gpa, 3, &.{
        .{ .row = 2, .col = 0, .val = 1 },
        .{ .row = 0, .col = 1, .val = 2 },
        .{ .row = 0, .col = 1, .val = 3 },
        .{ .row = 1, .col = 1, .val = 4 },
    }, .sum);
    defer a.deinit(gpa);
    var y: [3]f64 = undefined;
    a.mul(&.{ 1, 10, 100 }, &y);
    try std.testing.expectEqualSlices(f64, &.{ 50, 40, 1 }, &y);
    try std.testing.expectEqual(@as(f64, 4), a.trace());

    const m = try fromEntries(gpa, 2, &.{ .{ .row = 0, .col = 0, .val = 2 }, .{ .row = 0, .col = 0, .val = 5 } }, .max);
    defer m.deinit(gpa);
    try std.testing.expectEqualSlices(f64, &.{5}, m.val);
}

test "combine" {
    const gpa = std.testing.allocator;
    const a = try fromEntries(gpa, 2, &.{.{ .row = 0, .col = 1, .val = 1 }}, .sum);
    defer a.deinit(gpa);
    const b = try fromEntries(gpa, 2, &.{ .{ .row = 0, .col = 1, .val = 1 }, .{ .row = 1, .col = 0, .val = 1 } }, .sum);
    defer b.deinit(gpa);
    const c = try combine(gpa, 2, a, -1, b);
    defer c.deinit(gpa);
    var y: [2]f64 = undefined;
    c.mul(&.{ 1, 1 }, &y);
    try std.testing.expectEqualSlices(f64, &.{ 1, -1 }, &y);
}
