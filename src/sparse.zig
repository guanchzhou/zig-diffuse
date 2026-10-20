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

/// alpha * A + beta * B in linear O(n + nnz(A) + nnz(B)) time via sorted row merge.
pub fn combine(gpa: std.mem.Allocator, alpha: f64, a: Csr, beta: f64, b: Csr) !Csr {
    std.debug.assert(a.n == b.n);
    const n = a.n;
    const row_ptr = try gpa.alloc(u32, n + 1);
    errdefer gpa.free(row_ptr);
    row_ptr[0] = 0;

    var total_nnz: usize = 0;
    for (0..n) |r| {
        var ia: usize = a.row_ptr[r];
        const a_end: usize = a.row_ptr[r + 1];
        var ib: usize = b.row_ptr[r];
        const b_end: usize = b.row_ptr[r + 1];
        while (ia < a_end and ib < b_end) {
            const ca = a.col[ia];
            const cb = b.col[ib];
            if (ca < cb) {
                ia += 1;
            } else if (cb < ca) {
                ib += 1;
            } else {
                ia += 1;
                ib += 1;
            }
            total_nnz += 1;
        }
        total_nnz += (a_end - ia) + (b_end - ib);
        row_ptr[r + 1] = @intCast(total_nnz);
    }

    const col = try gpa.alloc(u32, total_nnz);
    errdefer gpa.free(col);
    const val = try gpa.alloc(f64, total_nnz);

    var k: usize = 0;
    for (0..n) |r| {
        var ia: usize = a.row_ptr[r];
        const a_end: usize = a.row_ptr[r + 1];
        var ib: usize = b.row_ptr[r];
        const b_end: usize = b.row_ptr[r + 1];
        while (ia < a_end and ib < b_end) {
            const ca = a.col[ia];
            const cb = b.col[ib];
            if (ca < cb) {
                col[k] = ca;
                val[k] = alpha * a.val[ia];
                ia += 1;
            } else if (cb < ca) {
                col[k] = cb;
                val[k] = beta * b.val[ib];
                ib += 1;
            } else {
                col[k] = ca;
                val[k] = alpha * a.val[ia] + beta * b.val[ib];
                ia += 1;
                ib += 1;
            }
            k += 1;
        }
        while (ia < a_end) : (ia += 1) {
            col[k] = a.col[ia];
            val[k] = alpha * a.val[ia];
            k += 1;
        }
        while (ib < b_end) : (ib += 1) {
            col[k] = b.col[ib];
            val[k] = beta * b.val[ib];
            k += 1;
        }
    }

    return .{ .n = n, .row_ptr = row_ptr, .col = col, .val = val };
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
