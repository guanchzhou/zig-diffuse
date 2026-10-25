//! Square sparse matrices in compressed sparse row form.

const std = @import("std");
const unchecked = @import("unchecked.zig");

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

    /// Checks the structure every operation relies on: n is at most 2^32, `row_ptr` has n + 1
    /// entries, starts at 0, never decreases and ends at nnz, `col` and `val` both hold nnz
    /// entries, and each row's columns are below n and in non-decreasing order. Values are not
    /// checked. Costs O(n + nnz), about as much as one `mul`; every public operation that takes
    /// a `Csr` calls it once.
    pub fn validate(a: Csr) error{InvalidCsr}!void {
        if (a.n > 0 and a.n - 1 > std.math.maxInt(u32)) return error.InvalidCsr;
        if (a.row_ptr.len == 0 or a.row_ptr.len - 1 != a.n) return error.InvalidCsr;
        if (a.row_ptr[0] != 0 or a.row_ptr[a.n] != a.col.len or a.val.len != a.col.len) return error.InvalidCsr;
        for (0..a.n) |r| {
            const start = a.row_ptr[r];
            const end = a.row_ptr[r + 1];
            if (end < start or end > a.col.len) return error.InvalidCsr;
            for (a.col[start..end], 0..) |c, i| {
                if (c >= a.n or (i > 0 and c < a.col[start + i - 1])) return error.InvalidCsr;
            }
        }
    }

    /// y = A x.
    pub fn mul(a: Csr, x: []const f64, y: []f64) error{ InvalidCsr, LengthMismatch }!void {
        try a.validate();
        if (x.len != a.n or y.len != a.n) return error.LengthMismatch;
        unchecked.mul(a, x, y);
    }

    pub fn trace(a: Csr) error{InvalidCsr}!f64 {
        try a.validate();
        return unchecked.trace(a);
    }

    pub fn entries(a: Csr, gpa: std.mem.Allocator) ![]Entry {
        try a.validate();
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

/// Builds an n x n matrix, n at most 2^32. Entries may come in any order; entries at the same
/// position are combined by `merge`.
pub fn fromEntries(gpa: std.mem.Allocator, n: usize, list: []const Entry, merge: Merge) !Csr {
    if (n > 0 and n - 1 > std.math.maxInt(u32)) return error.TooManyNodes;
    for (list) |e| if (e.row >= n or e.col >= n) return error.NodeOutOfRange;
    const sorted = try gpa.dupe(Entry, list);
    defer gpa.free(sorted);
    std.mem.sort(Entry, sorted, {}, before);

    var unique: usize = 0;
    for (sorted, 0..) |e, i| {
        if (i == 0 or before({}, sorted[i - 1], e)) unique += 1;
    }
    if (unique > std.math.maxInt(u32)) return error.TooManyNonzeros;
    const row_ptr = try gpa.alloc(u32, try std.math.add(usize, n, 1));
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
        row_ptr[@as(usize, e.row) + 1] += 1;
        k += 1;
    }
    for (1..n + 1) |r| row_ptr[r] += row_ptr[r - 1];
    return .{ .n = n, .row_ptr = row_ptr, .col = col, .val = val };
}

/// alpha * A + beta * B in linear O(n + nnz(A) + nnz(B)) time via sorted row merge.
pub fn combine(gpa: std.mem.Allocator, alpha: f64, a: Csr, beta: f64, b: Csr) !Csr {
    try a.validate();
    try b.validate();
    if (a.n != b.n) return error.DimensionMismatch;
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
        if (total_nnz > std.math.maxInt(u32)) return error.TooManyNonzeros;
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
    try a.mul(&.{ 1, 10, 100 }, &y);
    try std.testing.expectEqualSlices(f64, &.{ 50, 40, 1 }, &y);
    try std.testing.expectEqual(@as(f64, 4), try a.trace());

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
    try c.mul(&.{ 1, 1 }, &y);
    try std.testing.expectEqualSlices(f64, &.{ 1, -1 }, &y);
}

test "fromEntries rejects an entry outside the matrix" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.NodeOutOfRange, fromEntries(gpa, 2, &.{.{ .row = 2, .col = 0, .val = 1 }}, .sum));
    try std.testing.expectError(error.NodeOutOfRange, fromEntries(gpa, 2, &.{.{ .row = 0, .col = 2, .val = 1 }}, .sum));
    try std.testing.expectError(error.NodeOutOfRange, fromEntries(gpa, 0, &.{.{ .row = 0, .col = 0, .val = 1 }}, .sum));
    try std.testing.expectError(error.NodeOutOfRange, fromEntries(gpa, 3, &.{.{ .row = std.math.maxInt(u32), .col = 0, .val = 1 }}, .sum));
    if (@sizeOf(usize) > 4) {
        try std.testing.expectError(error.TooManyNodes, fromEntries(gpa, (1 << 32) + 1, &.{}, .sum));
    }
}

test "validate rejects malformed matrices" {
    var row_ptr = [_]u32{ 0, 1, 2 };
    var col = [_]u32{ 1, 0 };
    var val = [_]f64{ 1, 1 };
    const ok: Csr = .{ .n = 2, .row_ptr = &row_ptr, .col = &col, .val = &val };
    try ok.validate();

    var bad = ok;
    bad.n = 3;
    try std.testing.expectError(error.InvalidCsr, bad.validate());
    bad = ok;
    bad.val = val[0..1];
    try std.testing.expectError(error.InvalidCsr, bad.validate());
    bad = ok;
    bad.row_ptr = row_ptr[0..0];
    bad.n = 0;
    try std.testing.expectError(error.InvalidCsr, bad.validate());

    var shifted = [_]u32{ 1, 1, 2 };
    bad = ok;
    bad.row_ptr = &shifted;
    try std.testing.expectError(error.InvalidCsr, bad.validate());
    var decreasing = [_]u32{ 0, 2, 1 };
    bad.row_ptr = &decreasing;
    try std.testing.expectError(error.InvalidCsr, bad.validate());
    var past_end = [_]u32{ 0, 3, 2 };
    bad.row_ptr = &past_end;
    try std.testing.expectError(error.InvalidCsr, bad.validate());
    var short_end = [_]u32{ 0, 1, 1 };
    bad.row_ptr = &short_end;
    try std.testing.expectError(error.InvalidCsr, bad.validate());

    var wide = [_]u32{ 2, 0 };
    bad = ok;
    bad.col = &wide;
    try std.testing.expectError(error.InvalidCsr, bad.validate());
    var unsorted_ptr = [_]u32{ 0, 2, 2 };
    var unsorted = [_]u32{ 1, 0 };
    bad = .{ .n = 2, .row_ptr = &unsorted_ptr, .col = &unsorted, .val = &val };
    try std.testing.expectError(error.InvalidCsr, bad.validate());

    var x = [_]f64{ 1, 1 };
    var y: [2]f64 = undefined;
    try std.testing.expectError(error.InvalidCsr, bad.mul(&x, &y));
    try std.testing.expectError(error.InvalidCsr, bad.trace());
    try std.testing.expectError(error.InvalidCsr, bad.entries(std.testing.allocator));
    try std.testing.expectError(error.InvalidCsr, combine(std.testing.allocator, 1, ok, 1, bad));
    try std.testing.expectError(error.LengthMismatch, ok.mul(x[0..1], &y));
    try std.testing.expectError(error.LengthMismatch, ok.mul(&x, y[0..1]));
}

test "combine rejects matrices of different sizes" {
    const gpa = std.testing.allocator;
    const a = try fromEntries(gpa, 2, &.{}, .sum);
    defer a.deinit(gpa);
    const b = try fromEntries(gpa, 3, &.{}, .sum);
    defer b.deinit(gpa);
    try std.testing.expectError(error.DimensionMismatch, combine(gpa, 1, a, 1, b));
}
