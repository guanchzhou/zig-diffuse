//! Operations on a `Csr` that has already passed `Csr.validate`, for loops that would otherwise
//! validate on every call. Not exported from the package.

const std = @import("std");
const Csr = @import("sparse.zig").Csr;

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

pub fn rowSums(a: Csr, out: []f64) void {
    std.debug.assert(out.len == a.n);
    for (0..a.n) |r| {
        var d: f64 = 0;
        for (a.row_ptr[r]..a.row_ptr[r + 1]) |k| d += a.val[k];
        out[r] = d;
    }
}
