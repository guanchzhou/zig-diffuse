//! zig-diffuse: re-ranking by diffusion on sparse graphs.
//!
//! - `sparse`: square compressed-sparse-row matrices.
//! - `graph`: symmetric weights, normalised adjacency and Laplacian, advection, k-nearest
//!   neighbours.
//! - `expm`: the action of the matrix exponential (Al-Mohy and Higham 2011).
//! - `rank`: heat diffusion, advection-diffusion and manifold ranking of scores.

const std = @import("std");

pub const sparse = @import("sparse.zig");
pub const graph = @import("graph.zig");
pub const expm = @import("expm.zig");
pub const rank = @import("rank.zig");

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(sparse);
    std.testing.refAllDecls(graph);
    std.testing.refAllDecls(expm);
    std.testing.refAllDecls(rank);
    _ = @import("fuzz.zig");
}
