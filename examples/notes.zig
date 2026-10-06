const std = @import("std");
const diffuse = @import("diffuse");
const graph = diffuse.graph;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    // Six notes with their query scores. Notes 0, 1 and 3 link to each other; 2, 4 and 5 form a
    // second group. Diffusion moves score along links, so note 3, which the query barely
    // matches, gains from its strongly matched neighbour 0. Whether that improves retrieval is
    // an empirical question; this example only shows the mechanics.
    const scores = [_]f64{ 0.82, 0.40, 0.55, 0.30, 0.52, 0.50 };
    const links = [_]graph.Edge{
        .{ .from = 0, .to = 1 },
        .{ .from = 1, .to = 3 },
        .{ .from = 0, .to = 3 },
        .{ .from = 2, .to = 4 },
        .{ .from = 4, .to = 5 },
    };

    const w = try graph.symmetric(gpa, scores.len, &links);
    defer w.deinit(gpa);
    const l = try graph.laplacian(gpa, w);
    defer l.deinit(gpa);
    var heat: [scores.len]f64 = undefined;
    _ = try diffuse.rank.heat(gpa, l, 2, 0.5, &scores, &heat);
    std.debug.assert(heat[0] < scores[0] and heat[3] > scores[3]);

    // Pure advection (gamma = 1) only moves score along link directions, so the total is kept.
    const l_adv = try graph.advection(gpa, scores.len, &links);
    defer l_adv.deinit(gpa);
    var advected: [scores.len]f64 = undefined;
    _ = try diffuse.rank.heat(gpa, l_adv, 1, 1, &scores, &advected);
    var before: f64 = 0;
    var after: f64 = 0;
    for (scores, advected) |s, a| {
        before += s;
        after += a;
    }
    std.debug.assert(@abs(before - after) < 1e-12);
    std.debug.assert(advected[3] > scores[3]);
}
