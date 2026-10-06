//! Command-line cases run by `zig build test-cli`. `stdout` is compared
//! exactly, `stderr` as a substring.

pub const Case = struct {
    name: []const u8,
    args: []const []const u8,
    stdin: ?[]const u8 = null,
    stdout: ?[]const u8 = null,
    stderr: ?[]const u8 = null,
    exit: u8 = 0,
};

pub const cases = [_]Case{
    .{
        .name = "heat: path graph",
        .args = &.{"heat"},
        .stdin = "{\"n\":3,\"edges\":[[0,1],[1,2]],\"scores\":[1,0,0],\"t\":1,\"beta\":1}",
        .stdout = "{\"scores\":[0.4677735413948744,0.3057051423380683,0.09989410022343197],\"m\":20,\"s\":1,\"matvecs\":19}\n",
    },
    .{
        .name = "advect: pure advection along a path",
        .args = &.{"advect"},
        .stdin = "{\"n\":3,\"edges\":[[0,1],[1,2]],\"scores\":[1,0,0],\"t\":1,\"gamma\":1,\"beta\":1}",
        .stdout = "{\"scores\":[0.3678794411714423,0.3678794411714422,0.26424111765711544],\"m\":20,\"s\":1,\"matvecs\":17}\n",
    },
    .{
        .name = "knn: positive cosine only",
        .args = &.{"knn"},
        .stdin = "{\"vectors\":[[1,0],[0.9,0.1],[0,1]],\"k\":1}",
        .stdout = "{\"edges\":[[0,1,0.9938837346736189],[1,0,0.9938837346736189],[2,1,0.11043152607484655]]}\n",
    },
    .{
        .name = "manifold: edges and blend",
        .args = &.{"manifold"},
        .stdin = "{\"n\":3,\"edges\":[[0,1],[1,2]],\"a\":0.9,\"y\":[1,0,0],\"scores\":[0.5,0.2,0.1],\"beta\":0.5}",
        .stdout = "{\"f\":[3.13157894736842,3.349453174041539,2.13157894736842],\"iterations\":3,\"scores\":[0.46422057025243263,0.35,0.1]}\n",
    },
    .{
        .name = "expm: quarter rotation",
        .args = &.{"expm"},
        .stdin = "{\"n\":2,\"entries\":[[0,1,-1],[1,0,1]],\"t\":1.5707963267948966,\"b\":[1,0]}",
        .stdout = "{\"x\":[0.00000000000000006429041063271854,1.0000000000000002],\"m\":21,\"s\":1,\"matvecs\":21}\n",
    },
    .{
        .name = "error: edge outside the graph",
        .args = &.{"heat"},
        .stdin = "{\"n\":2,\"edges\":[[0,5]],\"scores\":[1,0],\"t\":1}",
        .stderr = "5 is not a node index below 2",
        .exit = 1,
    },
    .{
        .name = "error: fractional node index",
        .args = &.{"heat"},
        .stdin = "{\"n\":2,\"edges\":[[0,0.5]],\"scores\":[1,0],\"t\":1}",
        .stderr = "is not a node index",
        .exit = 1,
    },
    .{
        .name = "error: negative weight",
        .args = &.{"advect"},
        .stdin = "{\"n\":2,\"edges\":[[0,1,-1]],\"scores\":[1,0],\"t\":1}",
        .stderr = "expected a finite value >= 0",
        .exit = 1,
    },
    .{
        .name = "error: a outside [0, 1)",
        .args = &.{"manifold"},
        .stdin = "{\"n\":1,\"edges\":[],\"a\":1,\"y\":[1]}",
        .stderr = "a must be in [0, 1)",
        .exit = 1,
    },
    .{
        .name = "error: scores length",
        .args = &.{"heat"},
        .stdin = "{\"n\":3,\"edges\":[],\"scores\":[1],\"t\":1}",
        .stderr = "scores has 1 values, expected 3",
        .exit = 1,
    },
    .{
        .name = "error: truncated JSON",
        .args = &.{"expm"},
        .stdin = "{",
        .stderr = "invalid input",
        .exit = 1,
    },
    .{
        .name = "error: unknown command",
        .args = &.{"bogus"},
        .stderr = "unknown command: bogus",
        .exit = 2,
    },
    .{
        .name = "error: no command prints usage",
        .args = &.{},
        .stderr = "usage: zig-diffuse COMMAND",
        .exit = 2,
    },
    .{
        .name = "help: exits 0",
        .args = &.{"--help"},
    },
};
