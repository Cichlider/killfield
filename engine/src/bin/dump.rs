//! Rust side of the differential test. Must produce a byte-identical dump to
//! `tools/difftest/dump_js.mjs`.

use kf_engine::constants::MAXDEADENDPENALTY;
use kf_engine::maze::*;
use kf_engine::rng::Rng;

fn f64hex(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    format!("{:016x}", v.to_bits())
}

fn main() {
    let mut out: Vec<String> = Vec::new();
    for seed in [1u32, 2, 3, 7, 42, 1337, 20260814, 4294967295] {
        out.push(format!("== seed {}", seed));
        let mut r = Rng::new(seed);
        let draws: Vec<String> = (0..16).map(|_| f64hex(r.random())).collect();
        out.push(format!("rng {}", draws.join(" ")));
        let mut r2 = Rng::new(seed);
        let rr: Vec<String> = (0..16).map(|_| r2.randrange(4).to_string()).collect();
        out.push(format!("randrange {}", rr.join(" ")));

        for (w, h) in [(4usize, 4usize), (7, 5), (12, 10), (5, 9)] {
            let mut rm = Rng::new(seed);
            let maze = create_maze(w, h, &mut rm);
            out.push(format!("-- maze {}x{}", w, h));

            let mut cells = String::new();
            for x in 0..w {
                for y in 0..h {
                    let c = maze.at(x, y);
                    cells.push_str(&format!("{}{}", c[1], c[2]));
                }
            }
            out.push(format!("cells {}", cells));

            let reach = calc_reachable(&maze, 0, 0);
            out.push(format!(
                "reach {}",
                reach.cells.iter().map(|&(x, y)| format!("{},{}", x, y))
                    .collect::<Vec<_>>().join(" ")
            ));

            let de = find_dead_ends(&maze, &reach.cells, MAXDEADENDPENALTY);
            let mut des: Vec<String> = Vec::new();
            for x in 0..w { for y in 0..h { des.push(f64hex(de[x * h + y])); } }
            out.push(format!("de {}", des.join(" ")));

            let dist = calc_distances(&maze, 0, 0);
            let mut ds: Vec<String> = Vec::new();
            for x in 0..w { for y in 0..h { ds.push(f64hex(dist[x * h + y])); } }
            out.push(format!("dist {}", ds.join(" ")));

            let path = shortest_path_with_distances(&maze, &dist, 0, 0, w - 1, h - 1);
            out.push(format!(
                "path {}",
                path.iter().map(|&(x, y)| format!("{},{}", x, y))
                    .collect::<Vec<_>>().join(" ")
            ));

            let walls = build_wall_segments(&maze, 50.0);
            let ws: Vec<String> = walls.iter()
                .map(|s| s.iter().map(|v| fmt_js_num(*v)).collect::<Vec<_>>().join(","))
                .collect();
            out.push(format!("walls {}", ws.join(" ")));
        }
    }
    println!("{}", out.join("\n"));
}

/// Match JS `Number.prototype.toString()` for the integral values wall
/// segments always hold (they are all `Math.floor` results).
fn fmt_js_num(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e21 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}
