//! Multi-stream sanity check: spawn N threads, each owning one `TrtInfer`,
//! and verify all of them can run inference concurrently without deadlock.
//!
//! Run: `cargo run -p rstrt --example bench [N]`

use std::path::{Path, PathBuf};
use std::thread;
use std::time::Instant;

const B: usize = 128;
const T: usize = 200;
const F: usize = 61;
const ITERS: usize = 200;

fn plan_path() -> PathBuf {
    if let Ok(p) = std::env::var("RSTRT_PLAN") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../2025Q1-stage2-selfattn-2o-onnx/model.fp16.plan")
}

fn run_one(plan: &Path) -> f32 {
    let mut infer = rstrt::TrtInfer::new(plan).expect("create");
    infer
        .allocate_memory_for("feature", &[B as i64, T as i64, F as i64])
        .unwrap();
    infer.allocate_memory_for("length", &[B as i64]).unwrap();
    infer
        .allocate_memory_for("probs", &[(B * T) as i64, 2])
        .unwrap();

    {
        let mut fv = infer.pinned_view_mut::<f32>("feature").unwrap();
        for (i, slot) in fv.iter_mut().enumerate() {
            *slot = 0.001f32 * (((i * 31) % 1000) as f32) - 0.5f32;
        }
    }
    {
        let mut lv = infer.pinned_view_mut::<i64>("length").unwrap();
        lv.fill(200);
    }

    // Warmup.
    infer.infer().unwrap();

    let t0 = Instant::now();
    let mut first = 0.0f32;
    for _ in 0..ITERS {
        infer.infer().unwrap();
        let p = infer.pinned_view::<f32>("probs").unwrap();
        first = p[1];
    }
    let dt = t0.elapsed();
    eprintln!(
        "thread done: iters/s={:.1}  first[1]={first}",
        ITERS as f64 / dt.as_secs_f64()
    );
    first
}

fn main() {
    let n: usize = match std::env::args().nth(1) {
        Some(s) => match s.parse() {
            Ok(n) => n,
            Err(_) => {
                eprintln!("invalid thread count: {s}");
                std::process::exit(2);
            }
        },
        None => 2,
    };

    let plan = plan_path();
    if !plan.exists() {
        eprintln!(
            "engine not found at {}; set RSTRT_PLAN to the .plan file",
            plan.display()
        );
        std::process::exit(1);
    }

    let t0 = Instant::now();
    let handles: Vec<_> = (0..n)
        .map(|i| {
            let plan = plan.clone();
            thread::Builder::new()
                .name(format!("infer-{i}"))
                .spawn(move || run_one(&plan))
                .expect("spawn")
        })
        .collect();
    let results: Vec<f32> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let wall = t0.elapsed();
    let all_match = results.iter().all(|&v| (v - results[0]).abs() < 1e-4);
    println!(
        "\n{n} threads, {ITERS} iters each, total wall={:?} ({:.2} ms/iter)\nall outputs consistent: {all_match}",
        wall,
        wall.as_secs_f64() * 1000.0 / (n * ITERS) as f64
    );
    assert!(all_match, "thread outputs diverged");
}
