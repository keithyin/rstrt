//! Shared helpers for the integration tests.
//!
//! The sample engine lives in the repo (excluded from the crates.io package),
//! so tests skip gracefully when it is absent. Override the location with
//! `RSTRT_PLAN`.

use std::path::{Path, PathBuf};

pub fn plan_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("RSTRT_PLAN") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../2025Q1-stage2-selfattn-2o-onnx/model.fp16.plan");
    p.canonicalize().ok()
}
