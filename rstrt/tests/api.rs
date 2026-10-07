//! API-level tests for the validation added in 0.2.0: dtype checks, shape
//! checks, re-allocation errors, and the `allocate_all` fast path.
//!
//! Skips (passes) when the sample engine is unavailable; see `common/mod.rs`.

mod common;

use rstrt::{Error, IoMode};

fn open() -> Option<rstrt::TrtInfer> {
    let plan = common::plan_path()?;
    Some(rstrt::TrtInfer::new(plan.to_str().unwrap()).expect("create"))
}

#[test]
fn engine_metadata_snapshot() {
    let Some(infer) = open() else {
        eprintln!("skipping: sample engine not found (set RSTRT_PLAN to enable)");
        return;
    };
    assert_eq!(infer.nb_io(), 3);
    assert_eq!(infer.tensors().len(), infer.nb_io());

    let by_name = |n: &str| infer.tensors().iter().find(|m| m.name == n).unwrap();
    assert_eq!(by_name("feature").mode, IoMode::Input);
    assert_eq!(by_name("feature").dtype, rstrt::DType::F32);
    assert_eq!(by_name("feature").shape, vec![128, 200, 61]);
    assert_eq!(by_name("length").dtype, rstrt::DType::I64);
    assert_eq!(by_name("probs").mode, IoMode::Output);

    assert!(matches!(infer.io(99), Err(Error::Generic(_))));
}

#[test]
fn dtype_mismatch_is_rejected() {
    let Some(infer) = open() else {
        eprintln!("skipping: sample engine not found (set RSTRT_PLAN to enable)");
        return;
    };
    // `length` is i64; an f32 view would reinterpret (and over-read) the buffer.
    let err = infer.get_pinned_memory_f32("length").unwrap_err();
    assert!(matches!(err, Error::Generic(_)), "got: {err:?}");
    assert!(err.to_string().contains("dtype mismatch"), "got: {err}");
}

#[test]
fn unknown_tensor_is_not_found() {
    let Some(infer) = open() else {
        eprintln!("skipping: sample engine not found (set RSTRT_PLAN to enable)");
        return;
    };
    let err = infer.get_pinned_memory_f32("nonexistent").unwrap_err();
    assert_eq!(err, Error::NotFound("nonexistent".to_string()));
}

#[test]
fn double_allocation_is_rejected() {
    let Some(infer) = open() else {
        eprintln!("skipping: sample engine not found (set RSTRT_PLAN to enable)");
        return;
    };
    infer.allocate_memory_for("length", &[128]).unwrap();
    let err = infer.allocate_memory_for("length", &[64]).unwrap_err();
    assert!(matches!(err, Error::Generic(_)), "got: {err:?}");
    assert!(err.to_string().contains("already allocated"), "got: {err}");
}

#[test]
fn output_shape_mismatch_is_rejected_at_alloc() {
    let Some(infer) = open() else {
        eprintln!("skipping: sample engine not found (set RSTRT_PLAN to enable)");
        return;
    };
    // Undersizing an output would make the engine write past the buffer.
    let err = infer.allocate_memory_for("probs", &[1]).unwrap_err();
    assert!(matches!(err, Error::Generic(_)), "got: {err:?}");
    assert!(err.to_string().contains("shape mismatch"), "got: {err}");
}

#[test]
fn infer_reports_unallocated_tensors() {
    let Some(infer) = open() else {
        eprintln!("skipping: sample engine not found (set RSTRT_PLAN to enable)");
        return;
    };
    infer.allocate_memory_for("feature", &[128, 200, 61]).unwrap();
    let err = infer.infer().unwrap_err();
    assert!(err.to_string().contains("length"), "got: {err}");
    assert!(err.to_string().contains("probs"), "got: {err}");
}

#[test]
fn allocate_all_runs_baseline() {
    let Some(mut infer) = open() else {
        eprintln!("skipping: sample engine not found (set RSTRT_PLAN to enable)");
        return;
    };
    infer.allocate_all().unwrap();

    {
        let feat: Vec<f32> = (0..128 * 200 * 61)
            .map(|i| 0.001f32 * (((i * 31) % 1000) as f32) - 0.5f32)
            .collect();
        let mut fv = infer.get_pinned_memory_f32_mut("feature").unwrap();
        fv.assign(&ndarray::ArrayView1::from(&feat));
    }
    infer.get_pinned_memory_i64_mut("length").unwrap().fill(200);
    infer.infer().unwrap();

    let p = infer.get_pinned_memory_f32("probs").unwrap();
    let close = |a: f32, b: f32| (a - b).abs() < 1e-5;
    assert!(close(p[1], 0.999_511_7_f32), "p[1]={}", p[1]);
    assert!(close(p[0], 0.0f32), "p[0]={}", p[0]);
}
