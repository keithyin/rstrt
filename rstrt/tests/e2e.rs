//! End-to-end validation: replicate the C++ reference input and check the
//! `probs` output matches the recorded baseline from `cpp_code/ref_main.cpp`.

const PLAN: &str = "/root/projects/rstrt/2025Q1-stage2-selfattn-2o-onnx/model.fp16.plan";
const B: usize = 128;
const T: usize = 200;
const F: usize = 61;

/// Same deterministic input as cpp_code/ref_main.cpp.
fn ref_feature() -> Vec<f32> {
    let n = B * T * F;
    (0..n).map(|i| 0.001f32 * (((i * 31) % 1000) as f32) - 0.5f32).collect()
}

#[test]
fn matches_cpp_baseline() {
    let mut infer = rstrt::TrtInfer::new(PLAN).expect("create");

    // Allocate all I/O.
    infer.allocate_memory_for("feature", &[B as i64, T as i64, F as i64]).unwrap();
    infer.allocate_memory_for("length", &[B as i64]).unwrap();
    infer.allocate_memory_for("probs", &[(B * T) as i64, 2]).unwrap();

    // Write inputs (each borrow of `infer` must end before the next).
    {
        let feat = ref_feature();
        let mut fv = infer.get_pinned_memory_f32_mut("feature").unwrap();
        for (slot, &v) in fv.iter_mut().zip(feat.iter()) {
            *slot = v;
        }
    }
    {
        let mut lv = infer.get_pinned_memory_i64_mut("length").unwrap();
        lv.fill(200);
    }

    infer.infer().unwrap();

    // Read output (read-only view).
    let p = infer.get_pinned_memory_f32("probs").unwrap();
    assert_eq!(p.len(), 51_200);

    let close = |a: f32, b: f32| (a - b).abs() < 1e-5;
    assert!(close(p[1], 0.999511719f32), "p[1]={}", p[1]);
    assert!(close(p[3], 0.999511719f32), "p[3]={}", p[3]);
    assert!(close(p[12345], 0.999511719f32), "p[12345]={}", p[12345]);
    assert!(close(p[0], 0.0f32), "p[0]={}", p[0]);

    let abs: f64 = p.iter().map(|v| f64::from(v.abs())).sum();
    assert!((abs - 25_587.5f64).abs() < 0.01, "abs={}", abs);
}
