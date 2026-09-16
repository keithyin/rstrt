use std::env;
use std::path::PathBuf;

fn main() {
    // TensorRT: env TENSORRT_ROOT overrides, default to distro paths confirmed on this box.
    let trt_root = env::var("TENSORRT_ROOT").unwrap_or_else(|_| "/usr".to_string());
    let trt_include = PathBuf::from(trt_root.clone()).join("include/x86_64-linux-gnu");
    let trt_lib = PathBuf::from(trt_root.clone()).join("lib/x86_64-linux-gnu");

    // CUDA: env CUDA_ROOT / CUDA_HOME overrides.
    let cuda_root = env::var("CUDA_ROOT")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    println!("cargo:rustc-link-search=native={}", trt_lib.display());
    println!("cargo:rustc-link-search=native={}", PathBuf::from(cuda_root.clone()).join("lib64").display());
    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=cudart");

    cc::Build::new()
        .cpp(true)
        .file("csrc/wrapper.cpp")
        .include(&trt_include)
        .include(PathBuf::from(cuda_root.clone()).join("include"))
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O2")
        .compile("rstrt_wrappers");

    println!("cargo:rerun-if-changed=csrc/wrapper.cpp");
    println!("cargo:rerun-if-changed=csrc/wrapper.h");
}
