use std::env;
use std::path::PathBuf;

fn first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| p.is_dir()).cloned()
}

fn main() {
    println!("cargo:rerun-if-env-changed=TENSORRT_ROOT");
    println!("cargo:rerun-if-env-changed=TENSORRT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=TENSORRT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    // TensorRT ships either `<root>/include/x86_64-linux-gnu` (deb/rpm) or a
    // flat `<root>/include` (pip wheels, tarballs, aarch64 layouts).
    let trt_root = PathBuf::from(env::var("TENSORRT_ROOT").unwrap_or_else(|_| "/usr".into()));
    let arch = std::env::consts::ARCH;
    let trt_include = match env::var("TENSORRT_INCLUDE_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => first_existing(&[
            trt_root.join("include").join(format!("{arch}-linux-gnu")),
            trt_root.join("include").join(arch),
            trt_root.join("include"),
        ])
        .unwrap_or_else(|| trt_root.join("include")),
    };
    let trt_lib = match env::var("TENSORRT_LIB_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => first_existing(&[
            trt_root.join("lib").join(format!("{arch}-linux-gnu")),
            trt_root.join("lib").join(arch),
            trt_root.join("lib64"),
            trt_root.join("lib"),
        ])
        .unwrap_or_else(|| trt_root.join("lib64")),
    };

    // CUDA: env CUDA_ROOT / CUDA_HOME overrides.
    let cuda_root =
        PathBuf::from(env::var("CUDA_ROOT").or_else(|_| env::var("CUDA_HOME")).unwrap_or_else(|_| "/usr/local/cuda".into()));

    println!("cargo:rustc-link-search=native={}", trt_lib.display());
    println!("cargo:rustc-link-search=native={}", cuda_root.join("lib64").display());
    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=cudart");

    cc::Build::new()
        .cpp(true)
        .file("csrc/wrapper.cpp")
        .include(&trt_include)
        .include(cuda_root.join("include"))
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O2")
        .compile("rstrt_wrappers");

    println!("cargo:rerun-if-changed=csrc/wrapper.cpp");
    println!("cargo:rerun-if-changed=csrc/wrapper.h");
}
