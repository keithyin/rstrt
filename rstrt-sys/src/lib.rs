//! Raw FFI bindings for the TensorRT + CUDA wrapper.
//!
//! This crate is the thin `-sys` layer: it compiles the C wrapper in `csrc/`,
//! links against `nvinfer` / `cudart`, and re-exposes the `extern "C"` API.
//! Higher-level safe wrappers live in the `rstrt` crate.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int};

pub const TRT_OK: c_int = 0;
pub const TRT_ERR_GENERIC: c_int = 1;
pub const TRT_ERR_NOT_FOUND: c_int = 2;
pub const TRT_ERR_CUDA: c_int = 3;

/// `nvinfer1::TensorIOMode` values as passed over the C ABI.
pub const TRT_IO_INPUT: i32 = 1;
pub const TRT_IO_OUTPUT: i32 = 2;

/// Opaque handle returned by [`trt_infer_create`].
pub type TrtInfer = std::ffi::c_void;

unsafe extern "C" {
    /// Last error message for the current thread; empty string when none.
    pub fn trt_infer_last_error() -> *const c_char;

    /// Load the engine, create the execution context and a dedicated stream.
    /// Returns null on failure — inspect [`trt_infer_last_error`].
    pub fn trt_infer_create(engine_path: *const c_char) -> *mut TrtInfer;
    pub fn trt_infer_free(handle: *mut TrtInfer);

    pub fn trt_infer_nb_io(handle: *mut TrtInfer) -> i32;
    /// Tensor name at index `i`, or NULL for a bad index.
    pub fn trt_infer_get_io_name(handle: *mut TrtInfer, i: i32) -> *const c_char;
    /// `TRT_IO_INPUT` / `TRT_IO_OUTPUT`, or -1 for a bad index.
    pub fn trt_infer_get_io_mode(handle: *mut TrtInfer, i: i32) -> i32;
    /// `nvinfer1::DataType` as int, or -1 for a bad index.
    pub fn trt_infer_get_io_dtype(handle: *mut TrtInfer, i: i32) -> i32;
    pub fn trt_infer_get_io_ndims(handle: *mut TrtInfer, i: i32) -> i32;
    pub fn trt_infer_get_io_dims(handle: *mut TrtInfer, i: i32, out: *mut i64, max_len: i32) -> i32;

    pub fn trt_infer_alloc(
        handle: *mut TrtInfer,
        name: *const c_char,
        dims: *const i64,
        ndims: i32,
    ) -> c_int;

    pub fn trt_infer_pinned_ptr(handle: *mut TrtInfer, name: *const c_char) -> usize;
    pub fn trt_infer_byte_size(handle: *mut TrtInfer, name: *const c_char) -> i64;

    pub fn trt_infer_infer(handle: *mut TrtInfer) -> c_int;
}

/// Read the current thread's last C error string into a `String`.
pub fn last_error() -> String {
    unsafe {
        let p = trt_infer_last_error();
        if p.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}
