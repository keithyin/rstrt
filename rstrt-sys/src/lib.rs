//! Raw FFI bindings for the TensorRT + CUDA wrapper.
//!
//! This crate is the thin `-sys` layer: it compiles the C wrapper in `csrc/`,
//! links against `nvinfer` / `cudart`, and re-exposes the `extern "C"` API.
//! Higher-level safe wrappers live in the `rstrt` crate.
//!
//! # Thread-safety
//!
//! Every function taking a [`TrtInfer`] handle requires exclusive access to
//! that handle for the duration of the call: the C side is not thread-safe,
//! and concurrently calling any two functions on the same handle is undefined
//! behavior even if the calls "look" read-only. Distinct handles are
//! independent.

use std::ffi::{c_char, c_int, c_void};

pub const TRT_OK: c_int = 0;
pub const TRT_ERR_GENERIC: c_int = 1;
pub const TRT_ERR_NOT_FOUND: c_int = 2;
pub const TRT_ERR_CUDA: c_int = 3;

/// `nvinfer1::TensorIOMode` values as passed over the C ABI.
pub const TRT_IO_INPUT: c_int = 1;
pub const TRT_IO_OUTPUT: c_int = 2;

/// Opaque handle returned by [`trt_infer_create`].
pub type TrtInfer = c_void;

unsafe extern "C" {
    /// Last error message for the current thread; empty string when none.
    ///
    /// # Safety
    ///
    /// Always safe to call (the returned string is owned by a thread-local
    /// and stays valid until the next fallible call on the same thread).
    pub fn trt_infer_last_error() -> *const c_char;

    /// Load the engine, create the execution context and a dedicated stream.
    /// Returns null on failure — inspect [`trt_infer_last_error`].
    ///
    /// # Safety
    ///
    /// `engine_path` must point to a valid, NUL-terminated UTF-8 C string for
    /// the duration of the call.
    pub fn trt_infer_create(engine_path: *const c_char) -> *mut TrtInfer;

    /// Free a handle returned by [`trt_infer_create`] along with all of its
    /// buffers. Null is accepted and ignored.
    ///
    /// # Safety
    ///
    /// `handle` must come from [`trt_infer_create`] and must not have been
    /// freed before (free at most once). The handle must not be used again
    /// afterwards — including from other threads.
    pub fn trt_infer_free(handle: *mut TrtInfer);

    /// Number of I/O tensors.
    ///
    /// # Safety
    ///
    /// `handle` must be a live handle from [`trt_infer_create`] (same
    /// requirement for every function below; not repeated).
    pub fn trt_infer_nb_io(handle: *mut TrtInfer) -> c_int;

    /// Tensor name at index `i`, or NULL for a bad index. The returned
    /// pointer is owned by the handle and stays valid until
    /// [`trt_infer_free`].
    ///
    /// # Safety
    ///
    /// `i` is only interpreted as `[0, nb_io)`; out-of-range returns NULL.
    pub fn trt_infer_get_io_name(handle: *mut TrtInfer, i: c_int) -> *const c_char;

    /// `TRT_IO_INPUT` / `TRT_IO_OUTPUT`, or -1 for a bad index.
    pub fn trt_infer_get_io_mode(handle: *mut TrtInfer, i: c_int) -> c_int;

    /// `nvinfer1::DataType` as int, or -1 for a bad index.
    pub fn trt_infer_get_io_dtype(handle: *mut TrtInfer, i: c_int) -> c_int;

    /// Number of dimensions of the tensor's shape, or -1 for a bad index.
    pub fn trt_infer_get_io_ndims(handle: *mut TrtInfer, i: c_int) -> c_int;

    /// Write up to `max_len` dims of tensor `i` into `out`; returns the
    /// tensor's actual ndims (may exceed `max_len`), or -1 on a bad index or
    /// bad arguments.
    ///
    /// # Safety
    ///
    /// `out` must be valid for writes of `max_len` `i64` values and
    /// `max_len` must be > 0.
    pub fn trt_infer_get_io_dims(
        handle: *mut TrtInfer,
        i: c_int,
        out: *mut i64,
        max_len: c_int,
    ) -> c_int;

    /// Allocate pinned (host) + device memory for the named tensor and bind
    /// the device pointer to the context; inputs also get their input shape
    /// set. Returns `TRT_OK` or an error code (and sets the thread-local
    /// error message).
    ///
    /// # Safety
    ///
    /// `name` must be a valid NUL-terminated C string; `dims` must be valid
    /// for reads of `ndims` `i64` values and `ndims` must be > 0. Allocating
    /// a tensor twice is rejected by the C layer, not UB.
    pub fn trt_infer_alloc(
        handle: *mut TrtInfer,
        name: *const c_char,
        dims: *const i64,
        ndims: c_int,
    ) -> c_int;

    /// Pinned host pointer for the named tensor, or NULL if not allocated.
    /// The pointer stays valid until [`trt_infer_free`].
    pub fn trt_infer_pinned_ptr(handle: *mut TrtInfer, name: *const c_char) -> *mut c_void;

    /// Byte size of the tensor's buffer, or 0 if not allocated.
    pub fn trt_infer_byte_size(handle: *mut TrtInfer, name: *const c_char) -> i64;

    /// Run inference: H2D all inputs -> enqueue -> D2H all outputs -> stream
    /// sync. Returns `TRT_OK` or an error code.
    ///
    /// # Safety
    ///
    /// `name`-style preconditions: every I/O tensor must have been allocated
    /// via [`trt_infer_alloc`] first; the engine reads the input buffers and
    /// overwrites the output buffers, so the caller must not hold live
    /// references into them across this call.
    pub fn trt_infer_infer(handle: *mut TrtInfer) -> c_int;
}

/// Read the current thread's last C error string into a `String`.
pub fn last_error() -> String {
    // SAFETY: the returned pointer is a valid NUL-terminated string owned by
    // the thread-local (or NULL, which is handled below).
    unsafe {
        let p = trt_infer_last_error();
        if p.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}
