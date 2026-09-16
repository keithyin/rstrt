//! Safe, ergonomic wrapper over [`rstrt-sys`].
//!
//! `TrtInfer` owns one TensorRT execution context, one CUDA stream, and a
//! pinned/device buffer per I/O tensor. It is `Send` but not `Sync`: create it
//! on one thread, hand it off, and drive it from a single owner thread. With
//! one object per thread you get multi-stream inference for free.

use std::ffi::CString;
use std::ptr::NonNull;

use ndarray::{ArrayView1, ArrayViewMut1};

pub use rstrt_sys::IoMode;

/// Error surfaced from the C layer or from argument validation.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct Error(String);

impl Error {
    fn msg(m: impl Into<String>) -> Self {
        Self(m.into())
    }
}

/// Tensor data type, mirroring the C wrapper's `TrtDataType` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    I32 = 3,
    Bool = 4,
    U8 = 5,
    Fp8 = 6,
    Bf16 = 7,
    I64 = 8,
    I4 = 9,
    Fp4 = 10,
}

impl From<i32> for DType {
    fn from(v: i32) -> Self {
        match v {
            0 => DType::F32,
            1 => DType::F16,
            2 => DType::I8,
            3 => DType::I32,
            4 => DType::Bool,
            5 => DType::U8,
            6 => DType::Fp8,
            7 => DType::Bf16,
            8 => DType::I64,
            9 => DType::I4,
            10 => DType::Fp4,
            _ => DType::F32,
        }
    }
}

/// Static metadata for one I/O tensor.
#[derive(Debug, Clone)]
pub struct IoMeta {
    pub name: String,
    pub mode: IoMode,
    pub dtype: DType,
    pub shape: Vec<i64>,
}

/// A TensorRT inference engine bound to its own stream and I/O buffers.
pub struct TrtInfer {
    ptr: NonNull<rstrt_sys::TrtInfer>,
}

// The execution context is single-owner; allow moving the handle to the thread
// that drives it, but never sharing it across threads.
unsafe impl Send for TrtInfer {}

impl std::fmt::Debug for TrtInfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let addr = self.ptr.as_ptr() as *const () as usize;
        f.debug_struct("TrtInfer").field("ptr", &addr).finish()
    }
}

impl Drop for TrtInfer {
    fn drop(&mut self) {
        unsafe { rstrt_sys::trt_infer_free(self.ptr.as_ptr()) };
    }
}

impl TrtInfer {
    /// Load the engine at `engine_path`, build the execution context, and create
    /// the dedicated stream.
    pub fn new(engine_path: &str) -> Result<Self, Error> {
        let cpath = CString::new(engine_path)
            .map_err(|e| Error::msg(format!("invalid engine path: {e}")))?;
        let ptr = unsafe { rstrt_sys::trt_infer_create(cpath.as_ptr()) };
        let raw = NonNull::new(ptr).ok_or_else(|| {
            Error::msg(format!(
                "failed to create TrtInfer: {}",
                rstrt_sys::last_error()
            ))
        })?;
        Ok(Self { ptr: raw })
    }

    /// Number of I/O tensors.
    pub fn nb_io(&self) -> usize {
        unsafe { rstrt_sys::trt_infer_nb_io(self.ptr.as_ptr()) as usize }
    }

    /// Metadata for the `i`-th I/O tensor (engine order).
    pub fn io(&self, i: usize) -> Result<IoMeta, Error> {
        let h = self.ptr.as_ptr();
        let c_name = unsafe { rstrt_sys::trt_infer_get_io_name(h, i as i32) };
        let name = if c_name.is_null() {
            return Err(Error::msg(format!("bad io index {i}")));
        } else {
            unsafe { std::ffi::CStr::from_ptr(c_name).to_string_lossy().into_owned() }
        };
        let mode = unsafe { rstrt_sys::trt_infer_get_io_mode(h, i as i32) };
        let dtype = unsafe { rstrt_sys::trt_infer_get_io_dtype(h, i as i32) };
        let ndims = unsafe { rstrt_sys::trt_infer_get_io_ndims(h, i as i32) };
        let mut dims = vec![0i64; ndims as usize];
        unsafe {
            rstrt_sys::trt_infer_get_io_dims(h, i as i32, dims.as_mut_ptr(), ndims);
        }
        Ok(IoMeta {
            name,
            mode: if mode == 1 { IoMode::Input } else { IoMode::Output },
            dtype: DType::from(dtype),
            shape: dims,
        })
    }

    /// Allocate pinned (host) + device memory for `name`, bind the device buffer
    /// to the context, and (for inputs) set its input shape. `shape` is the full
    /// tensor shape; call once per I/O tensor.
    pub fn allocate_memory_for(&self, name: &str, shape: &[i64]) -> Result<(), Error> {
        let cname = CString::new(name)
            .map_err(|e| Error::msg(format!("invalid tensor name: {e}")))?;
        let rc = unsafe {
            rstrt_sys::trt_infer_alloc(
                self.ptr.as_ptr(),
                cname.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
            )
        };
        if rc != rstrt_sys::TRT_OK {
            return Err(Error::msg(format!(
                "allocate_memory_for({name}): {}",
                rstrt_sys::last_error()
            )));
        }
        Ok(())
    }

    /// Total element count for a tensor's buffer given its shape.
    fn numel(shape: &[i64]) -> usize {
        shape.iter().map(|&d| d.max(0) as usize).product()
    }

    fn pinned_ptr(&self, name: &str) -> Result<*mut std::ffi::c_void, Error> {
        let cname = CString::new(name)
            .map_err(|e| Error::msg(format!("invalid tensor name: {e}")))?;
        let p = unsafe { rstrt_sys::trt_infer_pinned_ptr(self.ptr.as_ptr(), cname.as_ptr()) };
        if p == 0 {
            return Err(Error::msg(format!("tensor '{name}' has no allocated buffer")));
        }
        Ok(p as *mut std::ffi::c_void)
    }

    fn shape_of(&self, name: &str) -> Result<Vec<i64>, Error> {
        for i in 0..self.nb_io() {
            if let Ok(meta) = self.io(i) {
                if meta.name == name {
                    return Ok(meta.shape);
                }
            }
        }
        Err(Error::msg(format!("unknown tensor: {name}")))
    }

    /// Read-only 1-D `f32` view over the tensor's pinned buffer. Reshape to the
    /// tensor's shape on the caller's side.
    pub fn get_pinned_memory_f32(&self, name: &str) -> Result<ArrayView1<'_, f32>, Error> {
        let ptr = self.pinned_ptr(name)?;
        let n = Self::numel(&self.shape_of(name)?);
        Ok(unsafe { ArrayView1::from_shape_ptr(n, ptr.cast()) })
    }

    /// Mutable 1-D `f32` view over the tensor's pinned buffer.
    pub fn get_pinned_memory_f32_mut(&mut self, name: &str) -> Result<ArrayViewMut1<'_, f32>, Error> {
        let ptr = self.pinned_ptr(name)?;
        let n = Self::numel(&self.shape_of(name)?);
        Ok(unsafe { ArrayViewMut1::from_shape_ptr(n, ptr.cast()) })
    }

    /// Read-only 1-D `i64` view over the tensor's pinned buffer.
    pub fn get_pinned_memory_i64(&self, name: &str) -> Result<ArrayView1<'_, i64>, Error> {
        let ptr = self.pinned_ptr(name)?;
        let n = Self::numel(&self.shape_of(name)?);
        Ok(unsafe { ArrayView1::from_shape_ptr(n, ptr.cast()) })
    }

    /// Mutable 1-D `i64` view over the tensor's pinned buffer.
    pub fn get_pinned_memory_i64_mut(&mut self, name: &str) -> Result<ArrayViewMut1<'_, i64>, Error> {
        let ptr = self.pinned_ptr(name)?;
        let n = Self::numel(&self.shape_of(name)?);
        Ok(unsafe { ArrayViewMut1::from_shape_ptr(n, ptr.cast()) })
    }

    /// Run one inference: H2D all inputs, enqueue, D2H all outputs, then sync
    /// the stream. Inputs must already be written into their pinned buffers.
    pub fn infer(&self) -> Result<(), Error> {
        let rc = unsafe { rstrt_sys::trt_infer_infer(self.ptr.as_ptr()) };
        if rc != rstrt_sys::TRT_OK {
            return Err(Error::msg(format!(
                "infer: {}",
                rstrt_sys::last_error()
            )));
        }
        Ok(())
    }
}
