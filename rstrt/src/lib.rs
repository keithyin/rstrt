//! Safe, ergonomic wrapper over the C layer in [`rstrt-sys`].
//!
//! `TrtInfer` owns one TensorRT execution context, one CUDA stream, and a
//! pinned/device buffer per I/O tensor. It is `Send` but not `Sync`: create it
//! on one thread, hand it off, and drive it from a single owner thread. With
//! one object per thread you get multi-stream inference for free.
//!
//! # Buffer lifetime semantics
//!
//! The pinned host buffers are reused across [`TrtInfer::infer`] calls. A
//! read-only view obtained from a `get_pinned_memory_*` accessor borrows the
//! buffer and therefore stays valid across `infer()` — it simply observes the
//! next batch's data. Mutable views take `&mut self` and can only coexist
//! with one another by construction.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::ptr::NonNull;

use ndarray::{ArrayView1, ArrayViewMut1};

/// Error surfaced from the C layer or from argument validation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The named tensor does not exist in the engine.
    #[error("tensor not found: {0}")]
    NotFound(String),
    /// A CUDA API call failed (allocation, memcpy, stream sync, ...).
    #[error("CUDA error: {0}")]
    Cuda(String),
    /// Any other failure from the C layer or from argument validation.
    #[error("{0}")]
    Generic(String),
}

impl Error {
    fn msg(m: impl Into<String>) -> Self {
        Self::Generic(m.into())
    }
}

/// Tensor data type, mirroring `nvinfer1::DataType` (the numeric values are
/// fixed by the C ABI).
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

impl TryFrom<i32> for DType {
    type Error = Error;

    fn try_from(v: i32) -> Result<Self, Error> {
        match v {
            0 => Ok(DType::F32),
            1 => Ok(DType::F16),
            2 => Ok(DType::I8),
            3 => Ok(DType::I32),
            4 => Ok(DType::Bool),
            5 => Ok(DType::U8),
            6 => Ok(DType::Fp8),
            7 => Ok(DType::Bf16),
            8 => Ok(DType::I64),
            9 => Ok(DType::I4),
            10 => Ok(DType::Fp4),
            _ => Err(Error::msg(format!("unknown TensorRT dtype code {v}"))),
        }
    }
}

/// I/O direction of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoMode {
    Input,
    Output,
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
    /// I/O metadata snapshotted once at creation (engine order).
    metas: Vec<IoMeta>,
    /// name -> position in `metas`.
    index: HashMap<String, usize>,
    /// Names as C strings, parallel to `metas` (avoids re-encoding per call).
    cnames: Vec<CString>,
}

// The execution context is single-owner; allow moving the handle to the thread
// that drives it, but never sharing it across threads.
unsafe impl Send for TrtInfer {}

impl std::fmt::Debug for TrtInfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let addr = self.ptr.as_ptr() as *const () as usize;
        f.debug_struct("TrtInfer")
            .field("ptr", &addr)
            .field("tensors", &self.metas.len())
            .finish()
    }
}

impl Drop for TrtInfer {
    fn drop(&mut self) {
        unsafe { rstrt_sys::trt_infer_free(self.ptr.as_ptr()) };
    }
}

/// Map a C return code plus the thread-local error message to an [`Error`].
fn err_from_rc(rc: std::os::raw::c_int, context: &str) -> Error {
    let detail = rstrt_sys::last_error();
    let full = if detail.is_empty() {
        context.to_string()
    } else {
        format!("{context}: {detail}")
    };
    match rc {
        rstrt_sys::TRT_ERR_NOT_FOUND => Error::NotFound(full),
        rstrt_sys::TRT_ERR_CUDA => Error::Cuda(full),
        _ => Error::Generic(full),
    }
}

/// Product of the shape extents, rejecting negative dims and overflow.
fn numel(shape: &[i64]) -> Result<usize, Error> {
    shape.iter().try_fold(1usize, |acc, &d| {
        let d = usize::try_from(d)
            .map_err(|_| Error::msg(format!("negative dim {d} in shape {shape:?}")))?;
        acc.checked_mul(d)
            .ok_or_else(|| Error::msg(format!("element count overflows for shape {shape:?}")))
    })
}

/// Read one tensor's metadata from the C handle. Used only during `new()`.
fn read_io_meta(h: *mut rstrt_sys::TrtInfer, i: i32) -> Result<IoMeta, Error> {
    let ctx = |m: String| format!("io[{i}]: {m}");
    let c_name = unsafe { rstrt_sys::trt_infer_get_io_name(h, i) };
    if c_name.is_null() {
        return Err(Error::msg(format!("bad io index {i}")));
    }
    let name = unsafe { CStr::from_ptr(c_name) }.to_string_lossy().into_owned();
    let mode = match unsafe { rstrt_sys::trt_infer_get_io_mode(h, i) } {
        rstrt_sys::TRT_IO_INPUT => IoMode::Input,
        rstrt_sys::TRT_IO_OUTPUT => IoMode::Output,
        other => return Err(Error::msg(ctx(format!("invalid io mode code {other}")))),
    };
    let dtype = DType::try_from(unsafe { rstrt_sys::trt_infer_get_io_dtype(h, i) })
        .map_err(|e| Error::msg(ctx(e.to_string())))?;
    let ndims = unsafe { rstrt_sys::trt_infer_get_io_ndims(h, i) };
    if ndims < 0 {
        return Err(Error::msg(ctx(format!("bad io index {i}"))));
    }
    let mut dims = vec![0i64; ndims as usize];
    unsafe { rstrt_sys::trt_infer_get_io_dims(h, i, dims.as_mut_ptr(), ndims) };
    Ok(IoMeta {
        name,
        mode,
        dtype,
        shape: dims,
    })
}

impl TrtInfer {
    /// Load the engine at `engine_path`, build the execution context, and
    /// create the dedicated stream.
    pub fn new(engine_path: &str) -> Result<Self, Error> {
        let cpath = CString::new(engine_path)
            .map_err(|e| Error::msg(format!("invalid engine path: {e}")))?;
        let ptr = unsafe { rstrt_sys::trt_infer_create(cpath.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            Error::msg(format!(
                "failed to create TrtInfer: {}",
                rstrt_sys::last_error()
            ))
        })?;

        let nb = unsafe { rstrt_sys::trt_infer_nb_io(ptr.as_ptr()) };
        let (metas, cnames) = {
            let mut metas = Vec::with_capacity(nb.max(0) as usize);
            for i in 0..nb {
                match read_io_meta(ptr.as_ptr(), i) {
                    Ok(m) => metas.push(m),
                    Err(e) => {
                        // Don't leak the C handle; Drop would double-free.
                        unsafe { rstrt_sys::trt_infer_free(ptr.as_ptr()) };
                        return Err(e);
                    }
                }
            }
            let cnames = metas
                .iter()
                .map(|m| CString::new(m.name.as_str()).expect("name round-trips through C"))
                .collect();
            (metas, cnames)
        };
        let index = metas
            .iter()
            .enumerate()
            .map(|(i, m)| (m.name.clone(), i))
            .collect();

        Ok(Self {
            ptr,
            metas,
            index,
            cnames,
        })
    }

    /// Number of I/O tensors.
    pub fn nb_io(&self) -> usize {
        self.metas.len()
    }

    /// Metadata for all I/O tensors, in engine order.
    pub fn tensors(&self) -> &[IoMeta] {
        &self.metas
    }

    /// Metadata for the `i`-th I/O tensor (engine order).
    pub fn io(&self, i: usize) -> Result<IoMeta, Error> {
        self.metas
            .get(i)
            .cloned()
            .ok_or_else(|| Error::msg(format!("bad io index {i}")))
    }

    fn idx(&self, name: &str) -> Result<usize, Error> {
        self.index
            .get(name)
            .copied()
            .ok_or_else(|| Error::NotFound(name.to_string()))
    }

    fn meta_of(&self, name: &str) -> Result<&IoMeta, Error> {
        Ok(&self.metas[self.idx(name)?])
    }

    fn c_name(&self, name: &str) -> Result<&CString, Error> {
        Ok(&self.cnames[self.idx(name)?])
    }

    /// Allocate pinned (host) + device memory for `name` and bind the device
    /// buffer to the context; for inputs the input shape is also set.
    ///
    /// `shape` must match the engine's declared shape for the tensor — the C
    /// layer validates this and rejects mismatches, because the engine reads
    /// and writes the buffers at its own static shape. Allocating a tensor
    /// twice is an error.
    pub fn allocate_memory_for(&self, name: &str, shape: &[i64]) -> Result<(), Error> {
        if shape.iter().any(|&d| d < 0) {
            return Err(Error::msg(format!(
                "allocate_memory_for({name}): negative dim in shape {shape:?}"
            )));
        }
        let cname = self.c_name(name)?;
        let rc = unsafe {
            rstrt_sys::trt_infer_alloc(
                self.ptr.as_ptr(),
                cname.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
            )
        };
        if rc != rstrt_sys::TRT_OK {
            return Err(err_from_rc(rc, &format!("allocate_memory_for({name})")));
        }
        Ok(())
    }

    /// Allocate pinned + device buffers for every I/O tensor using the
    /// engine's own shape. Equivalent to calling
    /// [`allocate_memory_for`](Self::allocate_memory_for) with each tensor's
    /// declared shape; use [`allocate_memory_for`](Self::allocate_memory_for)
    /// only when you need to override a (dynamic) shape.
    pub fn allocate_all(&self) -> Result<(), Error> {
        for meta in &self.metas {
            self.allocate_memory_for(&meta.name, &meta.shape)?;
        }
        Ok(())
    }

    /// Pinned host pointer and byte size for `name`; errors when the tensor
    /// has not been allocated yet.
    fn buffer_info(&self, name: &str) -> Result<(*mut std::ffi::c_void, i64), Error> {
        let cname = self.c_name(name)?;
        let p = unsafe { rstrt_sys::trt_infer_pinned_ptr(self.ptr.as_ptr(), cname.as_ptr()) };
        if p == 0 {
            return Err(Error::msg(format!(
                "tensor '{name}' has no allocated buffer; call allocate_memory_for first"
            )));
        }
        let bytes = unsafe { rstrt_sys::trt_infer_byte_size(self.ptr.as_ptr(), cname.as_ptr()) };
        Ok((p as *mut std::ffi::c_void, bytes))
    }

    /// Common validation for every pinned-memory view: the tensor exists, its
    /// dtype matches the requested element type, and the element count derived
    /// from the shape fits the allocated buffer.
    fn view_buffer<T>(&self, name: &str, expected: DType) -> Result<(*mut T, usize), Error> {
        let meta = self.meta_of(name)?;
        if meta.dtype != expected {
            return Err(Error::msg(format!(
                "dtype mismatch for tensor '{name}': buffer is {:?}, requested a {expected:?} view",
                meta.dtype
            )));
        }
        let (ptr, bytes) = self.buffer_info(name)?;
        let n = numel(&meta.shape)?;
        let need = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::msg(format!("byte size overflows for tensor '{name}'")))?;
        if need > bytes.max(0) as usize {
            return Err(Error::msg(format!(
                "view of tensor '{name}' needs {need} bytes but its buffer holds {bytes}"
            )));
        }
        Ok((ptr.cast(), n))
    }

    /// Read-only 1-D `f32` view over the tensor's pinned buffer. The buffer is
    /// reused across [`infer`](Self::infer) calls, so a held view observes the
    /// next batch's data.
    pub fn get_pinned_memory_f32(&self, name: &str) -> Result<ArrayView1<'_, f32>, Error> {
        let (ptr, n) = self.view_buffer::<f32>(name, DType::F32)?;
        Ok(unsafe { ArrayView1::from_shape_ptr(n, ptr) })
    }

    /// Mutable 1-D `f32` view over the tensor's pinned buffer.
    pub fn get_pinned_memory_f32_mut(&mut self, name: &str) -> Result<ArrayViewMut1<'_, f32>, Error> {
        let (ptr, n) = self.view_buffer::<f32>(name, DType::F32)?;
        Ok(unsafe { ArrayViewMut1::from_shape_ptr(n, ptr) })
    }

    /// Read-only 1-D `i64` view over the tensor's pinned buffer.
    pub fn get_pinned_memory_i64(&self, name: &str) -> Result<ArrayView1<'_, i64>, Error> {
        let (ptr, n) = self.view_buffer::<i64>(name, DType::I64)?;
        Ok(unsafe { ArrayView1::from_shape_ptr(n, ptr) })
    }

    /// Mutable 1-D `i64` view over the tensor's pinned buffer.
    pub fn get_pinned_memory_i64_mut(&mut self, name: &str) -> Result<ArrayViewMut1<'_, i64>, Error> {
        let (ptr, n) = self.view_buffer::<i64>(name, DType::I64)?;
        Ok(unsafe { ArrayViewMut1::from_shape_ptr(n, ptr) })
    }

    /// Read-only 1-D `i32` view over the tensor's pinned buffer.
    pub fn get_pinned_memory_i32(&self, name: &str) -> Result<ArrayView1<'_, i32>, Error> {
        let (ptr, n) = self.view_buffer::<i32>(name, DType::I32)?;
        Ok(unsafe { ArrayView1::from_shape_ptr(n, ptr) })
    }

    /// Mutable 1-D `i32` view over the tensor's pinned buffer.
    pub fn get_pinned_memory_i32_mut(&mut self, name: &str) -> Result<ArrayViewMut1<'_, i32>, Error> {
        let (ptr, n) = self.view_buffer::<i32>(name, DType::I32)?;
        Ok(unsafe { ArrayViewMut1::from_shape_ptr(n, ptr) })
    }

    /// Read-only 1-D `f16` view over the tensor's pinned buffer.
    pub fn get_pinned_memory_f16(&self, name: &str) -> Result<ArrayView1<'_, half::f16>, Error> {
        let (ptr, n) = self.view_buffer::<half::f16>(name, DType::F16)?;
        Ok(unsafe { ArrayView1::from_shape_ptr(n, ptr) })
    }

    /// Mutable 1-D `f16` view over the tensor's pinned buffer.
    pub fn get_pinned_memory_f16_mut(
        &mut self,
        name: &str,
    ) -> Result<ArrayViewMut1<'_, half::f16>, Error> {
        let (ptr, n) = self.view_buffer::<half::f16>(name, DType::F16)?;
        Ok(unsafe { ArrayViewMut1::from_shape_ptr(n, ptr) })
    }

    /// Read-only 1-D `bf16` view over the tensor's pinned buffer.
    pub fn get_pinned_memory_bf16(&self, name: &str) -> Result<ArrayView1<'_, half::bf16>, Error> {
        let (ptr, n) = self.view_buffer::<half::bf16>(name, DType::Bf16)?;
        Ok(unsafe { ArrayView1::from_shape_ptr(n, ptr) })
    }

    /// Mutable 1-D `bf16` view over the tensor's pinned buffer.
    pub fn get_pinned_memory_bf16_mut(
        &mut self,
        name: &str,
    ) -> Result<ArrayViewMut1<'_, half::bf16>, Error> {
        let (ptr, n) = self.view_buffer::<half::bf16>(name, DType::Bf16)?;
        Ok(unsafe { ArrayViewMut1::from_shape_ptr(n, ptr) })
    }

    /// Run one inference: H2D all inputs, enqueue, D2H all outputs, then sync
    /// the stream. Inputs must already be written into their pinned buffers.
    ///
    /// The pinned buffers are reused across calls; see the module docs for
    /// what that means for views held across `infer()`.
    pub fn infer(&self) -> Result<(), Error> {
        let mut unallocated: Vec<&str> = Vec::new();
        for meta in &self.metas {
            let cname = self.c_name(&meta.name)?;
            let p = unsafe { rstrt_sys::trt_infer_pinned_ptr(self.ptr.as_ptr(), cname.as_ptr()) };
            if p == 0 {
                unallocated.push(&meta.name);
            }
        }
        if !unallocated.is_empty() {
            return Err(Error::msg(format!(
                "allocate_memory_for not called for: {}",
                unallocated.join(", ")
            )));
        }

        let rc = unsafe { rstrt_sys::trt_infer_infer(self.ptr.as_ptr()) };
        if rc != rstrt_sys::TRT_OK {
            return Err(err_from_rc(rc, "infer"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_try_from_rejects_unknown_codes() {
        assert_eq!(DType::try_from(0), Ok(DType::F32));
        assert_eq!(DType::try_from(10), Ok(DType::Fp4));
        assert!(matches!(DType::try_from(11), Err(Error::Generic(_))));
        assert!(matches!(DType::try_from(-1), Err(Error::Generic(_))));
    }

    #[test]
    fn numel_rejects_negative_and_overflow() {
        assert_eq!(numel(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(numel(&[]).unwrap(), 1);
        assert!(numel(&[-1]).is_err());
        assert!(numel(&[i64::MAX, i64::MAX]).is_err());
    }
}
