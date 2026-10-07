//! Safe, ergonomic wrapper over the C layer in [`rstrt_sys`].
//!
//! `TrtInfer` owns one TensorRT execution context, one CUDA stream, and a
//! pinned/device buffer per I/O tensor. It is `Send` but not `Sync`: the C
//! handle is not thread-safe, so never share it across threads — wrapping it
//! in a `Mutex` does not make concurrent use sound. Create it on one thread,
//! hand it off, and drive it from a single owner thread; with one object per
//! thread you get multi-stream inference for free.
//!
//! # Borrow semantics
//!
//! [`TrtInfer::pinned_view`] / [`TrtInfer::pinned_view_mut`] return typed
//! views over the pinned host buffers. Views borrow the `TrtInfer`, and the
//! borrow checker enforces all access rules: mutable views exclude each
//! other and every other access, and [`TrtInfer::infer`] — which overwrites
//! every output buffer and therefore takes `&mut self` — cannot run while
//! any view is alive. The pinned buffers are reused across `infer()` calls;
//! re-take a view after each call to observe the new batch's data.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_int, c_void};
use std::path::Path;
use std::ptr::NonNull;

use ndarray::{ArrayView1, ArrayViewMut1};

/// Error surfaced from the C layer or from argument validation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
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

impl TryFrom<i32> for IoMode {
    type Error = Error;

    fn try_from(v: i32) -> Result<Self, Error> {
        match v {
            rstrt_sys::TRT_IO_INPUT => Ok(IoMode::Input),
            rstrt_sys::TRT_IO_OUTPUT => Ok(IoMode::Output),
            _ => Err(Error::msg(format!("invalid io mode code {v}"))),
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

/// Marks the Rust element types a pinned tensor buffer can be viewed as,
/// pairing each with the [`DType`] it requires.
///
/// Implemented for `f32`, `i32`, `i64`, and `half::f16` / `half::bf16`; the
/// bound on [`TrtInfer::pinned_view`] / [`TrtInfer::pinned_view_mut`].
pub trait Element: Sized {
    /// The dtype a tensor must have for a `Self` view of its buffer to be
    /// valid.
    const DTYPE: DType;
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;
}
impl Element for i32 {
    const DTYPE: DType = DType::I32;
}
impl Element for i64 {
    const DTYPE: DType = DType::I64;
}
impl Element for half::f16 {
    const DTYPE: DType = DType::F16;
}
impl Element for half::bf16 {
    const DTYPE: DType = DType::Bf16;
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

// SAFETY: the C handle is not bound to the thread that created it and holds
// no Rust data, so it may be *moved* to another thread. It must not be
// *shared*: all use requires exclusive access (the C side is not
// thread-safe), which is why no `Sync` impl exists.
unsafe impl Send for TrtInfer {}

/// RAII ownership of a C handle from `trt_infer_create`: frees it on drop.
/// Used only inside `TrtInfer::new` so that early returns cannot leak the
/// handle; the finished `TrtInfer` takes over ownership via `release`.
struct HandleGuard(NonNull<rstrt_sys::TrtInfer>);

impl HandleGuard {
    fn release(self) -> NonNull<rstrt_sys::TrtInfer> {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }
}

impl Drop for HandleGuard {
    fn drop(&mut self) {
        // SAFETY: the handle came from `trt_infer_create` and is freed here
        // exactly once — `release` hands ownership to `TrtInfer` instead of
        // dropping, and `TrtInfer::Drop` frees it there.
        unsafe { rstrt_sys::trt_infer_free(self.0.as_ptr()) };
    }
}

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
        // SAFETY: `ptr` came from `trt_infer_create` and has not been freed:
        // `HandleGuard::release` handed sole ownership here, and no other
        // code frees it.
        unsafe { rstrt_sys::trt_infer_free(self.ptr.as_ptr()) };
    }
}

/// Map a C return code plus the thread-local error message to an [`Error`].
fn err_from_rc(rc: c_int, context: &str) -> Error {
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
fn read_io_meta(h: *mut rstrt_sys::TrtInfer, i: c_int) -> Result<IoMeta, Error> {
    let ctx = |m: String| format!("io[{i}]: {m}");
    // SAFETY: `h` is a live handle from `trt_infer_create`; a bad `i` yields
    // NULL, handled below. A returned name is owned by the handle and lives
    // until `trt_infer_free`.
    let c_name = unsafe { rstrt_sys::trt_infer_get_io_name(h, i) };
    if c_name.is_null() {
        return Err(Error::msg(format!("bad io index {i}")));
    }
    // SAFETY: `c_name` is a valid NUL-terminated string per the check above.
    let name = unsafe { CStr::from_ptr(c_name) }
        .to_string_lossy()
        .into_owned();
    let mode = IoMode::try_from(
        // SAFETY: live handle; a bad `i` yields -1, rejected by `try_from`.
        unsafe { rstrt_sys::trt_infer_get_io_mode(h, i) },
    )
    .map_err(|e| Error::msg(ctx(e.to_string())))?;
    let dtype = DType::try_from(
        // SAFETY: live handle.
        unsafe { rstrt_sys::trt_infer_get_io_dtype(h, i) },
    )
    .map_err(|e| Error::msg(ctx(e.to_string())))?;
    // SAFETY: live handle.
    let ndims = unsafe { rstrt_sys::trt_infer_get_io_ndims(h, i) };
    if ndims < 0 {
        return Err(Error::msg(format!("bad io index {i}")));
    }
    let mut dims = vec![0i64; ndims as usize];
    // SAFETY: `dims` is valid for writes of `ndims` `i64` values.
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
    pub fn new(engine_path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = engine_path.as_ref();
        let cpath = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| {
            Error::msg(format!(
                "invalid engine path (contains NUL): {}",
                path.display()
            ))
        })?;
        // SAFETY: `cpath` is a valid NUL-terminated C string for the
        // duration of the call.
        let ptr = unsafe { rstrt_sys::trt_infer_create(cpath.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            Error::msg(format!(
                "failed to create TrtInfer: {}",
                rstrt_sys::last_error()
            ))
        })?;

        // `guard` owns the C handle until ownership moves into `Self`; any
        // early return below frees it.
        let guard = HandleGuard(ptr);

        // SAFETY: `guard` holds a live handle from `trt_infer_create`.
        let nb = unsafe { rstrt_sys::trt_infer_nb_io(guard.0.as_ptr()) };
        if nb < 0 {
            return Err(Error::msg("trt_infer_nb_io returned a negative count"));
        }
        let mut metas = Vec::with_capacity(nb as usize);
        for i in 0..nb {
            // SAFETY: live handle; `i` is within [0, nb).
            metas.push(read_io_meta(guard.0.as_ptr(), i)?);
        }
        let cnames = metas
            .iter()
            .map(|m| {
                CString::new(m.name.as_str())
                    .map_err(|e| Error::msg(format!("tensor name {:?} contains NUL: {e}", m.name)))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let index = metas
            .iter()
            .enumerate()
            .map(|(i, m)| (m.name.clone(), i))
            .collect();

        let ptr = guard.release();
        Ok(Self {
            ptr,
            metas,
            index,
            cnames,
        })
    }

    /// Metadata for all I/O tensors, in engine order.
    pub fn tensors(&self) -> &[IoMeta] {
        &self.metas
    }

    /// Metadata for the `i`-th I/O tensor (engine order), or `None` if out
    /// of range.
    pub fn io(&self, i: usize) -> Option<&IoMeta> {
        self.metas.get(i)
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
    /// Takes `&mut self` because it permanently changes engine-side state:
    /// allocations last for the object's lifetime, and allocating a tensor
    /// twice is an error.
    ///
    /// `shape` must match the engine's declared shape for the tensor — the C
    /// layer validates this and rejects mismatches, because the engine reads
    /// and writes the buffers at its own static shape.
    pub fn allocate_memory_for(&mut self, name: &str, shape: &[i64]) -> Result<(), Error> {
        if shape.iter().any(|&d| d < 0) {
            return Err(Error::msg(format!(
                "allocate_memory_for({name}): negative dim in shape {shape:?}"
            )));
        }
        let cname = self.c_name(name)?;
        // SAFETY: `cname` is a live NUL-terminated name; `shape.as_ptr()` is
        // valid for reads of `shape.len()` `i64` values for the call.
        let rc = unsafe {
            rstrt_sys::trt_infer_alloc(
                self.ptr.as_ptr(),
                cname.as_ptr(),
                shape.as_ptr(),
                shape.len() as c_int,
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
    pub fn allocate_all(&mut self) -> Result<(), Error> {
        // Snapshot name/shape pairs first: `allocate_memory_for` takes
        // `&mut self`, which excludes borrowing `self.metas` mid-loop.
        let shapes: Vec<(String, Vec<i64>)> = self
            .metas
            .iter()
            .map(|m| (m.name.clone(), m.shape.clone()))
            .collect();
        for (name, shape) in &shapes {
            self.allocate_memory_for(name, shape)?;
        }
        Ok(())
    }

    /// Pinned host pointer and byte size for `name`; errors when the tensor
    /// has not been allocated yet.
    fn buffer_info(&self, name: &str) -> Result<(*mut c_void, i64), Error> {
        let cname = self.c_name(name)?;
        // SAFETY: `cname` is a live NUL-terminated name; the returned
        // pointer stays valid for as long as `self` (until `trt_infer_free`).
        let p = unsafe { rstrt_sys::trt_infer_pinned_ptr(self.ptr.as_ptr(), cname.as_ptr()) };
        if p.is_null() {
            return Err(Error::msg(format!(
                "tensor '{name}' has no allocated buffer; call allocate_memory_for first"
            )));
        }
        // SAFETY: live handle and valid name, as above.
        let bytes = unsafe { rstrt_sys::trt_infer_byte_size(self.ptr.as_ptr(), cname.as_ptr()) };
        Ok((p, bytes))
    }

    /// Common validation for every pinned-memory view: the tensor exists,
    /// its dtype matches the requested element type, and the element count
    /// derived from the shape fits the allocated buffer.
    fn view_buffer<T: Element>(&self, name: &str) -> Result<(*mut T, usize), Error> {
        let meta = self.meta_of(name)?;
        if meta.dtype != T::DTYPE {
            return Err(Error::msg(format!(
                "dtype mismatch for tensor '{name}': buffer is {:?}, requested a {} view",
                meta.dtype,
                std::any::type_name::<T>()
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

    /// Read-only 1-D view over the tensor's pinned host buffer, with the
    /// element type selecting the expected dtype (see [`Element`]).
    ///
    /// The view borrows `self`: it cannot outlive the `TrtInfer`, cannot
    /// coexist with a mutable view, and must be dropped before
    /// [`infer`](Self::infer) — the borrow checker enforces all of this. The
    /// underlying buffer is reused across `infer()` calls; re-take the view
    /// after each call to observe the new batch.
    pub fn pinned_view<T: Element>(&self, name: &str) -> Result<ArrayView1<'_, T>, Error> {
        let (ptr, n) = self.view_buffer::<T>(name)?;
        // SAFETY: the pinned buffer holds exactly `n` elements of `T` (dtype
        // and byte size checked in `view_buffer`), `cudaHostAlloc` memory is
        // aligned far beyond any `T`, and no mutable access to the buffer
        // can exist while this shared view is borrowed: the only writes
        // happen inside `infer`, which takes `&mut self` and is excluded by
        // the borrow.
        Ok(unsafe { ArrayView1::from_shape_ptr(n, ptr) })
    }

    /// Mutable 1-D view over the tensor's pinned host buffer, with the
    /// element type selecting the expected dtype (see [`Element`]).
    pub fn pinned_view_mut<T: Element>(
        &mut self,
        name: &str,
    ) -> Result<ArrayViewMut1<'_, T>, Error> {
        let (ptr, n) = self.view_buffer::<T>(name)?;
        // SAFETY: as in `pinned_view`, plus: the exclusive `&mut self` borrow
        // guarantees no other view of this buffer exists for this view's
        // lifetime.
        Ok(unsafe { ArrayViewMut1::from_shape_ptr(n, ptr) })
    }

    /// Run one inference: H2D all inputs, enqueue, D2H all outputs, then sync
    /// the stream. Inputs must already be written into their pinned buffers.
    ///
    /// Takes `&mut self` because the engine reads every input buffer and
    /// overwrites every output buffer — no view may be alive across this
    /// call, which the borrow checker enforces.
    pub fn infer(&mut self) -> Result<(), Error> {
        let mut unallocated: Vec<&str> = Vec::new();
        for meta in &self.metas {
            let cname = self.c_name(&meta.name)?;
            // SAFETY: live handle and valid name owned by `self`.
            let p = unsafe { rstrt_sys::trt_infer_pinned_ptr(self.ptr.as_ptr(), cname.as_ptr()) };
            if p.is_null() {
                unallocated.push(&meta.name);
            }
        }
        if !unallocated.is_empty() {
            return Err(Error::msg(format!(
                "allocate_memory_for not called for: {}",
                unallocated.join(", ")
            )));
        }

        // SAFETY: live handle; every tensor is allocated per the check above.
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
    fn io_mode_try_from_matches_abi_constants() {
        assert_eq!(IoMode::try_from(rstrt_sys::TRT_IO_INPUT), Ok(IoMode::Input));
        assert_eq!(
            IoMode::try_from(rstrt_sys::TRT_IO_OUTPUT),
            Ok(IoMode::Output)
        );
        assert!(matches!(IoMode::try_from(0), Err(Error::Generic(_))));
        assert!(matches!(IoMode::try_from(-1), Err(Error::Generic(_))));
    }

    #[test]
    fn element_dtype_mapping() {
        assert_eq!(<f32 as Element>::DTYPE, DType::F32);
        assert_eq!(<i64 as Element>::DTYPE, DType::I64);
        assert_eq!(<half::f16 as Element>::DTYPE, DType::F16);
        assert_eq!(<half::bf16 as Element>::DTYPE, DType::Bf16);
    }

    #[test]
    fn numel_rejects_negative_and_overflow() {
        assert_eq!(numel(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(numel(&[]).unwrap(), 1);
        assert!(numel(&[-1]).is_err());
        assert!(numel(&[i64::MAX, i64::MAX]).is_err());
    }
}
