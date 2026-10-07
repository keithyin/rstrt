# rstrt

A safe, ergonomic Rust wrapper over NVIDIA TensorRT. A single `TrtInfer` owns
one TensorRT execution context, one CUDA stream, and a pinned/device buffer per
I/O tensor. It is `Send` but not `Sync`: the C handle is not thread-safe, so
never share it across threads (wrapping it in a `Mutex` does not make
concurrent use sound). Create it on one thread, hand it off, and drive it from
a single owner thread. With one object per thread you get multi-stream
inference for free.

The caller is responsible for static shapes — the engine is built for fixed
shapes and the buffer sizes derive from them. `allocate_memory_for` validates
the requested shape against the engine (element count must match), views are
checked against the tensor's dtype, and allocating a tensor twice or running
`infer()` with unallocated tensors are errors — misuse fails loudly instead of
reinterpreting memory.

## Workspace layout

- **`rstrt`** — the safe Rust API (`TrtInfer`, `DType`, `IoMeta`) built on top of the C layer.
- **`rstrt-sys`** — the FFI/sys crate: a thin C++ wrapper (`csrc/wrapper.cpp`) plus the `build.rs` that compiles it and links against TensorRT + CUDA.
- **`cpp_code/`** — the original standalone C++ reference implementation (and the recorded baseline the Rust e2e test is validated against).

## Prerequisites

The build links against system-installed TensorRT and CUDA. Defaults are the
distro paths; override with env vars if your install differs:

| env var | default |
| --- | --- |
| `TENSORRT_ROOT` | `/usr` — probed as `<root>/include[/lib]/x86_64-linux-gnu`, then flat `<root>/include` / `<root>/lib64` / `<root>/lib` (pip wheels and tarballs) |
| `TENSORRT_INCLUDE_DIR` / `TENSORRT_LIB_DIR` | direct overrides, win over `TENSORRT_ROOT` |
| `CUDA_ROOT` / `CUDA_HOME` | `/usr/local/cuda` (probed as `<root>/lib64` then `<root>/lib`) |
| `CUDA_LIB_DIR` | direct override for the CUDA library directory |

TensorRT 10.x pip wheels ship only libraries (no headers); one way to assemble
a `TENSORRT_ROOT` is to point `lib/` at the wheel's `tensorrt_libs/` directory
and copy the matching headers from the
[NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) release branch. Engines
can only be deserialized by the TensorRT version that built them (or a newer
one within the same serialization format), so match the version accordingly.

## Building the TensorRT engine

`TrtInfer::new` loads a serialized engine (`.plan`). Build one from the ONNX
model with `trtexec` for the target static shapes:

```
trtexec   --onnx=model.onnx   --saveEngine=model.plan   --fp16   --workspace=4096   --minShapes=feature:128x200x61,length:128   --optShapes=feature:128x200x61,length:128   --maxShapes=feature:128x200x61,length:128
```

## Usage

**Setup** — either allocate every I/O tensor from the engine's own shape:

```rust
let mut infer = rstrt::TrtInfer::new("/path/to/model.plan")?;
infer.allocate_all()?; // pinned + device buffers sized by the engine
```

or allocate a buffer per tensor with an explicit full shape (an element-count
match with the engine's shape is required; flattening an output such as
`[128, 200, 2]` into `[128 * 200, 2]` is fine):

```rust
infer.allocate_memory_for("feature", &[128, 200, 61])?;
infer.allocate_memory_for("length",  &[128])?;
infer.allocate_memory_for("probs",   &[(128 * 200), 2])?;
```

**Infer** — write inputs into the pinned buffers, run, then read outputs:

```rust
{
    let mut feat = infer.pinned_view_mut::<f32>("feature")?;
    // fill `feat` with a batch ...
}
{
    let mut len = infer.pinned_view_mut::<i64>("length")?;
    len.fill(200);
}

infer.infer()?; // H2D inputs, enqueue, D2H outputs, then sync the stream

let probs = infer.pinned_view::<f32>("probs")?; // read-only 1-D view
```

Notes:

- The `pinned_view` / `pinned_view_mut` accessors are generic over the element
  type and return **1-D** `ndarray` views; reshape on the caller's side.
  Supported element types are `f32`, `f16`, `bf16`, `i32`, and `i64`
  (`f16`/`bf16` via the `half` crate). Requesting a view whose dtype differs
  from the tensor's is an error.
- Views borrow the `TrtInfer`: `infer()` takes `&mut self`, so no view can be
  alive across an inference — the borrow checker enforces it. The pinned
  buffers are reused across calls; re-take the view after each call to observe
  the new batch.
- `infer()` is synchronous: it blocks until the outputs are back in host
  memory, and errors early if any I/O tensor has not been allocated yet.
- Query engine metadata with `tensors()` / `io(i)` for tensor names, modes,
  dtypes, and shapes (snapshotted once at load time).
- Errors are `rstrt::Error`: `NotFound` (unknown tensor name), `Cuda`
  (CUDA API failure), or `Generic` (everything else, including shape/dtype
  validation failures).

## Examples & tests

```bash
# multi-stream sanity check: N threads, each owning a TrtInfer
cargo run -p rstrt --example bench 4

# end-to-end: replicate the C++ reference input and check `probs` against the
# recorded baseline
cargo test -p rstrt
```

Tests and the bench example locate the sample engine at
`2025Q1-stage2-selfattn-2o-onnx/model.fp16.plan` relative to the repo, or via
the `RSTRT_PLAN` env var. Tests that need it skip gracefully when it is
absent — the engine directory lives outside both crates' package directories,
so it is never part of the crates.io package in the first place.
