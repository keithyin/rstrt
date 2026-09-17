# rstrt

A safe, ergonomic Rust wrapper over NVIDIA TensorRT. A single `TrtInfer` owns
one TensorRT execution context, one CUDA stream, and a pinned/device buffer per
I/O tensor. It is `Send` but not `Sync`: create it on one thread and drive it
from that owner thread. With one object per thread you get multi-stream
inference for free.

The caller is responsible for static shapes — the engine is built for fixed
shapes and the buffer sizes derive from them.

## Workspace layout

- **`rstrt`** — the safe Rust API (`TrtInfer`, `DType`, `IoMeta`) built on top of the C layer.
- **`rstrt-sys`** — the FFI/sys crate: a thin C++ wrapper (`csrc/wrapper.cpp`) plus the `build.rs` that compiles it and links against TensorRT + CUDA.
- **`cpp_code/`** — the original standalone C++ reference implementation (and the recorded baseline the Rust e2e test is validated against).

## Prerequisites

The build links against system-installed TensorRT and CUDA. Defaults are the
distro paths; override with env vars if your install differs:

| env var | default |
| --- | --- |
| `TENSORRT_ROOT` | `/usr` (include at `<root>/include/x86_64-linux-gnu`, lib at `<root>/lib/x86_64-linux-gnu`) |
| `CUDA_ROOT` / `CUDA_HOME` | `/usr/local/cuda` |

## Building the TensorRT engine

`TrtInfer::new` loads a serialized engine (`.plan`). Build one from the ONNX
model with `trtexec` for the target static shapes:

```
trtexec   --onnx=model.onnx   --saveEngine=model.plan   --fp16   --workspace=4096   --minShapes=feature:128x200x61,length:128   --optShapes=feature:128x200x61,length:128   --maxShapes=feature:128x200x61,length:128
```

## Usage

**Setup** — build the object, then allocate a buffer for every I/O tensor with
its full shape:

```rust
let mut infer = rstrt::TrtInfer::new("/path/to/model.plan")?;
infer.allocate_memory_for("feature", &[128, 200, 61])?;
infer.allocate_memory_for("length",  &[128])?;
infer.allocate_memory_for("probs",   &[(128 * 200), 2])?;
```

**Infer** — write inputs into the pinned buffers, run, then read outputs:

```rust
{
    let mut feat = infer.get_pinned_memory_f32_mut("feature")?;
    // fill `feat` with a batch ...
}
{
    let mut len = infer.get_pinned_memory_i64_mut("length")?;
    len.fill(200);
}

infer.infer()?; // H2D inputs, enqueue, D2H outputs, then sync the stream

let probs = infer.get_pinned_memory_f32("probs")?; // read-only 1-D view
```

Notes:

- The `get_pinned_memory_*` accessors return **1-D** `ndarray` views; reshape on
  the caller's side. An accessor is provided per supported dtype — `f32` and
  `i64` for now, plus `f16`/`bf16` once the C layer exposes them (see
  [`DType`](rstrt/src/lib.rs) for the full set).
- `infer()` is synchronous: it blocks until the outputs are back in host
  memory.
- Query engine metadata with `nb_io()` / `io(i)` for tensor names, modes,
  dtypes, and shapes.

## Examples & tests

```bash
# multi-stream sanity check: N threads, each owning a TrtInfer
cargo run -p rstrt --example bench 4

# end-to-end: replicate the C++ reference input and check `probs` against the
# recorded baseline
cargo test -p rstrt --test e2e
```

Both reference the sample engine at
`2025Q1-stage2-selfattn-2o-onnx/model.fp16.plan`.
