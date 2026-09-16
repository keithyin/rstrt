#ifndef RSTRT_WRAPPER_H
#define RSTRT_WRAPPER_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle.
typedef struct TrtInfer TrtInfer;

// Tensor I/O mode (matches nvinfer1::TensorIOMode).
enum TrtIoMode : int32_t {
    kTrtIoInput = 1,
    kTrtIoOutput = 2,
};

// Error codes. 0 == success.
#define TRT_OK                 0
#define TRT_ERR_GENERIC        1
#define TRT_ERR_NOT_FOUND      2
#define TRT_ERR_CUDA           3

// Last error message for this thread. Valid C string until next call.
const char* trt_infer_last_error(void);

// Load engine + create context + create stream. Returns NULL on failure (see last_error).
TrtInfer* trt_infer_create(const char* engine_path);
void trt_infer_free(TrtInfer* h);

// IO metadata. Index i in [0, nb_io).
int32_t trt_infer_nb_io(TrtInfer* h);
// Returns a pointer to a statically-lived C string (owned by the handle).
const char* trt_infer_get_io_name(TrtInfer* h, int32_t i);
int32_t trt_infer_get_io_mode(TrtInfer* h, int32_t i);   // TrtIoMode
int32_t trt_infer_get_io_dtype(TrtInfer* h, int32_t i);  // TrtDataType (see wrapper enum values)
int32_t trt_infer_get_io_ndims(TrtInfer* h, int32_t i);
// Writes up to max_len dims; returns actual ndims.
int32_t trt_infer_get_io_dims(TrtInfer* h, int32_t i, int64_t* out, int32_t max_len);

// Allocate pinned (host) + device memory for the named tensor and bind device ptr to context.
// Inputs also get their input shape set. Returns TRT_OK or error code.
int32_t trt_infer_alloc(TrtInfer* h, const char* name, const int64_t* dims, int32_t ndims);

// Pinned host pointer (as integer) for the named tensor, or 0 if not allocated.
uintptr_t trt_infer_pinned_ptr(TrtInfer* h, const char* name);
// Byte size of the tensor's buffer, or 0 if not allocated.
int64_t trt_infer_byte_size(TrtInfer* h, const char* name);

// Run inference: H2D all inputs -> enqueue -> D2H all outputs -> stream sync.
int32_t trt_infer_infer(TrtInfer* h);

#ifdef __cplusplus
}
#endif

#endif  // RSTRT_WRAPPER_H
