#include "wrapper.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // Surface errors + warnings to stderr for debugging.
        if (severity <= Severity::kWARNING)
            std::fprintf(stderr, "[TRT] %s\n", msg);
    }
};

// Thread-local error message, reset by each fallible call.
thread_local std::string g_err;

void err(const std::string& msg)
{
    g_err = msg;
}

struct TensorBuf
{
    std::string name;
    int32_t mode = 0;     // TrtIoMode
    int32_t dtype = 0;    // nvinfer1::DataType as int
    int32_t ndims = 0;
    std::vector<int64_t> dims;

    void* pinned = nullptr;
    void* device = nullptr;
    int64_t bytes = 0;
};

}  // namespace

// Global definition of the opaque handle forward-declared in wrapper.h.
struct TrtInfer
{
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;

    std::vector<std::string> io_names;                 // engine order
    std::map<std::string, TensorBuf> tensors;          // by name
};

namespace {

// Frees everything owned by `h` (mirrors trt_infer_free minus the per-tensor
// buffer frees). Used by the failure paths in trt_infer_create so that a
// partially-built handle does not leak the runtime/engine/stream.
void destroy(TrtInfer* h)
{
    // Destroy consumers of the stream before the stream itself.
    delete h->context;
    delete h->engine;
    delete h->runtime;
    if (h->stream)
        cudaStreamDestroy(h->stream);
    delete h;
}

// Renders dims as "[a,b,c]" for error messages.
std::string dims_str(const std::vector<int64_t>& d)
{
    std::string s = "[";
    for (size_t i = 0; i < d.size(); ++i)
    {
        if (i)
            s += ",";
        s += std::to_string(d[i]);
    }
    return s + "]";
}

// Element count of a shape; negative dims (dynamic) count as 1. Sets
// `overflow` instead of overflowing.
size_t shape_numel(const std::vector<int64_t>& dims, bool& overflow)
{
    overflow = false;
    size_t n = 1;
    for (int64_t v : dims)
    {
        if (v < 0)
            continue;  // dynamic dim: unknown extent, contributes nothing here
        const size_t d = static_cast<size_t>(v);
        if (d != 0 && n > SIZE_MAX / d)
        {
            overflow = true;
            return n;
        }
        n *= d;
    }
    return n;
}

void* read_file(const char* path, size_t& out_size)
{
    std::vector<char> data;
    {
        FILE* f = std::fopen(path, "rb");
        if (!f)
        {
            err(std::string("failed to open engine file: ") + path);
            return nullptr;
        }
        std::fseek(f, 0, SEEK_END);
        long tell = std::ftell(f);
        if (tell < 0)
        {
            std::fclose(f);
            err("failed to determine engine file size");
            return nullptr;
        }
        size_t size = static_cast<size_t>(tell);
        if (size == 0)
        {
            std::fclose(f);
            err("engine file is empty: " + std::string(path));
            return nullptr;
        }
        std::fseek(f, 0, SEEK_SET);
        data.resize(size);
        if (std::fread(data.data(), 1, size, f) != size)
        {
            std::fclose(f);
            err("failed to read engine file");
            return nullptr;
        }
        std::fclose(f);
        out_size = size;
    }
    void* buf = std::malloc(out_size);
    if (!buf)
    {
        err("out of memory reading engine file");
        return nullptr;
    }
    std::memcpy(buf, data.data(), out_size);
    return buf;
}

nvinfer1::IRuntime* make_runtime()
{
    static Logger logger;
    auto* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime)
        err("failed to create TensorRT runtime");
    return runtime;
}

// Element size in bytes for a TensorRT data type.
size_t elem_size(int32_t dtype)
{
    using DT = nvinfer1::DataType;
    switch (static_cast<DT>(dtype))
    {
        case DT::kFLOAT:
        case DT::kINT32:
            return 4;
        case DT::kINT64:
            return 8;
        case DT::kHALF:
        case DT::kBF16:
            return 2;
        case DT::kINT8:
        case DT::kUINT8:
        case DT::kBOOL:
        case DT::kFP8:
            return 1;
        case DT::kINT4:
        case DT::kFP4:
            return 1;  // packed; byte size still bounded by element count
        default:
            return 4;
    }
}

}  // namespace

extern "C" {

const char* trt_infer_last_error(void)
{
    return g_err.empty() ? "" : g_err.c_str();
}

TrtInfer* trt_infer_create(const char* engine_path)
{
    g_err.clear();

    size_t size = 0;
    void* blob = read_file(engine_path, size);
    if (!blob)
        return nullptr;

    auto* h = new (std::nothrow) TrtInfer();
    if (!h)
    {
        err("out of memory");
        std::free(blob);
        return nullptr;
    }

    h->runtime = make_runtime();
    if (!h->runtime)
    {
        destroy(h);
        std::free(blob);
        return nullptr;
    }

    h->engine = h->runtime->deserializeCudaEngine(blob, size);
    std::free(blob);
    if (!h->engine)
    {
        err("failed to deserialize engine");
        destroy(h);
        return nullptr;
    }

    h->context = h->engine->createExecutionContext();
    if (!h->context)
    {
        err("failed to create execution context");
        destroy(h);
        return nullptr;
    }

    if (cudaStreamCreate(&h->stream) != cudaSuccess)
    {
        err(std::string("cudaStreamCreate failed: ") + cudaGetErrorString(cudaGetLastError()));
        destroy(h);
        return nullptr;
    }

    if (!h->context->setOptimizationProfileAsync(0, h->stream))
    {
        err("failed to set optimization profile");
        destroy(h);
        return nullptr;
    }

    int nb = h->engine->getNbIOTensors();
    h->io_names.reserve(nb);
    for (int i = 0; i < nb; ++i)
    {
        const char* name = h->engine->getIOTensorName(i);
        TensorBuf tb;
        tb.name = name;
        tb.mode = static_cast<int32_t>(h->engine->getTensorIOMode(name));
        tb.dtype = static_cast<int32_t>(h->engine->getTensorDataType(name));

        nvinfer1::Dims d = h->context->getTensorShape(name);
        tb.ndims = d.nbDims;
        for (int j = 0; j < d.nbDims; ++j)
            tb.dims.push_back(d.d[j]);

        h->io_names.push_back(tb.name);
        h->tensors[tb.name] = std::move(tb);
    }

    return h;
}

void trt_infer_free(TrtInfer* h)
{
    if (!h)
        return;
    for (auto& [name, tb] : h->tensors)
    {
        if (tb.pinned)
            cudaFreeHost(tb.pinned);
        if (tb.device)
            cudaFree(tb.device);
    }
    destroy(h);
}

int32_t trt_infer_nb_io(TrtInfer* h)
{
    if (!h)
        return 0;
    return static_cast<int32_t>(h->io_names.size());
}

const char* trt_infer_get_io_name(TrtInfer* h, int32_t i)
{
    if (!h || i < 0 || i >= static_cast<int32_t>(h->io_names.size()))
        return nullptr;
    return h->io_names[i].c_str();
}

int32_t trt_infer_get_io_mode(TrtInfer* h, int32_t i)
{
    if (!h || i < 0 || i >= static_cast<int32_t>(h->io_names.size()))
        return -1;
    return h->tensors.at(h->io_names[i]).mode;
}

int32_t trt_infer_get_io_dtype(TrtInfer* h, int32_t i)
{
    if (!h || i < 0 || i >= static_cast<int32_t>(h->io_names.size()))
        return -1;
    return h->tensors.at(h->io_names[i]).dtype;
}

int32_t trt_infer_get_io_ndims(TrtInfer* h, int32_t i)
{
    if (!h || i < 0 || i >= static_cast<int32_t>(h->io_names.size()))
        return -1;
    return h->tensors.at(h->io_names[i]).ndims;
}

int32_t trt_infer_get_io_dims(TrtInfer* h, int32_t i, int64_t* out, int32_t max_len)
{
    if (!h || i < 0 || i >= static_cast<int32_t>(h->io_names.size()) || !out || max_len <= 0)
        return -1;
    const auto& tb = h->tensors.at(h->io_names[i]);
    int32_t n = std::min(tb.ndims, max_len);
    for (int32_t j = 0; j < n; ++j)
        out[j] = tb.dims[j];
    return tb.ndims;
}

int32_t trt_infer_alloc(TrtInfer* h, const char* name, const int64_t* dims, int32_t ndims)
{
    if (!h || !name || !dims || ndims <= 0)
    {
        err("trt_infer_alloc: invalid arguments (null handle/name/dims or ndims <= 0)");
        return TRT_ERR_GENERIC;
    }
    if (ndims > nvinfer1::Dims::MAX_DIMS)
    {
        // `setInputShape` receives a stack-allocated nvinfer1::Dims whose
        // d[] array holds exactly MAX_DIMS entries; a larger rank would
        // write past it.
        err(std::string("trt_infer_alloc: '") + name + "' requests " + std::to_string(ndims)
                + " dims, but TensorRT supports at most " + std::to_string(nvinfer1::Dims::MAX_DIMS));
        return TRT_ERR_GENERIC;
    }

    auto it = h->tensors.find(name);
    if (it == h->tensors.end())
    {
        err(std::string("unknown tensor: ") + name);
        return TRT_ERR_NOT_FOUND;
    }
    TensorBuf& tb = it->second;
    if (tb.device)
    {
        err(std::string("tensor '") + name + "' already allocated");
        return TRT_ERR_GENERIC;
    }

    // Buffers are flat: what matters for safety is the element count. The
    // engine reads/writes exactly engine_numel elements, so a requested shape
    // with a different element count would over/under-run the allocation.
    // Inputs additionally get their exact shape validated by setInputShape
    // below, and outputs are re-checked against the engine-resolved shape at
    // infer time (trt_infer_infer). Flattened shapes with an equal element
    // count stay allowed (e.g. [B,T,2] allocated as [B*T,2]).
    auto is_dynamic = [](const std::vector<int64_t>& d) {
        return std::any_of(d.begin(), d.end(), [](int64_t v) { return v < 0; });
    };
    for (int32_t j = 0; j < ndims; ++j)
    {
        if (dims[j] < 0)
        {
            err(std::string("dim ") + std::to_string(j) + " of '" + name
                    + "' must be a positive extent, got " + std::to_string(dims[j]));
            return TRT_ERR_GENERIC;
        }
    }
    const std::vector<int64_t> req(dims, dims + ndims);
    bool req_ovf = false;
    const size_t req_numel = shape_numel(req, req_ovf);
    if (req_ovf)
    {
        err(std::string("element count overflows for '") + name + "': shape " + dims_str(req));
        return TRT_ERR_GENERIC;
    }
    if (req_numel == 0)
    {
        err(std::string("tensor '") + name + "' has a zero extent in shape " + dims_str(req)
                + "; cannot allocate an empty buffer");
        return TRT_ERR_GENERIC;
    }
    if (!is_dynamic(tb.dims))
    {
        bool eng_ovf = false;
        const size_t eng_numel = shape_numel(tb.dims, eng_ovf);
        if (eng_ovf || eng_numel != req_numel)
        {
            err(std::string("shape mismatch for '") + name + "': engine " + dims_str(tb.dims) + " ("
                + (eng_ovf ? std::string("overflow") : std::to_string(eng_numel)) + " elements) vs requested "
                + dims_str(req) + " (" + std::to_string(req_numel) + " elements)");
            return TRT_ERR_GENERIC;
        }
    }

    // A wrapped-around byte size would allocate a tiny buffer that the engine
    // then writes past, so reject the overflow explicitly.
    const size_t esz = elem_size(tb.dtype);
    if (req_numel > SIZE_MAX / esz)
    {
        err(std::string("byte size overflows for '") + name + "': shape " + dims_str(req));
        return TRT_ERR_GENERIC;
    }
    const size_t nbytes = req_numel * esz;

    // Snapshot the pre-alloc metadata: the fields are assigned below before
    // setTensorAddress/setInputShape can still fail, and a failed allocation
    // must leave the tensor in its pre-alloc state.
    const std::vector<int64_t> engine_dims = tb.dims;
    const int32_t engine_ndims = tb.ndims;
    auto release = [&]() {
        cudaFree(tb.device);
        cudaFreeHost(tb.pinned);
        tb.device = tb.pinned = nullptr;
        tb.bytes = 0;
        tb.ndims = engine_ndims;
        tb.dims = engine_dims;
    };

    if (cudaHostAlloc(&tb.pinned, nbytes, cudaHostAllocDefault) != cudaSuccess)
    {
        err(std::string("cudaHostAlloc failed: ") + cudaGetErrorString(cudaGetLastError()));
        return TRT_ERR_CUDA;
    }
    if (cudaMalloc(&tb.device, nbytes) != cudaSuccess)
    {
        err(std::string("cudaMalloc failed: ") + cudaGetErrorString(cudaGetLastError()));
        cudaFreeHost(tb.pinned);
        tb.pinned = nullptr;
        return TRT_ERR_CUDA;
    }

    tb.bytes = static_cast<int64_t>(nbytes);
    tb.ndims = ndims;
    tb.dims = req;

    if (!h->context->setTensorAddress(name, tb.device))
    {
        err(std::string("setTensorAddress failed for ") + name);
        release();
        return TRT_ERR_GENERIC;
    }

    if (tb.mode == kTrtIoInput)
    {
        nvinfer1::Dims d;
        d.nbDims = ndims;
        for (int32_t j = 0; j < ndims; ++j)
            d.d[j] = dims[j];
        if (!h->context->setInputShape(name, d))
        {
            err(std::string("setInputShape failed for ") + name);
            release();
            return TRT_ERR_GENERIC;
        }
    }

    g_err.clear();
    return TRT_OK;
}

void* trt_infer_pinned_ptr(TrtInfer* h, const char* name)
{
    if (!h || !name)
        return nullptr;
    auto it = h->tensors.find(name);
    if (it == h->tensors.end())
        return nullptr;
    return it->second.pinned;
}

int64_t trt_infer_byte_size(TrtInfer* h, const char* name)
{
    if (!h || !name)
        return 0;
    auto it = h->tensors.find(name);
    if (it == h->tensors.end())
        return 0;
    return it->second.bytes;
}

int32_t trt_infer_infer(TrtInfer* h)
{
    if (!h)
        return TRT_ERR_GENERIC;

    g_err.clear();
    cudaStream_t s = h->stream;

    for (const auto& name : h->io_names)
    {
        const TensorBuf& tb = h->tensors.at(name);
        if (!tb.device)
            continue;  // not allocated; skip
        if (tb.mode == kTrtIoInput)
        {
            if (cudaMemcpyAsync(tb.device, tb.pinned, tb.bytes, cudaMemcpyHostToDevice, s) != cudaSuccess)
            {
                err(std::string("H2D failed for ") + name + ": " + cudaGetErrorString(cudaGetLastError()));
                return TRT_ERR_CUDA;
            }
        }
    }

    // The engine writes the resolved output shape, so validate every output
    // buffer against it before enqueueing: an under-sized allocation (only
    // possible for dynamic engines, where trt_infer_alloc cannot know the
    // output size up front) would otherwise let the engine write past the
    // buffer. -1 dims mean the shape is not resolved yet; enqueueV3 reports
    // that itself.
    for (const auto& name : h->io_names)
    {
        const TensorBuf& tb = h->tensors.at(name);
        if (!tb.device || tb.mode != kTrtIoOutput)
            continue;
        const nvinfer1::Dims d = h->context->getTensorShape(name.c_str());
        bool resolved = true;
        bool overflow = false;
        size_t numel = 1;
        for (int j = 0; j < d.nbDims; ++j)
        {
            const int64_t v = d.d[j];
            if (v < 0)
            {
                resolved = false;
                break;
            }
            const size_t ext = static_cast<size_t>(v);
            if (ext != 0 && numel > SIZE_MAX / ext)
            {
                overflow = true;
                break;
            }
            numel *= ext;
        }
        if (!resolved)
            continue;
        const size_t esz = elem_size(tb.dtype);
        if (overflow || numel > static_cast<size_t>(tb.bytes) / esz)
        {
            err(std::string("output '") + name + "' resolves to " + std::to_string(numel)
                    + " elements but its buffer only holds "
                    + std::to_string(static_cast<size_t>(tb.bytes) / esz) + "; allocate it with a larger shape");
            return TRT_ERR_GENERIC;
        }
    }

    if (!h->context->enqueueV3(s))
    {
        err("enqueueV3 failed");
        return TRT_ERR_GENERIC;
    }

    for (const auto& name : h->io_names)
    {
        const TensorBuf& tb = h->tensors.at(name);
        if (!tb.device)
            continue;
        if (tb.mode == kTrtIoOutput)
        {
            if (cudaMemcpyAsync(tb.pinned, tb.device, tb.bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess)
            {
                err(std::string("D2H failed for ") + name + ": " + cudaGetErrorString(cudaGetLastError()));
                return TRT_ERR_CUDA;
            }
        }
    }

    if (cudaStreamSynchronize(s) != cudaSuccess)
    {
        err(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(cudaGetLastError()));
        return TRT_ERR_CUDA;
    }

    g_err.clear();
    return TRT_OK;
}

}  // extern "C"
