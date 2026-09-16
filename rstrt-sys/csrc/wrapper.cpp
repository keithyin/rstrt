#include "wrapper.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

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

const char* err(const std::string& msg)
{
    g_err = msg;
    return g_err.c_str();
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
        size_t size = static_cast<size_t>(std::ftell(f));
        std::fseek(f, 0, SEEK_SET);
        data.resize(size);
        if (size && std::fread(data.data(), 1, size, f) != size)
        {
            std::fclose(f);
            err("failed to read engine file");
            return nullptr;
        }
        std::fclose(f);
        out_size = size;
    }
    void* buf = std::malloc(out_size);
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
        delete h;
        std::free(blob);
        return nullptr;
    }

    h->engine = h->runtime->deserializeCudaEngine(blob, size);
    std::free(blob);
    if (!h->engine)
    {
        err("failed to deserialize engine");
        delete h;
        return nullptr;
    }

    h->context = h->engine->createExecutionContext();
    if (!h->context)
    {
        err("failed to create execution context");
        delete h;
        return nullptr;
    }

    if (cudaStreamCreate(&h->stream) != cudaSuccess)
    {
        err(std::string("cudaStreamCreate failed: ") + cudaGetErrorString(cudaGetLastError()));
        delete h;
        return nullptr;
    }

    if (!h->context->setOptimizationProfileAsync(0, h->stream))
    {
        err("failed to set optimization profile");
        delete h;
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
    if (h->stream)
        cudaStreamDestroy(h->stream);
    delete h->context;
    delete h->engine;
    delete h->runtime;
    delete h;
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
        return "";
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
        return TRT_ERR_GENERIC;

    auto it = h->tensors.find(name);
    if (it == h->tensors.end())
    {
        err(std::string("unknown tensor: ") + name);
        return TRT_ERR_NOT_FOUND;
    }
    TensorBuf& tb = it->second;
    if (tb.device)
        return TRT_OK;  // already allocated

    size_t numel = 1;
    for (int32_t j = 0; j < ndims; ++j)
        numel *= static_cast<size_t>(dims[j] < 0 ? 1 : dims[j]);
    size_t nbytes = numel * elem_size(tb.dtype);

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
    tb.dims.assign(dims, dims + ndims);

    if (!h->context->setTensorAddress(name, tb.device))
    {
        err(std::string("setTensorAddress failed for ") + name);
        cudaFree(tb.device);
        cudaFreeHost(tb.pinned);
        tb.device = tb.pinned = nullptr;
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
            cudaFree(tb.device);
            cudaFreeHost(tb.pinned);
            tb.device = tb.pinned = nullptr;
            return TRT_ERR_GENERIC;
        }
    }

    g_err.clear();
    return TRT_OK;
}

uintptr_t trt_infer_pinned_ptr(TrtInfer* h, const char* name)
{
    if (!h || !name)
        return 0;
    auto it = h->tensors.find(name);
    if (it == h->tensors.end())
        return 0;
    return reinterpret_cast<uintptr_t>(it->second.pinned);
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
