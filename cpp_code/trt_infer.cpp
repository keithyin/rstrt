#include "trt_infer.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace nvinfer1;

class Logger : public ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

TRTInfer::TRTInfer(const std::string &engine_file)
{
    load_engine(engine_file);

    cudaStreamCreate(&stream_);

    if (!context_->setOptimizationProfileAsync(0, stream_))
    {
        throw std::runtime_error("Failed to set optimization profile!");
    }

    int nb = engine_->getNbIOTensors();

    for (int i = 0; i < nb; ++i)
    {
        const char *name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dtype = engine_->getTensorDataType(name);
        bool is_shape = engine_->isShapeInferenceIO(name);

        std::cout << "Tensor: " << name
                  << " | mode=" << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
                  << " | dtype=" << (int)dtype
                  << " | is_shape=" << is_shape
                  << std::endl;

        if (mode == TensorIOMode::kINPUT)
        {
            if (feat_name_.empty())
            {
                feat_name_ = name;
            }
            else
            {
                lengths_name_ = name;
            }
        }
        else
        {
            output_name_ = name;
        }
    }

    int nbProfiles = engine_->getNbOptimizationProfiles();
    std::cout << "Profiles: " << nbProfiles << std::endl;

    for (int i = 0; i < nbProfiles; ++i)
    {
        auto min_dims = engine_->getProfileShape(feat_name_.c_str(), i, nvinfer1::OptProfileSelector::kMIN);
        auto opt_dims = engine_->getProfileShape(feat_name_.c_str(), i, nvinfer1::OptProfileSelector::kOPT);
        auto max_dims = engine_->getProfileShape(feat_name_.c_str(), i, nvinfer1::OptProfileSelector::kMAX);

        auto print_dims = [](const nvinfer1::Dims &d)
        {
            for (int j = 0; j < d.nbDims; ++j)
                std::cout << d.d[j] << " ";
        };

        std::cout << "[Profile " << i << "] MIN: ";
        print_dims(min_dims);
        std::cout << " OPT: ";
        print_dims(opt_dims);
        std::cout << " MAX: ";
        print_dims(max_dims);
        std::cout << std::endl;
    }

    if (feat_name_.empty() || lengths_name_.empty() || output_name_.empty())
    {
        throw std::runtime_error("Tensor names not properly detected!");
    }

    // dtype 打印
    std::cout << "feat dtype: "
              << (int)engine_->getTensorDataType(feat_name_.c_str()) << std::endl;
    std::cout << "length dtype: "
              << (int)engine_->getTensorDataType(lengths_name_.c_str()) << std::endl;

    int B = 128, T = 200, F = 61;

    feat_bytes_ = B * T * F * sizeof(float);
    lengths_bytes_ = B * sizeof(int64_t);

    cudaMalloc(&feat_device_, feat_bytes_);
    cudaMalloc(&lengths_device_, lengths_bytes_);

    // ⭐ 关键：初始化 lengths（shape tensor 必须有值）
    std::vector<int64_t> init_lengths(B, 200);
    cudaMemcpy(lengths_device_, init_lengths.data(), lengths_bytes_, cudaMemcpyHostToDevice);

    // 先给一个临时 output
    output_bytes_ = 1024 * sizeof(float);
    cudaMalloc(&output_device_, output_bytes_);

    // ⭐ 先绑定 tensor
    context_->setTensorAddress(feat_name_.c_str(), feat_device_);
    context_->setTensorAddress(lengths_name_.c_str(), lengths_device_);
    context_->setTensorAddress(output_name_.c_str(), output_device_);

    // ⭐ 再设 shape（只对 feature）
    Dims feat_dims{3, {B, T, F}};
    context_->setInputShape(feat_name_.c_str(), feat_dims);

    auto feat_after = context_->getTensorShape(feat_name_.c_str());
    std::cout << "After setInputShape (feature): ";
    for (int i = 0; i < feat_after.nbDims; ++i)
        std::cout << feat_after.d[i] << " ";
    std::cout << std::endl;

    std::cout << "=== Before inferShapes ===" << std::endl;

    for (int i = 0; i < engine_->getNbIOTensors(); ++i)
    {
        const char *name = engine_->getIOTensorName(i);

        auto dims = context_->getTensorShape(name);

        std::cout << name << ": ";
        for (int j = 0; j < dims.nbDims; ++j)
        {
            std::cout << dims.d[j] << " ";
        }
        std::cout << std::endl;
    }

    // ❗不要对 shape tensor 调 setInputShape

    // if (!context_->inferShapes(0, nullptr))
    // {
    //     throw std::runtime_error("inferShapes failed!");
    // }

    auto out_dims = context_->getTensorShape(output_name_.c_str());

    std::cout << "Output dims: ";
    for (int i = 0; i < out_dims.nbDims; ++i)
    {
        std::cout << out_dims.d[i] << " ";
    }
    std::cout << std::endl;

    output_numel_ = 1;
    for (int i = 0; i < out_dims.nbDims; ++i)
    {
        if (out_dims.d[i] < 0)
        {
            throw std::runtime_error("Output shape not resolved!");
        }
        output_numel_ *= out_dims.d[i];
    }

    cudaFree(output_device_);
    output_bytes_ = output_numel_ * sizeof(float);
    cudaMalloc(&output_device_, output_bytes_);

    context_->setTensorAddress(output_name_.c_str(), output_device_);
}

TRTInfer::~TRTInfer()
{
    cudaStreamDestroy(stream_);

    cudaFree(feat_device_);
    cudaFree(lengths_device_);
    cudaFree(output_device_);

    delete context_;
    delete engine_;
    delete runtime_;
}

void TRTInfer::load_engine(const std::string &engine_file)
{
    std::ifstream file(engine_file, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open engine file!");
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    runtime_ = createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    context_ = engine_->createExecutionContext();

    if (!context_)
    {
        throw std::runtime_error("Failed to create execution context!");
    }
}

void TRTInfer::infer(const std::vector<float> &feat,
                     const std::vector<int64_t> &lengths,
                     std::vector<float> &output)
{

    cudaMemcpyAsync(feat_device_, feat.data(), feat_bytes_,
                    cudaMemcpyHostToDevice, stream_);

    // ⭐ shape tensor 每次也要更新
    cudaMemcpyAsync(lengths_device_, lengths.data(), lengths_bytes_,
                    cudaMemcpyHostToDevice, stream_);

    context_->enqueueV3(stream_);

    cudaMemcpyAsync(output.data(), output_device_, output_bytes_,
                    cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);
}