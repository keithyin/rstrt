#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

class TRTInfer
{
public:
    TRTInfer(const std::string &engine_file);
    ~TRTInfer();

    void infer(const std::vector<float> &feat,
               const std::vector<int64_t> &lengths,
               std::vector<float> &output);

    size_t get_output_numel() const { return output_numel_; }

private:
    void load_engine(const std::string &engine_file);

private:
    nvinfer1::IRuntime *runtime_ = nullptr;
    nvinfer1::ICudaEngine *engine_ = nullptr;
    nvinfer1::IExecutionContext *context_ = nullptr;

    cudaStream_t stream_;

    std::string feat_name_;
    std::string lengths_name_;
    std::string output_name_;

    void *feat_device_ = nullptr;
    void *lengths_device_ = nullptr;
    void *output_device_ = nullptr;

    size_t feat_bytes_;
    size_t lengths_bytes_;
    size_t output_bytes_;
    size_t output_numel_;
};