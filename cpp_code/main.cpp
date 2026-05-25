#include "trt_infer.h"
#include <iostream>
#include <vector>
#include <chrono>

int main()
{
    TRTInfer infer("/root/projects/rstrt/2025Q1-stage2-selfattn-2o-onnx/model.fp16.plan");

    int B = 128, T = 200, F = 61;

    std::vector<float> feat(B * T * F, 0.0f);
    std::vector<int64_t> lengths(B, 200);

    size_t out_size = infer.get_output_numel();
    std::vector<float> output(out_size);

    float result = 0.0;

    // ===== warmup =====
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 20; ++i)
    {
        infer.infer(feat, lengths, output);
        result += output[0];
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "TRT warmup: " << result
              << " secs: "
              << std::chrono::duration<float>(t1 - t0).count()
              << std::endl;

    result = 0.0;

    // ===== benchmark =====
    t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i)
    {
        std::vector<float> feat(B * T * F, 0.0f);
        std::vector<int64_t> lengths(B, 200);
        std::vector<float> output(out_size);

        infer.infer(feat, lengths, output);
        result += output[0];
    }

    t1 = std::chrono::high_resolution_clock::now();

    std::cout << "TRT result: " << result
              << " secs: "
              << std::chrono::duration<float>(t1 - t0).count()
              << std::endl;

    return 0;
}