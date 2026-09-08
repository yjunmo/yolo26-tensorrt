#pragma once

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

struct LetterboxInfo {
    float scale = 1.0f;
    int padX = 0;
    int padY = 0;
};

struct AffineMatrix {
    float value[6];
};

void cuda_preprocess_init(size_t max_image_size);
void cuda_preprocess_destroy();
LetterboxInfo cuda_preprocess(
    const cv::Mat& img, float* deviceInput, int inputW, int inputH, cudaStream_t stream);
