#include "preprocess.h"

#include <opencv2/opencv.hpp>

static uint8_t* g_img_buffer_device = nullptr;
static size_t g_img_buffer_size = 0;

__global__ void warpaffine_kernel(
    const uint8_t* src,
    int srcWidth,
    int srcHeight,
    int srcStride,
    float* dst,
    int dstWidth,
    int dstHeight,
    AffineMatrix d2s,
    float padValue) {
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dstWidth || dy >= dstHeight) {
        return;
    }

    float src_x = d2s.value[0] * dx + d2s.value[1] * dy + d2s.value[2];
    float src_y = d2s.value[3] * dx + d2s.value[4] * dy + d2s.value[5];

    float c0 = padValue;
    float c1 = padValue;
    float c2 = padValue;

    if (src_x > -1 && src_x < srcWidth && src_y > -1 && src_y < srcHeight) {
        int x_low = floorf(src_x);
        int y_low = floorf(src_y);
        int x_high = x_low + 1;
        int y_high = y_low + 1;
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1.f - ly;
        float hx = 1.f - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        auto px = [&](int x, int y, int c) -> float {
            if (x < 0 || x >= srcWidth || y < 0 || y >= srcHeight) {
                return padValue;
            }
            return static_cast<float>(src[y * srcStride + x * 3 + c]);
        };

        c0 = w1 * px(x_low, y_low, 0) + w2 * px(x_high, y_low, 0) +
             w3 * px(x_low, y_high, 0) + w4 * px(x_high, y_high, 0);
        c1 = w1 * px(x_low, y_low, 1) + w2 * px(x_high, y_low, 1) +
             w3 * px(x_low, y_high, 1) + w4 * px(x_high, y_high, 1);
        c2 = w1 * px(x_low, y_low, 2) + w2 * px(x_high, y_low, 2) +
             w3 * px(x_low, y_high, 2) + w4 * px(x_high, y_high, 2);
    }

    c0 /= 255.f;
    c1 /= 255.f;
    c2 /= 255.f;

    const int plane = dstWidth * dstHeight;
    const int idx = dy * dstWidth + dx;
    dst[idx] = c2;
    dst[idx + plane] = c1;
    dst[idx + plane * 2] = c0;
}

void cuda_preprocess_init(size_t max_image_size) {
    if (g_img_buffer_device == nullptr) {
        cudaMalloc(&g_img_buffer_device, max_image_size);
        g_img_buffer_size = max_image_size;
    } else if (max_image_size > g_img_buffer_size) {
        cudaFree(g_img_buffer_device);
        cudaMalloc(&g_img_buffer_device, max_image_size);
        g_img_buffer_size = max_image_size;
    }
}

void cuda_preprocess_destroy() {
    if (g_img_buffer_device != nullptr) {
        cudaFree(g_img_buffer_device);
        g_img_buffer_device = nullptr;
        g_img_buffer_size = 0;
    }
}

LetterboxInfo cuda_preprocess(
    const cv::Mat& img, float* deviceInput, int inputW, int inputH, cudaStream_t stream) {
    LetterboxInfo info;
    if (img.empty()) {
        return info;
    }

    const int srcWidth = img.cols;
    const int srcHeight = img.rows;
    float scale = std::min(
        static_cast<float>(inputW) / srcWidth, static_cast<float>(inputH) / srcHeight);
    info.scale = scale;

    int new_unpad_w = static_cast<int>(std::round(srcWidth * scale));
    int new_unpad_h = static_cast<int>(std::round(srcHeight * scale));
    info.padX = (inputW - new_unpad_w) / 2;
    info.padY = (inputH - new_unpad_h) / 2;

    cv::Mat s2d(2, 3, CV_32F);
    s2d.at<float>(0, 0) = scale;
    s2d.at<float>(0, 1) = 0.f;
    s2d.at<float>(0, 2) = static_cast<float>(info.padX);
    s2d.at<float>(1, 0) = 0.f;
    s2d.at<float>(1, 1) = scale;
    s2d.at<float>(1, 2) = static_cast<float>(info.padY);

    cv::Mat d2s;
    cv::invertAffineTransform(s2d, d2s);
    AffineMatrix d2s_matrix;
    d2s_matrix.value[0] = d2s.at<float>(0, 0);
    d2s_matrix.value[1] = d2s.at<float>(0, 1);
    d2s_matrix.value[2] = d2s.at<float>(0, 2);
    d2s_matrix.value[3] = d2s.at<float>(1, 0);
    d2s_matrix.value[4] = d2s.at<float>(1, 1);
    d2s_matrix.value[5] = d2s.at<float>(1, 2);

    const size_t srcBytes = static_cast<size_t>(img.step) * srcHeight;
    uint8_t* deviceImage = nullptr;
    bool need_free = false;
    if (g_img_buffer_device != nullptr && srcBytes <= g_img_buffer_size) {
        deviceImage = g_img_buffer_device;
    } else {
        cudaMalloc(&deviceImage, srcBytes);
        need_free = true;
    }

    cudaMemcpyAsync(deviceImage, img.data, srcBytes, cudaMemcpyHostToDevice, stream);

    dim3 block(32, 32);
    dim3 grid((inputW + block.x - 1) / block.x, (inputH + block.y - 1) / block.y);
    warpaffine_kernel<<<grid, block, 0, stream>>>(
        deviceImage, srcWidth, srcHeight, static_cast<int>(img.step), deviceInput, inputW,
        inputH, d2s_matrix, 114.f);

    if (need_free) {
        cudaFree(deviceImage);
    }
    return info;
}
