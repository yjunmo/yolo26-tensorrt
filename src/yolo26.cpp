#include "yolo26.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

using namespace nvinfer1;

namespace {

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

}  // namespace

Yolo26::Yolo26(const std::string& engine_path) : m_engine_path(engine_path) {}

Yolo26::~Yolo26() {
    if (m_stream) {
        cudaStreamDestroy(m_stream);
    }
    if (m_input_device) {
        cudaFree(m_input_device);
    }
    if (m_output_device) {
        cudaFree(m_output_device);
    }
    if (m_output_host) {
        cudaFreeHost(m_output_host);
    }
    cuda_preprocess_destroy();
}

bool Yolo26::init() {
    std::ifstream file(m_engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "cannot read engine: " << m_engine_path << std::endl;
        return false;
    }
    file.seekg(0, file.end);
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), static_cast<std::streamsize>(size));
    file.close();

    m_runtime.reset(createInferRuntime(gLogger));
    if (!m_runtime) {
        return false;
    }
    m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), size));
    if (!m_engine) {
        return false;
    }
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* name = m_engine->getIOTensorName(i);
        if (m_engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            m_input_name = name;
        } else {
            m_output_name = name;
        }
    }
    if (m_input_name.empty() || m_output_name.empty()) {
        std::cerr << "engine missing input/output tensor" << std::endl;
        return false;
    }

    auto input_dims = m_engine->getTensorShape(m_input_name.c_str());
    m_input_h = input_dims.d[2];
    m_input_w = input_dims.d[3];

    auto output_dims = m_engine->getTensorShape(m_output_name.c_str());
    if (output_dims.nbDims != 3 || output_dims.d[2] != 6) {
        std::cerr << "unexpected output shape, want [1, max_det, 6], got:";
        for (int i = 0; i < output_dims.nbDims; ++i) {
            std::cerr << " " << output_dims.d[i];
        }
        std::cerr << std::endl;
        return false;
    }
    m_max_det = output_dims.d[1];

    std::cout << "[Yolo26] input " << m_input_w << "x" << m_input_h << " max_det=" << m_max_det
              << std::endl;

    const size_t input_bytes = 1ull * 3 * m_input_h * m_input_w * sizeof(float);
    const size_t output_bytes = 1ull * m_max_det * 6 * sizeof(float);
    if (cudaMalloc(&m_input_device, input_bytes) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&m_output_device, output_bytes) != cudaSuccess) {
        return false;
    }
    if (cudaMallocHost(reinterpret_cast<void**>(&m_output_host), output_bytes) != cudaSuccess) {
        return false;
    }

    m_context->setInputTensorAddress(m_input_name.c_str(), m_input_device);
    m_context->setOutputTensorAddress(m_output_name.c_str(), m_output_device);
    cudaStreamCreate(&m_stream);
    cuda_preprocess_init(1920 * 1080 * 3);
    return true;
}

std::vector<Object> Yolo26::detect(const cv::Mat& img, float conf_thres) {
    if (img.empty()) {
        return {};
    }

    LetterboxInfo info = cuda_preprocess(
        img, static_cast<float*>(m_input_device), m_input_w, m_input_h, m_stream);
    m_context->enqueueV3(m_stream);

    const size_t output_bytes = 1ull * m_max_det * 6 * sizeof(float);
    cudaMemcpyAsync(
        m_output_host, m_output_device, output_bytes, cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    const float inv_scale = info.scale > 0.f ? 1.f / info.scale : 1.f;
    const int orig_w = img.cols;
    const int orig_h = img.rows;

    std::vector<Object> objs;
    for (int i = 0; i < m_max_det; ++i) {
        const float* det = m_output_host + i * 6;
        const float conf = det[4];
        if (conf < conf_thres) {
            continue;
        }

        float x1 = (det[0] - info.padX) * inv_scale;
        float y1 = (det[1] - info.padY) * inv_scale;
        float x2 = (det[2] - info.padX) * inv_scale;
        float y2 = (det[3] - info.padY) * inv_scale;

        int left = std::max(0, std::min(static_cast<int>(std::floor(x1)), orig_w - 1));
        int top = std::max(0, std::min(static_cast<int>(std::floor(y1)), orig_h - 1));
        int right = std::max(left + 1, std::min(static_cast<int>(std::ceil(x2)), orig_w));
        int bottom = std::max(top + 1, std::min(static_cast<int>(std::ceil(y2)), orig_h));

        Object obj;
        obj.rect = cv::Rect(left, top, right - left, bottom - top);
        obj.label = static_cast<int>(det[5]);
        obj.prob = conf;
        objs.push_back(obj);
    }
    return objs;
}
