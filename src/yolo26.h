#pragma once

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include "preprocess.h"

struct Object {
    cv::Rect rect;
    int label = 0;
    float prob = 0.f;
};

// YOLO26 端到端 TensorRT 检测：engine 输出须为 [1, max_det, 6]
// 每行 [x1, y1, x2, y2, conf, class_id]（letterbox 输入坐标）。模型内已去重，这里只做置信度过滤和坐标还原。
class Yolo26 {
public:
    explicit Yolo26(const std::string& engine_path);
    ~Yolo26();

    Yolo26(const Yolo26&) = delete;
    Yolo26& operator=(const Yolo26&) = delete;

    bool init();
    std::vector<Object> detect(const cv::Mat& img, float conf_thres = 0.25f);

    int inputWidth() const { return m_input_w; }
    int inputHeight() const { return m_input_h; }

private:
    std::string m_engine_path;
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    cudaStream_t m_stream = nullptr;

    void* m_input_device = nullptr;
    void* m_output_device = nullptr;
    float* m_output_host = nullptr;

    int m_input_w = 640;
    int m_input_h = 640;
    int m_max_det = 300;

    std::string m_input_name;
    std::string m_output_name;
};
