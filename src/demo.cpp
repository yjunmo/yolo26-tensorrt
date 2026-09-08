#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "yolo26.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: yolo26_trt_demo <engine> [source] [conf]\n"
                  << "  source: camera index (default 0) or video/image path\n";
        return 1;
    }

    const std::string engine = argv[1];
    const std::string source = argc >= 3 ? argv[2] : "0";
    const float conf = argc >= 4 ? std::stof(argv[3]) : 0.25f;

    Yolo26 detector(engine);
    if (!detector.init()) {
        std::cerr << "failed to load engine: " << engine << std::endl;
        return 1;
    }

    cv::VideoCapture cap;
    if (source.size() == 1 && source[0] >= '0' && source[0] <= '9') {
        cap.open(std::stoi(source));
    } else {
        cap.open(source);
    }
    if (!cap.isOpened()) {
        std::cerr << "cannot open source: " << source << std::endl;
        return 1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        const auto objs = detector.detect(frame, conf);
        for (const auto& obj : objs) {
            cv::rectangle(frame, obj.rect, cv::Scalar(0, 255, 0), 2);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "id:%d %.2f", obj.label, obj.prob);
            cv::putText(
                frame, buf, cv::Point(obj.rect.x, std::max(0, obj.rect.y - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        cv::imshow("yolo26_trt", frame);
        if ((cv::waitKey(1) & 0xFF) == 'q') {
            break;
        }
    }
    return 0;
}
