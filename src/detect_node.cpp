#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "yolo26.h"

class Yolo26DetectNode : public rclcpp::Node {
public:
    Yolo26DetectNode() : Node("yolo26_trt") {
        const std::string engine_path = declare_parameter<std::string>("engine_path", "");
        conf_thres_ = static_cast<float>(declare_parameter<double>("conf_threshold", 0.25));
        const std::string image_topic =
            declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
        const std::string output_topic =
            declare_parameter<std::string>("output_topic", "/yolo26/image");
        class_names_ = declare_parameter<std::vector<std::string>>("class_names", {});

        if (engine_path.empty()) {
            throw std::runtime_error("engine_path is empty");
        }
        detector_ = std::make_unique<Yolo26>(engine_path);
        if (!detector_->init()) {
            throw std::runtime_error("failed to load engine: " + engine_path);
        }

        pub_ = create_publisher<sensor_msgs::msg::Image>(output_topic, 10);
        sub_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic, rclcpp::SensorDataQoS(),
            std::bind(&Yolo26DetectNode::onImage, this, std::placeholders::_1));
        RCLCPP_INFO(
            get_logger(), "engine=%s  sub=%s  pub=%s", engine_path.c_str(), image_topic.c_str(),
            output_topic.c_str());
    }

private:
    void onImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        cv::Mat bgr;
        try {
            bgr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
            return;
        }

        const auto objs = detector_->detect(bgr, conf_thres_);
        cv::Mat vis = bgr;
        for (const auto& obj : objs) {
            cv::rectangle(vis, obj.rect, cv::Scalar(0, 255, 0), 2);
            std::string label = "id:" + std::to_string(obj.label);
            if (obj.label >= 0 && obj.label < static_cast<int>(class_names_.size())) {
                label = class_names_[obj.label];
            }
            char buf[64];
            std::snprintf(buf, sizeof(buf), " %.2f", obj.prob);
            label += buf;
            cv::putText(
                vis, label, cv::Point(obj.rect.x, std::max(0, obj.rect.y - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        auto out = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, vis).toImageMsg();
        pub_->publish(*out);
    }

    std::unique_ptr<Yolo26> detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    std::vector<std::string> class_names_;
    float conf_thres_ = 0.25f;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<Yolo26DetectNode>());
    } catch (const std::exception& e) {
        fprintf(stderr, "yolo26_trt failed: %s\n", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
