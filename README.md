# YOLO26 TensorRT

YOLO26 的 TensorRT 10 C++ 部署：CUDA letterbox 预处理、端到端检测、画框可视化。提供 ROS 2 节点和无 ROS 依赖的摄像头 / 视频 demo。

TensorRT 10 C++ deploy for **YOLO26**: CUDA letterbox, end-to-end detect, box visualization. Includes a ROS 2 node and a standalone camera/video demo.

## Features

- TensorRT 10 runtime（`enqueueV3`）
- CUDA warp-affine letterbox（BGR→RGB、`/255`、HWC→CHW）
- 置信度过滤 + letterbox 坐标还原
- `yolo26_trt_node`：订阅 `sensor_msgs/Image`，发布标注图
- `yolo26_trt_demo`：USB 摄像头或视频文件

## Requirements

- Ubuntu 20.04 / 22.04，ROS 2 Humble（或兼容发行版）
- CUDA + TensorRT 10
- OpenCV、`cv_bridge`
- Engine 须在**运行设备**上导出（同一 GPU / JetPack）
- Engine 输出为 rank-3，最后一维为 **6**：`[x1, y1, x2, y2, conf, class_id]`

Jetson Orin 默认 `sm_87`，可用 `-DCMAKE_CUDA_ARCHITECTURES=` 覆盖。x86 默认在 `/usr/local/TensorRT` 查找 TensorRT，可用 `-DTENSORRT_ROOT=` 指定。

## Build

```bash
mkdir -p ~/ws/src
git clone https://github.com/yjunmo/yolo26-tensorrt.git ~/ws/src/yolo26_trt
cd ~/ws
colcon build --packages-select yolo26_trt
source install/setup.bash
```

## Export engine

在目标机器上导出（本仓库不包含权重）：

```bash
pip install ultralytics
yolo export model=yolo26n.pt format=onnx imgsz=640
trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.engine --fp16
```

或使用脚本：

```bash
bash scripts/01_check_jetson.sh
bash scripts/export_engine.sh yolo26n.pt yolo26n.engine 640
```

Engine 需为端到端输出（模型内已完成 NMS），形状 `[1, max_det, 6]`。不要使用仍需宿主机 NMS 的旧版 Ultralytics engine。

## Run

```bash
# ROS 2
ros2 launch yolo26_trt detect.launch.py \
  engine_path:=$HOME/yolo26n.engine \
  image_topic:=/camera/color/image_raw

# 摄像头 demo（q 退出）
ros2 run yolo26_trt yolo26_trt_demo $HOME/yolo26n.engine 0 0.25
```

`yolo26_trt_demo` 用法：`yolo26_trt_demo <engine> [source] [conf]`。`source` 为摄像头编号（默认 `0`）或视频 / 图片路径。

默认标签为 `id:<class_index>`。如需显示类别名，传入 `class_names`：

```bash
ros2 run yolo26_trt yolo26_trt_node --ros-args \
  -p engine_path:=$HOME/yolo26n.engine \
  -p class_names:="['person','bicycle','car']"
```

### ROS 2 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `engine_path` | （必填） | TensorRT engine 路径 |
| `image_topic` | `/camera/color/image_raw` | 输入图像话题 |
| `output_topic` | `/yolo26/image` | 标注图话题 |
| `conf_threshold` | `0.25` | 置信度阈值 |
| `class_names` | `[]` | 类别名列表；为空时显示 `id:<index>` |

## Layout

```
CMakeLists.txt
package.xml
launch/detect.launch.py
src/yolo26.{h,cpp}      # engine + detect()
src/preprocess.{h,cu}   # CUDA letterbox
src/detect_node.cpp     # ROS 2
src/demo.cpp            # OpenCV demo
scripts/
```

## License

[MIT](LICENSE)。请自行准备权重；本仓库不包含模型或数据集。
