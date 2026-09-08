# YOLO26 TensorRT

开源 YOLO26 TensorRT 10 部署：GPU letterbox、端到端 engine `[1, max_det, 6]`、画框。提供 ROS 2 节点和摄像头/视频 demo。

这不是业务导航工程。没有类别表、追踪关联或规划。默认显示 `id:<类别下标>`，需要名字时自己传入 `class_names`。

Open-source **YOLO26** deploy on NVIDIA TensorRT 10: GPU letterbox, end-to-end engine `[1, max_det, 6]`, draw boxes. ROS 2 node plus a camera/video demo.

This is **not** a product navigation stack. There is no class taxonomy, tracking association, or planning. Boxes show `id:<class_index>` unless you pass your own `class_names`.

## Features

- TensorRT 10 C++ runtime (`enqueueV3`)
- CUDA warp-affine letterbox (BGR→RGB, `/255`, HWC→CHW)
- Confidence filter + letterbox inverse mapping
- `yolo26_trt_node`: subscribe `sensor_msgs/Image`, publish annotated image
- `yolo26_trt_demo`: USB camera or video file, no ROS required at runtime besides the build

## Requirements

- Ubuntu 20.04 / 22.04, ROS 2 Humble (or compatible)
- CUDA + TensorRT 10
- OpenCV, `cv_bridge`
- Engine **must** be built on the same device / JetPack you run
- Engine output rank-3 with last dim **6**: `[x1, y1, x2, y2, conf, class_id]`

Jetson Orin defaults to `sm_87`. Override `-DCMAKE_CUDA_ARCHITECTURES=` if needed. x86 builds look for TensorRT under `/usr/local/TensorRT` (`-DTENSORRT_ROOT=`).

## Build (ROS 2 workspace)

```bash
mkdir -p ~/ws/src
git clone <this-repo> ~/ws/src/yolo26_trt
cd ~/ws
colcon build --packages-select yolo26_trt
source install/setup.bash
```

## Export engine (on the target)

```bash
pip install ultralytics
# official or your trained YOLO26 weights
yolo export model=yolo26n.pt format=onnx imgsz=640
trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.engine --fp16
```

Helper:

```bash
bash scripts/01_check_jetson.sh
bash scripts/export_engine.sh yolo26n.pt yolo26n.engine 640
```

Do **not** feed a classic Ultralytics engine that still expects host-side NMS.

## Run

```bash
# ROS 2
ros2 launch yolo26_trt detect.launch.py \
  engine_path:=$HOME/yolo26n.engine \
  image_topic:=/camera/color/image_raw

# camera demo (q to quit)
ros2 run yolo26_trt yolo26_trt_demo $HOME/yolo26n.engine 0 0.25
```

Optional names (your list, not shipped here):

```bash
ros2 run yolo26_trt yolo26_trt_node --ros-args \
  -p engine_path:=$HOME/yolo26n.engine \
  -p class_names:="['person','bicycle','car']"
```

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

MIT. Bring your own weights; this repo does not include models or datasets.
