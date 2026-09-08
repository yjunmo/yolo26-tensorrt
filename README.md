# Jetson YOLO 部署包

把 Ultralytics YOLO 权重部署到 NVIDIA Jetson（Orin Nano / Orin NX / Xavier），并可选接入 ROS 2。

本包是**原创脚本 + 排错步骤**，不含第三方商业权重，不含破解软件。你需要自备 `.pt` 模型（官方 `yolov8n.pt` 或自己训练的权重均可）。

## 本仓库明确不包含

这是通用 Jetson / Ultralytics 部署示例，**不是**任何公司产品的开源版。因此不会出现、也不会后续加入：

- 业务模型的类别表、中文映射、按类置信度
- 检测/分割结果之后的导航、关联、状态机、BEV 路径等处理
- 内部权重、engine、数据集、启动业务用的 launch/config

C++ / Python 节点都只画框并发布带框图。类别名以你自己的权重为准（COCO 或自训均可），本仓库不维护一份业务类别清单。默认框上写的是 `id:数字`；若要显示名字，自己在 launch 里传入 `class_names`。

## 你能得到什么

| 文件 | 作用 |
|------|------|
| `scripts/01_check_jetson.sh` | 检查 JetPack、CUDA、TensorRT、摄像头 |
| `scripts/02_export_tensorrt.py` | `.pt` → TensorRT engine（FP16 / INT8） |
| `scripts/03_bench_infer.py` | 测真实 FPS，不拿桌面 GPU 的数字糊弄 |
| `scripts/04_usb_camera_detect.py` | USB 摄像头实时画框 |
| `ros2/yolo_edge_kit/` | ROS 2 Python 节点：订图像、发检测图 |
| `ros2/yolo26_trt/` | YOLO26 TensorRT C++ 节点：GPU letterbox + `[1,max_det,6]` 端到端推理，只画框 |
| `排错手册.md` | Jetson 上最常见的 12 个坑 |

## 环境

- Ubuntu 20.04 / 22.04
- JetPack 5.x 或 6.x（engine 必须在**同一块板、同一 JetPack**上导出）
- Python 3.8+
- `pip install ultralytics opencv-python-headless`

ROS 2 节点额外需要：Humble 或 Iron、`cv_bridge`、`sensor_msgs`。

## 10 分钟跑通

在 **Jetson 本机**执行（不要在 x86 电脑上导出 engine 再拷过去）：

```bash
# 1. 看环境缺什么
bash scripts/01_check_jetson.sh

# 2. 导出 FP16 engine（把 your.pt 换成你的权重）
python3 scripts/02_export_tensorrt.py --weights yolov8n.pt --imgsz 640 --half

# 3. 测速
python3 scripts/03_bench_infer.py --weights yolov8n.engine --imgsz 640

# 4. 摄像头
python3 scripts/04_usb_camera_detect.py --weights yolov8n.engine --source 0
```

ROS 2：

```bash
cd ros2
# 把 yolo_edge_kit 拷进你的 workspace/src 后：
colcon build --packages-select yolo_edge_kit
source install/setup.bash
ros2 launch yolo_edge_kit detect.launch.py \
  weights:=$HOME/yolov8n.engine \
  image_topic:=/camera/color/image_raw
```

YOLO26 TensorRT C++（engine 必须是端到端 `[1, max_det, 6]`，不要把 Ultralytics 旧版带 NMS 的 engine 塞进来）：

```bash
yolo export model=yolo26n.pt format=onnx imgsz=640
trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.engine --fp16

# 把 ros2/yolo26_trt 拷进 workspace/src 后
colcon build --packages-select yolo26_trt
source install/setup.bash
ros2 launch yolo26_trt detect.launch.py \
  engine_path:=$HOME/yolo26n.engine \
  image_topic:=/camera/color/image_raw
```

## 验收标准（建议你按这个跟卖家/自己核对）

- `01_check_jetson.sh` 能打印 JetPack / TensorRT 版本
- `03_bench_infer.py` 在 Orin Nano 640 输入、YOLOv8n FP16 下，常见结果大约十几到几十 FPS（取决于散热和 nvpmodel）
- 摄像头窗口或 ROS 话题 `/yolo/image` 能看到框

## 许可

个人学习与项目使用。禁止把本包改头换面再去闲鱼/网盘二次售卖。
