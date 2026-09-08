#!/usr/bin/env bash
# 在 Jetson 上检查部署 YOLO / TensorRT 还缺什么。
set -euo pipefail

ok() { printf '  [OK] %s\n' "$1"; }
warn() { printf '  [!!] %s\n' "$1"; }
info() { printf '  [--] %s\n' "$1"; }

echo "=== Jetson YOLO 环境检查 ==="

if [[ -f /etc/nv_tegra_release ]]; then
  ok "这是 Jetson：$(head -n1 /etc/nv_tegra_release)"
else
  warn "没找到 /etc/nv_tegra_release，当前多半不是 Jetson。engine 必须在目标板上导出。"
fi

if command -v jetson_release >/dev/null 2>&1; then
  jetson_release -v || true
elif [[ -f /etc/nv_tegra_release ]]; then
  info "未安装 jetson-stats。可选：sudo pip3 install -U jetson-stats"
fi

if command -v nvpmodel >/dev/null 2>&1; then
  info "电源模式：$(nvpmodel -q 2>/dev/null | tr '\n' ' ' || echo unknown)"
else
  warn "没有 nvpmodel（非 Jetson 或精简镜像）"
fi

echo
echo "--- 磁盘 / 内存 ---"
df -h / | tail -n1
free -h | head -n2

echo
echo "--- Python ---"
if command -v python3 >/dev/null 2>&1; then
  ok "$(python3 --version 2>&1)"
  python3 - <<'PY' || true
import sys
mods = ["cv2", "numpy", "torch", "ultralytics"]
for m in mods:
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK] {m} {ver}")
    except Exception as e:
        print(f"  [!!] {m} 未安装 ({e.__class__.__name__})")
PY
else
  warn "没有 python3"
fi

echo
echo "--- TensorRT / CUDA ---"
python3 - <<'PY' || true
try:
    import tensorrt as trt
    print(f"  [OK] tensorrt {trt.__version__}")
except Exception:
    print("  [!!] Python 导入不了 tensorrt。可用: dpkg -l | grep tensorrt")
PY
if command -v nvcc >/dev/null 2>&1; then
  ok "nvcc $(nvcc --version | tail -n1)"
else
  info "没有 nvcc（很多推理场景不需要，缺编译时再装）"
fi

echo
echo "--- 摄像头节点 ---"
if ls /dev/video* >/dev/null 2>&1; then
  ok "找到：$(ls /dev/video* | tr '\n' ' ')"
else
  warn "没有 /dev/video*。USB 相机没插，或 CSI 要走厂商驱动。"
fi

echo
echo "检查结束。下一步：python3 scripts/02_export_tensorrt.py --help"
