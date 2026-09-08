#!/usr/bin/env bash
# Check Jetson / TensorRT / CUDA for YOLO26 engine deploy.
set -euo pipefail

ok() { printf '  [OK] %s\n' "$1"; }
warn() { printf '  [!!] %s\n' "$1"; }
info() { printf '  [--] %s\n' "$1"; }

echo "=== YOLO26 TensorRT environment ==="

if [[ -f /etc/nv_tegra_release ]]; then
  ok "Jetson: $(head -n1 /etc/nv_tegra_release)"
else
  warn "not a Jetson host; build the engine on the same GPU/JetPack you will run"
fi

if command -v nvpmodel >/dev/null 2>&1; then
  info "nvpmodel: $(nvpmodel -q 2>/dev/null | tr '\n' ' ' || echo unknown)"
fi

echo
echo "--- disk / mem ---"
df -h / | tail -n1
free -h | head -n2

echo
echo "--- compilers ---"
if command -v nvcc >/dev/null 2>&1; then
  ok "nvcc $(nvcc --version | tail -n1)"
else
  warn "nvcc missing"
fi

TRTEXEC=""
if command -v trtexec >/dev/null 2>&1; then
  TRTEXEC=$(command -v trtexec)
elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
if [[ -n "$TRTEXEC" ]]; then
  ok "trtexec: $TRTEXEC"
else
  warn "trtexec not found"
fi

python3 - <<'PY' || true
mods = ["cv2", "numpy"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"  [OK] {m} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  [!!] {m} ({e.__class__.__name__})")
try:
    import tensorrt as trt
    print(f"  [OK] tensorrt {trt.__version__}")
except Exception:
    print("  [--] python tensorrt module optional")
PY

echo
echo "--- cameras ---"
if ls /dev/video* >/dev/null 2>&1; then
  ok "$(ls /dev/video* | tr '\n' ' ')"
else
  info "no /dev/video*"
fi

echo "next: bash scripts/export_engine.sh yolo26n.pt"
