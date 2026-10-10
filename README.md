# 端边协同双分辨率图像传输

> 端侧只传「低分背景 + 高清 ROI」，边侧还原重建——用更少的带宽，尽量保住目标区域画质。

## 一、这是什么

图像/视频链路里，整张高清图直传非常吃带宽，而真正有价值的信息往往只集中在少数目标区域（ROI，Region of Interest）。本项目把一张图拆成两部分分别传输：

- **低分辨率背景**：整图按比例降采样后传输，数据量很小；
- **高清 ROI 块**：只用 YOLO 检测出的目标区域，按**原始分辨率**单独裁剪传输。

边侧收到后，把低分背景上采样回原尺寸，再把高清 ROI 块贴回原位并做边缘融合，最终在**显著减少传输字节**的前提下，让目标区域依然清晰。

## 二、核心思路

```
原图 (1024×1024)
        │
        ├── 整图降采样 ×0.25  ─────────────►  低分背景 (256×256)   ← 数据量约 1/16
        │
        └── YOLO 检测目标 ──► 裁剪 ROI 高清块 ─►  高清 ROI 块 (原始分辨率)
                                          │
                              ┌───────────┴───────────┐
                              │   压缩(JPEG/PNG)打包    │
                              │   TCP 传输到边侧        │
                              └───────────┬───────────┘
                                          ▼
                                 边侧：背景上采样 + ROI 贴回 + 融合
                                          ▼
                                    重建图 + 评估指标
```

关键点：

- 背景用低质量 JPEG，ROI 块用高质量 JPEG，Mask 用无损 PNG；
- 只传"糊背景 + 清晰的目标小块"，而不是整张高清图；
- 边侧可以选 Bicubic 插值或 **Real-ESRGAN 超分**来放大背景，再用**高斯羽化**或**泊松融合**把 ROI 无缝贴回。

## 三、系统架构与分工

| | 端侧（算力弱） | 边侧（算力强） |
|---|---|---|
| 角色 | 读图 → 检测 → 双分辨率编码 → 发送 | 接收 → 解码重建 → 评估 → 回传 |
| 动作 | YOLO 检测、降采样、裁剪、压缩、TCP 发送 | 上采样、贴回、融合、回传重建图 |
| 依赖 | 轻量（YOLO + OpenCV） | 较重（可含 PyTorch / Real-ESRGAN） |

```
┌──────────────────────┐   TCP 9999    ┌──────────────────────┐
│  端侧 edge_sender.py  │ ═══════════►  │  边侧 edge_receiver.py│
│  ① 读图  ② YOLO 检测  │               │  ⑤ 接收  ⑥ 解包       │
│  ③ 双分辨率编码       │               │  ⑦ 上采样 + ROI 贴回   │
│  ④ 压缩打包 + 发送    │ ◄═══════════  │  ⑧ 回传指标 + 重建图   │
│  ⑨ 收重建图 ⑩ 存对比图 │               │                      │
└──────────────────────┘               └──────────────────────┘
```

## 四、文件说明

| 文件 | 作用 |
|---|---|
| `edge_sender.py` | 端侧发送端：检测 + 编码 + 发送 + 收结果 + 存输出 |
| `edge_receiver.py` | 边侧接收端：监听 + 解码重建 + 回传 |
| `utils.py` | 核心工具：检测、编码、解码、融合、超分、评估 |
| `api_server.py` | 第二版：仅做 ROI 检测的 FastAPI 后端（见第八节） |
| `requirements.txt` | Python 依赖清单 |

> 详细设计文档（环境搭建手册、通信流程详解、第二版前后端对接说明）放在本地 `说明文档/` 目录里，未纳入本仓库。

## 五、环境与依赖

- Python 3.12
- 依赖见 `requirements.txt`：

```
ultralytics      # YOLOv8 检测
opencv-python    # 图像读写 / 编解码 / 融合
numpy
scikit-image     # PSNR / SSIM 评估
pillow
matplotlib
```

- 使用 **Real-ESRGAN 超分**（`--super-res`）时需要额外安装：`torch`、`basicsr`、`realesrgan`（首次运行会自动下载超分权重）。
- 模型权重 `yolov8n.pt` 需放在项目根目录（`utils.py` / `api_server.py` 会按脚本所在目录定位它）。

```bash
pip install -r requirements.txt
```

## 六、快速开始（端边协同）

### 1. 启动边侧接收端（先开）

```bash
python edge_receiver.py --host 0.0.0.0 --port 9999
```

可选参数：

```bash
--super-res              # 用 Real-ESRGAN 超分代替 Bicubic 插值
--blend-mode poisson     # 融合方式：gaussian(默认) / poisson
--blend-width 15         # 高斯羽化宽度（像素）
--data-dir data          # 原图目录（用于评估）
```

### 2. 启动端侧发送端（后发）

```bash
python edge_sender.py --host <边侧IP> --port 9999 \
    --image data/newspaper.jpg --scale 0.25
```

参数说明：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--host` | `127.0.0.1` | 边侧 IP |
| `--port` | `9999` | 端口 |
| `--image` | `data/newspaper.jpg` | 待发送图片路径 |
| `--scale` | `0.25` | 背景下采样比例（越小越省带宽、背景越糊） |
| `--conf` | `0.25` | YOLO 置信度阈值 |
| `--bg-quality` | `30` | 背景 JPEG 质量 |
| `--roi-quality` | `90` | ROI 块 / 基准 JPEG 质量 |

### 3. 查看结果

端侧终端会打印检测数量、各分量字节数、压缩节省比例、PSNR/SSIM、传输耗时与实际带宽；运行产物保存在 `output/` 下（见第七节）。

## 七、输出说明

端侧每处理一张图，会在 `output/{图片名}/` 下生成：

```
output/{图片名}/
├── {name}_original.jpg        原始图
├── {name}_roi_01.jpg ...      每个 ROI 的高清裁剪
├── {name}_annotated.jpg       检测标注图（画框 + 类别/置信度）
├── {name}_hollow_bg.jpg       低分辨率挖空背景（展示实际传输的背景）
├── {name}_reconstructed.jpg   边侧回传的重建图
└── {name}_comparison.jpg      原图 | 重建 并排对比
```

同时会在 `output/results.csv` 追加一行汇总，并同步生成对齐的 `output/results.txt`，字段包括：数据分量字节数（背景/掩码/ROI块/元数据）、基准 JPEG 大小、压缩节省百分比、编解码与传输耗时、实测带宽等。

> `output/`、`data/`、`yolov8n.pt` 均已通过 `.gitignore` 排除，不会提交到仓库。

## 八、第二版：仅 ROI 检测后端（FastAPI）

第二版把"人工鼠标绘制 ROI"替换为"后端 YOLOv8n 自动检测 ROI"，其余双分辨率处理放在浏览器前端完成。此时后端**只负责返回一个 ROI 坐标**，不返回裁剪图、重建图或字节数。

### 接口

```
POST /api/detect-roi
Content-Type: multipart/form-data
字段: image = 图片文件 (JPG / PNG)
```

成功响应：

```json
{
  "roi": { "x": 100, "y": 50, "width": 300, "height": 550 },
  "className": "person",
  "confidence": 0.87,
  "imageWidth": 1024,
  "imageHeight": 1024
}
```

- 选择规则：**置信度最高**；同置信度时取**面积最大**的框；
- 坐标已做边界保护并取整。

错误响应（`{ "error": { "code", "message" } }`）：

| HTTP | code | 含义 |
|---|---|---|
| 415 | `UNSUPPORTED_IMAGE_TYPE` | 非 JPG/PNG |
| 400 | `INVALID_IMAGE` | 图片无法解析 |
| 500 | `INFERENCE_FAILED` | YOLO 推理失败 |
| 422 | `NO_ROI_DETECTED` | 图中未检测到目标 |

### 启动

```bash
pip install fastapi uvicorn python-multipart
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

CORS 默认放行 `http://127.0.0.1:5173`、`http://localhost:5173` 及一个 Netlify 前端地址（见 `api_server.py`）。

## 九、通信协议（端边协同）

- 传输层：**TCP socket**，默认端口 `9999`；
- 分帧方式：先发 **4 字节大端无符号整数**表示数据总长度，再发数据本体（`struct.pack("!I", n)`）；
- 负载：`pickle` 序列化后的字典，包含排序后的 JPEG 背景、PNG 掩码、各 ROI 的 JPEG 字节、ROI 坐标与原图尺寸；
- 回传：边侧同样以「长度前缀 + 数据」回传**评估结果（pickle）**与**重建图（JPEG）**。

## 十、已知限制

- 边侧不做 PSNR/SSIM：端侧传来的数据不含原图文件名，边侧默认返回 `N/A`；如需画质评估，需保证两端 `data/` 目录内容一致并自行匹配原图。
- `pickle` 反序列化仅适用于可信的对端，公网使用需替换为更安全的序列化方案。
- Real-ESRGAN 在无 GPU 时按 CPU 回归运行，较慢，可去掉 `--super-res` 改用 Bicubic。
- 检测依赖 COCO 80 类通用模型，未针对特定场景微调。

## 十一、技术栈

Python 3.12 · Ultralytics YOLOv8 · OpenCV · NumPy · scikit-image · PyTorch（可选）· Real-ESRGAN（可选）· FastAPI（第二版后端）
