"""
端边协同双分辨率传输 —— 核心工具函数
"""
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 模型路径固定到 utils.py 所在目录，不管从哪里运行都指向同一个文件
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_YOLO_PATH = os.path.join(_MODEL_DIR, "yolov8n.pt")

# ============================================================
# 全局：只加载一次模型
# ============================================================
_model = None
_sr_upsampler = None


def get_model():
    """懒加载 YOLO 模型"""
    global _model
    if _model is None:
        _model = YOLO(_YOLO_PATH)
    return _model


def get_sr_upsampler():
    """懒加载 Real-ESRGAN 超分模型（x4 放大）"""
    global _sr_upsampler
    if _sr_upsampler is None:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        if torch.cuda.is_available():
            device = "cuda"
            tile = 0
        elif torch.backends.mps.is_available():
            device = "mps"
            tile = 128
        else:
            device = "cpu"
            tile = 128

        _sr_upsampler = RealESRGANer(
            scale=4,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device,
        )
        print(f"  Real-ESRGAN 加载完成，设备: {_sr_upsampler.device}")
    return _sr_upsampler


# ============================================================
# 检测
# ============================================================
def detect_rois(img, conf=0.25):
    """
    用 YOLO 检测目标，返回 ROI 列表和 mask
    参数:
        img: BGR 图像 (numpy array)
        conf: 置信度阈值，越低检出的越多
    返回:
        rois: [(x1, y1, x2, y2), ...]  每个ROI的坐标
        roi_mask: (h, w) 的二值 mask，255=ROI区域
    """
    h, w = img.shape[:2]
    model = get_model()
    results = model(img, conf=conf, verbose=False)[0]

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    rois = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # 边界保护
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:  # 有效框才保留
            roi_mask[y1:y2, x1:x2] = 255
            rois.append((x1, y1, x2, y2))

    return rois, roi_mask, results


# ============================================================
# ROI 输出
# ============================================================
def save_roi_outputs(img, rois, results, out_dir, img_name):
    """
    保存 ROI 裁剪图和标注图
    参数:
        img:       原始图像 (BGR)
        rois:      [(x1, y1, x2, y2), ...]
        results:   ultralytics YOLO 的检测结果对象（用于拿类别名和置信度）
        out_dir:   输出目录
        img_name:  原始图片文件名（如 "cat.jpg"）
    """
    base = os.path.splitext(img_name)[0]
    sub_dir = os.path.join(out_dir, base)
    os.makedirs(sub_dir, exist_ok=True)

    # —— 裁剪图：每个 ROI 单独保存 ——
    for i, (x1, y1, x2, y2) in enumerate(rois):
        crop = img[y1:y2, x1:x2]
        crop_path = os.path.join(sub_dir, f"{base}_roi_{i+1:02d}.jpg")
        cv2.imwrite(crop_path, crop)

    # —— 标注图：在原图上画框 + 标签 ——
    annotated = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(rois):
        # 绿框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 标签文字：类别 + 置信度（如果有 results）
        label = f"ROI #{i+1}"
        if results is not None and hasattr(results, 'boxes') and results.boxes is not None:
            try:
                cls_id = int(results.boxes.cls[i].item())
                conf = float(results.boxes.conf[i].item())
                cls_name = results.names.get(cls_id, str(cls_id)) if hasattr(results, 'names') else str(cls_id)
                label = f"{cls_name} {conf:.2f}"
            except (IndexError, AttributeError):
                pass

        # 标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    anno_path = os.path.join(sub_dir, f"{base}_annotated.jpg")
    cv2.imwrite(anno_path, annotated)

    print(f"  [ROI输出] {len(rois)} 张裁剪图 + 1 张标注图 → {sub_dir}/")


def make_hollow_background(encoded):
    """低分辨率挖空背景：将 ROI 区域涂黑，展示实际传输的背景图"""
    bg = encoded["lr_background"].copy()
    mask = encoded["lr_mask"]
    bg[mask > 0] = 0
    return bg


# ============================================================
# 端侧编码
# ============================================================
def encode(img, rois, scale=0.25):
    """
    端侧：把原图编码为双分辨率格式
    参数:
        img: 原始图像
        rois: ROI 坐标列表
        scale: 背景下采样比例 (0~1)
    返回:
        encoded: dict，包含 lr_background, lr_mask, hr_patches, rois, original_shape
    """
    h, w = img.shape[:2]

    # 背景 + mask 降采样
    img_small = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_LINEAR)

    # 生成原始分辨率的 mask，再降采样
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in rois:
        roi_mask[y1:y2, x1:x2] = 255
    mask_small = cv2.resize(roi_mask,
                            (img_small.shape[1], img_small.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    # 裁下高清 ROI 块（原始分辨率）
    hr_patches = []
    for x1, y1, x2, y2 in rois:
        hr_patches.append(img[y1:y2, x1:x2].copy())

    return {
        "lr_background": img_small,
        "lr_mask": mask_small,
        "hr_patches": hr_patches,
        "rois": rois,
        "original_shape": (h, w),
    }


# ============================================================
# 边侧解码
# ============================================================
def decode(encoded, blend_width=15, super_res=False, blend_mode="gaussian"):
    """
    边侧：还原图像
    参数:
        encoded: encode() 的输出 dict
        blend_width: 高斯羽化宽度（像素），only for blend_mode="gaussian"
        super_res: True=Real-ESRGAN超分, False=Bicubic插值
        blend_mode: "gaussian" 羽化融合, "poisson" 泊松融合
    返回:
        result: 重建的 BGR 图像
    """
    h, w = encoded["original_shape"]
    rois = encoded["rois"]
    hr_patches = encoded["hr_patches"]
    bg_small = encoded["lr_background"]
    mask_small = encoded["lr_mask"]

    # 背景上采样回原始尺寸
    if super_res:
        bg_upsampled = _super_res_upsample(bg_small, (w, h))
    else:
        bg_upsampled = cv2.resize(bg_small, (w, h),
                                  interpolation=cv2.INTER_CUBIC)

    # 贴 ROI
    result = bg_upsampled.copy()
    for (x1, y1, x2, y2), patch in zip(rois, hr_patches):
        pw, ph = x2 - x1, y2 - y1
        if pw <= 0 or ph <= 0:
            continue
        patch_resized = cv2.resize(patch, (pw, ph))

        if blend_mode == "poisson":
            _poisson_blend(result, patch_resized, x1, y1, x2, y2)
        elif blend_width > 0:
            _blend_patch(result, patch_resized, x1, y1, x2, y2, blend_width)
        else:
            result[y1:y2, x1:x2] = patch_resized

    return result


def _super_res_upsample(img_small, target_size):
    """用 Real-ESRGAN 把低分辨率背景超分到目标尺寸"""
    upsampler = get_sr_upsampler()

    # Real-ESRGAN 输入输出都是 RGB，需要转换
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    # 超分放大（outscale=4，所以先缩一点再超到刚好匹配目标尺寸）
    # 如果原始 scale 不是 0.25，需要适配。先简单处理：直接用 outscale 参数
    tw, th = target_size
    sw, sh = img_small.shape[1], img_small.shape[0]
    required_scale = max(tw / sw, th / sh)

    # Real-ESRGAN 支持非整数 scale
    output, _ = upsampler.enhance(img_rgb, outscale=required_scale)

    # 如果输出尺寸不完全匹配，resize 到精确尺寸
    if output.shape[1] != tw or output.shape[0] != th:
        output = cv2.resize(output, (tw, th), interpolation=cv2.INTER_CUBIC)

    # 转回 BGR
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def _blend_patch(background, patch, x1, y1, x2, y2, feather_width):
    """把 patch 羽化融合到背景上（原地修改 background）"""
    ph, pw = y2 - y1, x2 - x1
    if ph < 3 or pw < 3:
        background[y1:y2, x1:x2] = patch
        return

    # 构建权重 mask：中心=1，边缘渐变到 0
    mask = np.ones((ph, pw), dtype=np.float32)
    fw = min(feather_width, ph // 2, pw // 2)

    for i in range(fw):
        alpha = i / fw
        # 上下边缘
        mask[i, :] *= alpha
        mask[-(i + 1), :] *= alpha
        # 左右边缘
        mask[:, i] *= alpha
        mask[:, -(i + 1)] *= alpha

    mask_3ch = np.dstack([mask] * 3)

    # Alpha 混合
    bg_region = background[y1:y2, x1:x2].astype(np.float32)
    fg_region = patch.astype(np.float32)
    blended = fg_region * mask_3ch + bg_region * (1.0 - mask_3ch)
    background[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)


def _poisson_blend(background, patch, x1, y1, x2, y2):
    """用泊松融合把 patch 无缝贴到背景上（原地修改 background）"""
    ph, pw = y2 - y1, x2 - x1
    if ph < 3 or pw < 3:
        background[y1:y2, x1:x2] = patch
        return

    # 泊松融合需要三样东西：
    #   src:   要贴上去的 ROI 块
    #   dst:   目标背景图
    #   mask:  ROI 区域的二值遮罩
    #   center: 融合中心点
    roi_mask = np.full((ph, pw), 255, dtype=np.uint8)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    # OpenCV 的 seamlessClone 直接处理整张图，返回新图
    try:
        background[:] = cv2.seamlessClone(
            patch, background, roi_mask, center, cv2.NORMAL_CLONE
        )
    except cv2.error:
        # 如果泊松融合失败（ROI 太靠边缘等情况），回退到直接赋值
        background[y1:y2, x1:x2] = patch


# ============================================================
# 评估
# ============================================================
def evaluate(original, reconstructed, encoded):
    """
    评估重建质量
    返回:
        dict: {psnr, ssim, original_kb, transmitted_kb, saving_pct}
    """
    # PSNR / SSIM
    psnr_val = psnr(original, reconstructed, data_range=255)
    ssim_val = ssim(original, reconstructed, data_range=255, channel_axis=2)

    # 数据量
    original_kb = original.nbytes / 1024
    lr_kb = encoded["lr_background"].nbytes / 1024
    patches_kb = sum(p.nbytes for p in encoded["hr_patches"]) / 1024
    meta_kb = len(encoded["rois"]) * 16 / 1024  # 每个 ROI 约 16 字节
    transmitted_kb = lr_kb + patches_kb + meta_kb
    saving_pct = (1 - transmitted_kb / original_kb) * 100 if original_kb > 0 else 0

    return {
        "psnr": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "original_kb": round(original_kb, 1),
        "transmitted_kb": round(transmitted_kb, 1),
        "saving_pct": round(saving_pct, 1),
    }
