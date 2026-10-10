"""
端侧发送端 —— 运行在端侧电脑（算力弱的那台）
1. 读图 → YOLO检测 → 双分辨率编码
2. 通过 socket 发送给边侧
3. 接收边侧回传的重建图像和评估结果
"""
import socket
import struct
import pickle
import csv
import cv2
import numpy as np
import time
import argparse
import os

from utils import detect_rois, encode, save_roi_outputs, make_hollow_background


def send_all(sock, data):
    sock.sendall(data)


def recv_all(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("连接断开")
        data += chunk
    return data


def pack_encoded(encoded, bg_quality=30, roi_quality=90):
    """把 encoded dict 序列化为字节流，返回 (总数据, 各分量字节数)"""
    _, bg_jpg = cv2.imencode(".jpg", encoded["lr_background"],
                              [cv2.IMWRITE_JPEG_QUALITY, bg_quality])
    _, mask_png = cv2.imencode(".png", encoded["lr_mask"])

    bg_bytes = bg_jpg.tobytes()
    mask_bytes = mask_png.tobytes()

    patches_bytes_list = []
    for patch in encoded["hr_patches"]:
        _, p_jpg = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, roi_quality])
        patches_bytes_list.append(p_jpg.tobytes())

    payload = {
        "bg": bg_bytes,
        "mask": mask_bytes,
        "patches": patches_bytes_list,
        "rois": encoded["rois"],
        "shape": encoded["original_shape"],
    }
    data = pickle.dumps(payload)

    # pickle 序列化本身的开销 = 总数据 - 各分量裸字节
    meta_overhead = len(data) - len(bg_bytes) - len(mask_bytes) - sum(len(p) for p in patches_bytes_list)

    sizes = {
        "bg_kb": len(bg_bytes) / 1024,
        "mask_kb": len(mask_bytes) / 1024,
        "patches_kb": sum(len(p) for p in patches_bytes_list) / 1024,
        "meta_kb": meta_overhead / 1024,
        "total_kb": len(data) / 1024,
    }
    return data, sizes


def run_edge(host, port, image_path, scale, conf, bg_quality=30, roi_quality=90):
    img_name = os.path.basename(image_path)

    print("=" * 55)
    print(f"  端侧发送端")
    print(f"  目标: {host}:{port}  |  图片: {img_name}  |  scale={scale}  |  bg_q={bg_quality}  roi_q={roi_quality}")
    print("=" * 55)

    # 1. 读图 + 检测 + 编码
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取: {image_path}")
        return

    print(f"\n[1] 检测目标...")
    t0 = time.time()
    rois, _, results = detect_rois(img, conf=conf)
    detect_time = time.time() - t0
    print(f"    检出 {len(rois)} 个目标 ({detect_time:.3f}s)")

    if len(rois) == 0:
        print("    未检测到目标，退出")
        return

    # 输出 ROI 裁剪图 + 标注图 → 和重建对比图归到同一个图片子文件夹
    project_dir = os.path.dirname(os.path.abspath(__file__))
    out_base = os.path.join(project_dir, "output")
    save_roi_outputs(img, rois, results, out_base, img_name)

    print(f"\n[2] 双分辨率编码 (scale={scale})...")
    t0 = time.time()
    encoded = encode(img, rois, scale=scale)
    encode_time = time.time() - t0

    # 序列化（返回实际压缩后字节数 + 分量明细）
    data, enc_sizes = pack_encoded(encoded, bg_quality=bg_quality, roi_quality=roi_quality)
    transmitted_kb = enc_sizes["total_kb"]

    # 基准对比：整张原图直接 JPEG 压缩（与 ROI 块相同质量，公平对比）
    _, baseline_jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, roi_quality])
    baseline_kb = len(baseline_jpg.tobytes()) / 1024
    raw_kb = img.nbytes / 1024
    saving = (1 - transmitted_kb / baseline_kb) * 100

    print(f"    原始图片: {raw_kb:.0f} KB 裸数据 ({img.shape[1]}x{img.shape[0]})")
    print(f"    直接JPEG: {baseline_kb:.0f} KB  (整张原图, 质量{roi_quality})")
    print(f"    传输数据: {transmitted_kb:.1f} KB  (节省 {saving:.1f}%)")
    print(f"       ├─ 低分背景 JPEG: {enc_sizes['bg_kb']:.1f} KB")
    print(f"       ├─ ROI Mask PNG:  {enc_sizes['mask_kb']:.1f} KB")
    print(f"       ├─ 高清ROI块 JPEG:{enc_sizes['patches_kb']:.1f} KB  ({len(encoded['hr_patches'])}块)")
    print(f"       └─ 坐标+序列化:   {enc_sizes['meta_kb']:.1f} KB")
    print(f"    编码耗时: {encode_time:.3f}s")

    # 2. 连接 + 发送
    print(f"\n[3] 发送到边侧 ({host}:{port})...")
    t0 = time.time()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(600)
    sock.connect((host, port))

    # 发送长度 + 数据
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)

    # 3. 接收边侧回传的评估结果
    result_len = struct.unpack("!I", recv_all(sock, 4))[0]
    result_data = recv_all(sock, result_len)
    metrics = pickle.loads(result_data)

    # 4. 接收边侧回传的重建图
    img_len = struct.unpack("!I", recv_all(sock, 4))[0]
    img_bytes = recv_all(sock, img_len)
    trans_time = time.time() - t0
    sock.close()

    reconstructed = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 5. 打印结果
    total_time = detect_time + encode_time + trans_time
    # 实际传输带宽 = 发送数据量 / 传输耗时
    bw_mbps = (transmitted_kb * 8) / (trans_time * 1024) if trans_time > 0 else 0
    print(f"\n[4] 边侧回传结果:")
    print(f"    {'PSNR':>6}: {metrics.get('psnr', 'N/A'):>8} dB")
    print(f"    {'SSIM':>6}: {metrics.get('ssim', 'N/A'):>8}")
    print(f"    {'解码耗时':>6}: {metrics.get('decode_time_s', 0):.3f}s")
    print(f"    {'传输耗时':>6}: {trans_time:.3f}s  (含往返)")
    print(f"    {'实际带宽':>6}: {bw_mbps:.1f} Mbps")
    print(f"    {'端侧总耗时':>6}: {total_time:.3f}s")
    print(f"          (检测{detect_time:.3f} + 编码{encode_time:.3f} + 传输等待{trans_time:.3f})")

    # 6. 保存全部输出到 output/{图片名}/
    base = os.path.splitext(img_name)[0]
    out_dir = os.path.join(project_dir, "output", base)
    os.makedirs(out_dir, exist_ok=True)

    # 原始图
    cv2.imwrite(os.path.join(out_dir, f"{base}_original.jpg"), img)

    # 低分辨率挖空背景
    hollow = make_hollow_background(encoded)
    cv2.imwrite(os.path.join(out_dir, f"{base}_hollow_bg.jpg"), hollow)

    # 重建图 + 对比图
    if reconstructed is not None:
        rec_resized = cv2.resize(reconstructed, (img.shape[1], img.shape[0]))
        # 单独重建图
        cv2.imwrite(os.path.join(out_dir, f"{base}_reconstructed.jpg"), rec_resized)
        # 并排对比图：原图 | 重建
        comparison = np.hstack([img, rec_resized])
        cv2.imwrite(os.path.join(out_dir, f"{base}_comparison.jpg"), comparison)

    print(f"\n    全部输出已保存: {out_dir}/")
    print(f"       ├─ {base}_original.jpg       原始图")
    print(f"       ├─ {base}_roi_*.jpg         高清ROI裁剪 ({len(rois)}块)")
    print(f"       ├─ {base}_annotated.jpg     检测标注")
    print(f"       ├─ {base}_hollow_bg.jpg     低分辨率挖空背景")
    print(f"       ├─ {base}_reconstructed.jpg 重建图")
    print(f"       └─ {base}_comparison.jpg    原图|重建 并排对比")

    # 7. 追加到 output/results.csv 汇总表
    csv_path = os.path.join(out_base, "results.csv")
    file_exists = os.path.isfile(csv_path)
    fields = [
        "image", "resolution", "num_rois",
        "bg_kb", "mask_kb", "patches_kb", "meta_kb", "total_kb",
        "baseline_kb", "saving_pct",
        "encode_s", "decode_s", "trans_s", "total_s", "bw_mbps",
        "scale", "bg_quality", "roi_quality",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "image": img_name,
            "resolution": f"{img.shape[1]}x{img.shape[0]}",
            "num_rois": len(rois),
            "bg_kb": round(enc_sizes["bg_kb"], 1),
            "mask_kb": round(enc_sizes["mask_kb"], 1),
            "patches_kb": round(enc_sizes["patches_kb"], 1),
            "meta_kb": round(enc_sizes["meta_kb"], 1),
            "total_kb": round(enc_sizes["total_kb"], 1),
            "baseline_kb": round(baseline_kb, 1),
            "saving_pct": round(saving, 1),
            "encode_s": round(encode_time, 3),
            "decode_s": metrics.get("decode_time_s", 0),
            "trans_s": round(trans_time, 3),
            "total_s": round(total_time, 3),
            "bw_mbps": round(bw_mbps, 1),
            "scale": scale,
            "bg_quality": bg_quality,
            "roi_quality": roi_quality,
        })
    print(f"    汇总表已追加: {csv_path}")

    # 8. 生成对齐文本表格
    write_aligned_table(csv_path)
    table_path = os.path.join(out_base, "results.txt")
    print(f"    对齐表格已更新: {table_path}")

    print(f"\n{'='*55}\n")


def write_aligned_table(csv_path):
    """从 CSV 生成列对齐的纯文本表格"""
    table_path = csv_path.replace(".csv", ".txt")

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        return

    header = rows[0]

    # 列宽 = len()
    col_widths = [0] * len(header)
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def pad(s, width, right=False):
        return s.rjust(width) if right else s.ljust(width)

    num_cols = {"num_rois", "bg_kb", "mask_kb", "patches_kb", "meta_kb",
                "total_kb", "baseline_kb", "saving_pct",
                "encode_s", "decode_s", "trans_s", "total_s", "bw_mbps",
                "scale", "bg_quality", "roi_quality"}

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    lines = [sep]
    lines.append("|" + "|".join(pad(h, col_widths[i]) for i, h in enumerate(header)) + "|")
    lines.append(sep)

    for row in rows[1:]:
        cells = [pad(cell, col_widths[i], right=header[i] in num_cols)
                 for i, cell in enumerate(row)]
        lines.append("|" + "|".join(cells) + "|")

    lines.append(sep)

    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="端侧发送端")
    parser.add_argument("--host", default="127.0.0.1", help="边侧 IP 地址")
    parser.add_argument("--port", type=int, default=9999, help="端口号")
    parser.add_argument("--image", default="data/newspaper.jpg", help="图片路径")
    parser.add_argument("--scale", type=float, default=0.25, help="背景下采样比例")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度阈值")
    parser.add_argument("--bg-quality", type=int, default=30, help="背景 JPEG 质量 (0-100)")
    parser.add_argument("--roi-quality", type=int, default=90, help="ROI 块 + 基准 JPEG 质量 (0-100)")
    args = parser.parse_args()
    run_edge(args.host, args.port, args.image, args.scale, args.conf,
             bg_quality=args.bg_quality, roi_quality=args.roi_quality)
