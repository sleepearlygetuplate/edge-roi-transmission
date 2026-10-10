"""
边侧接收端 —— 运行在边侧服务器（算力强的那台）
1. 监听 socket → 接收端侧编码数据 → 解码还原
2. 评估画质（需要本地有同样的原图），回传给端侧
"""
import socket
import struct
import pickle
import cv2
import numpy as np
import time
import argparse
import os

from utils import decode, evaluate


def recv_all(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("连接断开")
        data += chunk
    return data


def unpack_encoded(data):
    """把字节流反序列化为 encoded dict"""
    payload = pickle.loads(data)

    bg = cv2.imdecode(np.frombuffer(payload["bg"], dtype=np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.frombuffer(payload["mask"], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    patches = []
    for p_bytes in payload["patches"]:
        patch = cv2.imdecode(np.frombuffer(p_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        patches.append(patch)

    return {
        "lr_background": bg,
        "lr_mask": mask,
        "hr_patches": patches,
        "rois": payload["rois"],
        "original_shape": payload["shape"],
    }


def run_edge_server(host, port, blend_width, super_res, blend_mode, data_dir):

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(5)

    mode_label = f"ESRGAN" if super_res else "Bicubic"
    print("=" * 55)
    print(f"  边侧接收端")
    print(f"  监听: {host}:{port}")
    print(f"  上采样: {mode_label}  |  融合: {blend_mode}")
    print("=" * 55)
    print("  等待端侧连接...\n")

    while True:
        conn, addr = sock.accept()
        print(f"[连接] 来自 {addr[0]}:{addr[1]}")

        try:
            # 1. 接收数据
            data_len = struct.unpack("!I", recv_all(conn, 4))[0]
            print(f"  [接收] {data_len/1024:.0f} KB ...")
            data = recv_all(conn, data_len)

            # 2. 解包 + 解码
            t0 = time.time()
            encoded = unpack_encoded(data)
            h, w = encoded["original_shape"]
            print(f"  [解包] 图像 {w}x{h}, {len(encoded['hr_patches'])} 个 ROI 块")

            reconstructed = decode(encoded,
                                   blend_width=blend_width,
                                   super_res=super_res,
                                   blend_mode=blend_mode)
            decode_time = time.time() - t0
            print(f"  [解码] 完成 ({decode_time:.3f}s)")

            # 3. 评估（如果 data_dir 里有原图）
            metrics = {
                "decode_time_s": round(decode_time, 3),
                "received_kb": round(data_len / 1024, 1),
            }

            # 尝试用文件名匹配原图——边侧需要有一份同样的原图
            # 端侧传来的数据里不包含文件名，所以边侧只能评估收到的重建图
            # 如果需要 PSNR/SSIM，请确保两边 data/ 目录内容一致
            metrics["psnr"] = "N/A（边侧无原图）"
            metrics["ssim"] = "N/A（边侧无原图）"

            # 5. 回传评估结果
            result = pickle.dumps(metrics)
            conn.sendall(struct.pack("!I", len(result)))
            conn.sendall(result)

            # 6. 回传重建图（JPEG 压缩）
            _, recon_jpg = cv2.imencode(".jpg", reconstructed, [cv2.IMWRITE_JPEG_QUALITY, 85])
            conn.sendall(struct.pack("!I", len(recon_jpg)))
            conn.sendall(recon_jpg.tobytes())

            print(f"  [回传] 评估结果 + 重建图完成\n")

        except Exception as e:
            print(f"  ❌ 出错: {e}\n")
        finally:
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="边侧接收端")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（0.0.0.0=任意IP）")
    parser.add_argument("--port", type=int, default=9999, help="端口号")
    parser.add_argument("--blend-width", type=int, default=15, help="高斯羽化宽度")
    parser.add_argument("--super-res", action="store_true", help="启用 ESRGAN 超分")
    parser.add_argument("--blend-mode", default="gaussian",
                        choices=["gaussian", "poisson"], help="融合方式")
    parser.add_argument("--data-dir", default="data", help="原图目录（用于评估）")
    args = parser.parse_args()

    run_edge_server(args.host, args.port, args.blend_width,
                    args.super_res, args.blend_mode, args.data_dir)
