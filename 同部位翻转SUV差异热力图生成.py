import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def cv_imread_safe(file_path):
    valid_path = file_path.strip().strip('"').strip("'")
    if not os.path.exists(valid_path):
        print(f"❌ 文件不存在: {valid_path}")
        return None
    try:
        img_array = np.fromfile(valid_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return None


def compare_brain_ecc_range(left_path, right_path, output_path):
    img_L = cv_imread_safe(left_path)
    img_R = cv_imread_safe(right_path)

    if img_L is None or img_R is None:
        print("⚠️ 读取图像失败，跳过。")
        return

    h, w = img_L.shape[:2]
    img_R = cv2.resize(img_R, (w, h))

    gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

    gray_R_flipped = cv2.flip(gray_R, 1)

    def get_centroid(img):
        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
        M = cv2.moments(thresh)
        if M["m00"] == 0:
            return w // 2, h // 2
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    cx_L, cy_L = get_centroid(gray_L)
    cx_R, cy_R = get_centroid(gray_R_flipped)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2], warp_matrix[1, 2] = cx_R - cx_L, cy_R - cy_L

    print("  -> 正在进行精细配准...")
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-5)
        _, warp_matrix = cv2.findTransformECC(gray_L, gray_R_flipped, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        print("配准成功")
    except cv2.error:
        print("精细配准未收敛，使用粗对齐。")

    gray_R_aligned = cv2.warpAffine(gray_R_flipped, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    blur_L = cv2.GaussianBlur(gray_L, (5, 5), 0)
    blur_R = cv2.GaussianBlur(gray_R_aligned, (5, 5), 0)

    diff = cv2.absdiff(blur_L, blur_R)
    max_diff = np.max(diff)

    high_light_mask = (diff > 40).astype(np.uint8)
    black_threshold = 5
    black_bg_L = (gray_L <= black_threshold)
    black_bg_R = (gray_R_aligned <= black_threshold)
    need_black_mask = black_bg_L | black_bg_R

    heatmap_before = np.zeros_like(diff)
    heatmap_before[high_light_mask == 1] = diff[high_light_mask == 1]

    heatmap_after = heatmap_before.copy()
    heatmap_after[need_black_mask] = 0

    # ====================== 核心修改：4张横排 ======================
    fig = plt.figure(figsize=(18, 4.5), constrained_layout=True)

    # 1行4列，几乎无间距
    gs = fig.add_gridspec(1, 4, wspace=0.02)

    # 1) 左图
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB))
    ax1.set_title("1. 原始左侧", fontsize=11, pad=2)
    ax1.axis('off')
    ax1.set_xlim(0, w)
    ax1.set_ylim(h, 0)
    ax1.set_aspect('equal', adjustable='box')

    # 2) 右图
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB))
    ax2.set_title("2. 原始右侧", fontsize=11, pad=2)
    ax2.axis('off')
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.set_aspect('equal', adjustable='box')

    # 3) 热力图 前
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(heatmap_before, cmap='jet', vmin=2, vmax=max_diff)
    ax3.set_title("3. 热力图（处理前）", fontsize=11, pad=2)
    ax3.axis('off')
    ax3.set_xlim(0, w)
    ax3.set_ylim(h, 0)
    ax3.set_aspect('equal', adjustable='box')

    # 4) 热力图 后
    ax4 = fig.add_subplot(gs[3])
    im4 = ax4.imshow(heatmap_after, cmap='jet', vmin=2, vmax=max_diff)
    ax4.set_title("4. 热力图（处理后）", fontsize=11, pad=2)
    ax4.axis('off')
    ax4.set_xlim(0, w)
    ax4.set_ylim(h, 0)
    ax4.set_aspect('equal', adjustable='box')

    # 统一色条
    cbar = fig.colorbar(im4, ax=[ax3, ax4], fraction=0.02, pad=0.01, shrink=0.8)
    cbar.set_label('Difference', fontsize=9)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close()
    # =================================================================


def batch_process(dir_L, dir_R, dir_out):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
        print(f"📂 创建输出文件夹: {dir_out}")

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    left_files = [f for f in os.listdir(dir_L) if f.lower().endswith(valid_exts)]

    if not left_files:
        print(f"⚠️ 在 {dir_L} 中没有找到图片。")
        return

    print(f"🔍 找到 {len(left_files)} 张左侧图片，开始批量处理...")
    print("-" * 40)

    for left_filename in left_files:
        right_filename = left_filename.replace('left', 'right')
        path_L = os.path.join(dir_L, left_filename)
        path_R = os.path.join(dir_R, right_filename)

        if not os.path.exists(path_R):
            print(f"⚠️ 跳过: {left_filename} (未找到对应的右图)")
            continue

        out_filename = left_filename.replace('left', 'SUV')
        out_path = os.path.join(dir_out, out_filename)

        print(f"🔄 正在处理: {left_filename} <-> {right_filename}")
        compare_brain_ecc_range(path_L, path_R, out_path)
        print(f"📁 已保存: {out_filename}")
        print("-" * 40)

    print("🎉 批量处理完成！")


if __name__ == "__main__":
    dir_L = r"D:\datastore\TESTS\fge\Origin\upper_left"
    dir_R = r"D:\datastore\TESTS\fge\Origin\upper_right"
    dir_out = r"D:\datastore\TESTS\fge\Origin\SUV-up"
    batch_process(dir_L, dir_R, dir_out)
