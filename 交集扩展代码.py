import os
import numpy as np
import cv2
from glob import glob


def rgb_mask_to_binary(rgb_mask):
    is_foreground = np.any(rgb_mask != [0, 0, 0], axis=-1)
    return is_foreground.astype(np.uint8) * 255


def find_image_with_same_name(folder, base_name):
    supported_formats = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']
    for fmt in supported_formats:
        possible = os.path.join(folder, f"{base_name}.{fmt}")
        if os.path.exists(possible):
            return possible
    return None


def expand_mask_based_on_similarity(original_img, mask, kernel_size=5, threshold=30):
    binary = mask.astype(bool)
    expanded_mask = binary.copy()

    h, w = mask.shape[:2]
    k = (kernel_size - 1) // 2

    foreground_coords = np.where(binary)

    for y, x in zip(foreground_coords[0], foreground_coords[1]):
        center_rgb = original_img[y, x]

        y_min, y_max = max(0, y - k), min(h, y + k + 1)
        x_min, x_max = max(0, x - k), min(w, x + k + 1)

        for ny in range(y_min, y_max):
            for nx in range(x_min, x_max):
                if not expanded_mask[ny, nx]:
                    current_rgb = original_img[ny, nx]
                    # 计算当前像素与前景像素的RGB欧氏距离
                    distance = np.sqrt(np.sum((center_rgb - current_rgb) ** 2))
                    if distance < threshold:
                        expanded_mask[ny, nx] = True

    return expanded_mask.astype(np.uint8) * 255


def process_masks_to_final_output(pred_folder1, pred_folder2, img_folder, output_folder, expand_kernel=5,
                                  expand_threshold=30):
    os.makedirs(output_folder, exist_ok=True)

    supported_formats = ['png', 'jpg', 'jpeg']
    pred_files1 = []
    for fmt in supported_formats:
        pred_files1.extend(glob(os.path.join(pred_folder1, f"*.{fmt}")))
        pred_files1.extend(glob(os.path.join(pred_folder1, f"*.{fmt.upper()}")))

    pred_files1 = list(set(pred_files1))
    print(f"预测文件夹1路径: {pred_folder1}\n找到 {len(pred_files1)} 个掩码文件，开始处理...\n")

    if not pred_files1:
        print("错误: 预测文件夹1中未找到文件")
        return

    # 2. 遍历处理每个文件
    for pred_path1 in pred_files1:
        file_name = os.path.basename(pred_path1)
        file_base, file_ext = os.path.splitext(file_name)

        pred_path2 = os.path.join(pred_folder2, f"{file_base}{file_ext}")
        img_path = find_image_with_same_name(img_folder, file_base)

        if not os.path.exists(pred_path2) or img_path is None:
            print(f"[-] 警告: 缺失预测图2或原图，跳过文件 {file_base}")
            continue

        try:
            # 读取图片 (保持RGB读取，因为色彩距离计算是基于RGB的)
            pred1_rgb = cv2.cvtColor(cv2.imread(pred_path1), cv2.COLOR_BGR2RGB)
            pred2_rgb = cv2.cvtColor(cv2.imread(pred_path2), cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[-] 图片读取失败 {file_base}: {str(e)}")
            continue

        # 将读取的预测图转为 0和255的 二值图
        pred1_binary = rgb_mask_to_binary(pred1_rgb)
        pred2_binary = rgb_mask_to_binary(pred2_rgb)

        # 核心步骤一：分别对两个预测掩码基于原图进行色彩扩展
        expanded_pred1 = expand_mask_based_on_similarity(
            original_img=img, mask=pred1_binary, kernel_size=expand_kernel, threshold=expand_threshold
        )
        expanded_pred2 = expand_mask_based_on_similarity(
            original_img=img, mask=pred2_binary, kernel_size=expand_kernel, threshold=expand_threshold
        )

        # 核心步骤二：扩展后取交集 (只有两者同时为前景时，最终才为前景)对每个像素位置，只有 expanded_pred1 且 expanded_pred2 同时为前景（非 0）时，最终掩码该位置才是前景，否则为背景 —— 这就是集合交集的像素级实现。
        final_intersection = np.logical_and(expanded_pred1, expanded_pred2).astype(np.uint8) * 255


        save_path = os.path.join(output_folder, f"{file_base}.png")
        cv2.imwrite(save_path, final_intersection)
        print(f"[+] 成功生成并保存最终掩码: {file_base}.png")

    print("\n✅ 所有文件处理完毕！")


if __name__ == "__main__":
    pre1 = r"D:\datastore\TESTS\hui-predict"#原图模型训练得到的预测结果文件夹
    pre2 = r"D:\datastore\TESTS\yuan-predict"#灰度模型得到的文件夹
    img = r"D:\datastore\TESTS\JPEGImages-512"#原图
    output = r"D:\temp\kuozhan-1"#扩展交集掩码

    # 超参数设置
    k = 5  # 扩展核大小 越大范围越广
    yuzhi = 30  # 色彩容差阈值

    process_masks_to_final_output(
        pred_folder1=pre1,
        pred_folder2=pre2,
        img_folder=img,
        output_folder=output,
        expand_kernel=k,
        expand_threshold=yuzhi
    )
