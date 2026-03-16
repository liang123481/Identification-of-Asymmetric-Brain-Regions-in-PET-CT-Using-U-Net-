import os
import cv2
import numpy as np
from glob import glob


def sync_resize_pad_two_folders(
        img_dir,  # 第一个文件夹（图片文件夹，作为基准）
        paired_dir,  # 第二个文件夹（与图片同名、后缀不同的文件）
        output_img_dir,  # 图片文件夹处理后的输出路径
        output_paired_dir,  # 第二个文件夹处理后的输出路径
        target_size=(512, 512)  # 目标尺寸（与之前一致）
):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_paired_dir, exist_ok=True)

    # 1. 处理第一个文件夹（图片文件夹），记录每个文件的「缩放+填充参数」
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob(os.path.join(img_dir, ext)))

    if not img_paths:
        print(f"警告：在图片文件夹 {img_dir} 中未找到任何图片文件")
        return
    file_param_map = {}

    # 先处理图片文件夹，记录参数
    print("开始处理图片文件夹...")
    for img_path in img_paths:
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过图片文件夹中无法读取的文件：{img_path}")
            continue

        # 获取文件名（无后缀）和原始尺寸
        filename_with_ext = os.path.basename(img_path)
        filename = os.path.splitext(filename_with_ext)[0]  # 提取无后缀的文件名（用于匹配）
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # 计算缩放比例（保持原比例，无拉伸）
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 计算填充量（上下左右，确保图片居中）
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        # 存储参数（后续给同名文件复用）
        file_param_map[filename] = {
            "scale": scale,
            "new_size": (new_w, new_h),
            "pad": (pad_top, pad_bottom, pad_left, pad_right),
            "original_ext": filename_with_ext.split('.')[-1]
        }

        # 缩放图片并填充
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_img = cv2.copyMakeBorder(
            resized_img,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # 黑色填充
        )

        # 保存处理后的图片
        output_img_path = os.path.join(output_img_dir, filename_with_ext)
        cv2.imwrite(output_img_path, padded_img)
        print(f"图片处理完成：{filename_with_ext}（原尺寸：{w}×{h} → 目标尺寸：{target_w}×{target_h}）")

    # 2. 处理第二个文件夹（与图片同名的文件），复用已计算的参数
    print("\n开始处理对应文件文件夹")
    # 获取第二个文件夹的所有文件（不限制后缀，按文件名匹配）
    paired_files = [f for f in os.listdir(paired_dir) if os.path.isfile(os.path.join(paired_dir, f))]

    if not paired_files:
        print(f"在对应文件文件夹 {paired_dir} 中未找到任何文件")
        return

    for paired_file in paired_files:
        # 提取无后缀的文件名（用于匹配图片的参数）
        filename = os.path.splitext(paired_file)[0]
        paired_file_path = os.path.join(paired_dir, paired_file)

        # 检查是否有对应的图片参数（确保文件名匹配）
        if filename not in file_param_map:
            print(f"跳过：对应文件 {paired_file} 无匹配的图片文件，未处理")
            continue

        # 读取对应文件（支持图片类文件，如掩码、辅助图）
        paired_data = cv2.imread(paired_file_path, cv2.IMREAD_UNCHANGED)  # 用IMREAD_UNCHANGED保留原通道（如单通道掩码）
        if paired_data is None:
            print(f"跳过对应文件文件夹中无法读取的文件：{paired_file_path}")
            continue

        # 复用图片的处理参数
        param = file_param_map[filename]
        scale = param["scale"]
        new_w, new_h = param["new_size"]
        pad_top, pad_bottom, pad_left, pad_right = param["pad"]

        # 缩放（注意：单通道/多通道均兼容）
        # 若为单通道（如掩码），resize后仍为单通道
        resized_paired = cv2.resize(paired_data, (new_w, new_h),
                                    interpolation=cv2.INTER_NEAREST)  # 掩码用INTER_NEAREST避免模糊

        # 黑色填充（单通道用0填充，多通道用(0,0,0)填充）
        if len(paired_data.shape) == 2:  # 单通道（如灰度掩码）
            padded_paired = cv2.copyMakeBorder(
                resized_paired,
                top=pad_top,
                bottom=pad_bottom,
                left=pad_left,
                right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=0  # 单通道黑色为0
            )
        else:  # 多通道（如RGB辅助图）
            padded_paired = cv2.copyMakeBorder(
                resized_paired,
                top=pad_top,
                bottom=pad_bottom,
                left=pad_left,
                right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]  # 多通道黑色为(0,0,0)
            )

        # 保存处理后的对应文件（保持原后缀）
        output_paired_path = os.path.join(output_paired_dir, paired_file)
        cv2.imwrite(output_paired_path, padded_paired)
        print(f"对应文件处理完成：{paired_file}（尺寸已与同名图片对齐）")

    print(f"\n所有文件处理完成！")
    print(f"处理后图片保存至：{output_img_dir}")
    print(f"处理后对应文件保存至：{output_paired_dir}")


# 示例用法
if __name__ == "__main__":
    # ---------------------- 请根据你的路径修改 ----------------------
    img_dir = r"D:\datastore\TESTS\JPEGImages"  # 原图
    paired_dir = r"D:\datastore\TESTS\SegmentationClassPNG"  # 掩码
    output_img_dir = r"D:\temp\ceshijiyuantu"   # 图片处理后的输出路径
    output_paired_dir = r"D:\temp\ceshijiyuantu"  # 对应文件处理后的输出路径
    target_size = (512, 512)  # 目标尺寸（与之前一致）

    sync_resize_pad_two_folders(
        img_dir=img_dir,
        paired_dir=paired_dir,
        output_img_dir=output_img_dir,
        output_paired_dir=output_paired_dir,
        target_size=target_size
    )
