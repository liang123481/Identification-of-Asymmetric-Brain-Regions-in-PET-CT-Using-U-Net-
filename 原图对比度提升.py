import cv2
import numpy as np
import os  # 用于遍历文件夹和处理路径


def smart_auto_enhance(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告：无法读取图片 {image_path}，已跳过")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 分析亮度并动态调整参数
    avg_brightness = np.mean(v)
    print(f"\n图片 {os.path.basename(image_path)} 平均亮度：{avg_brightness:.1f}")

    if avg_brightness < 80:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12))
    elif avg_brightness > 160:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

    # 应用增强并保存
    enhanced_v = clahe.apply(v)
    enhanced_hsv = cv2.merge((h, s, enhanced_v))
    result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, result)
    print(f"已保存增强结果：{output_path}")


def batch_process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # 定义支持的图片格式（可根据需要补充）
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否为图片
        if filename.lower().endswith(image_extensions):
            # 构建输入图片的完整路径
            input_path = os.path.join(input_folder, filename)
            # 构建输出图片的完整路径（保留原文件名，可加前缀区分）
            output_filename = f"{filename}"  # 增强后的文件名前加 "enhanced_"
            output_path = os.path.join(output_folder, output_filename)

            # 调用增强函数处理单张图片
            smart_auto_enhance(input_path, output_path)

    print("\n批量处理完成！所有增强后的图片已保存到：", output_folder)


# ------------------- 批量处理调用 -------------------
if __name__ == "__main__":
    # 输入文件夹（存放要处理的所有图片）
    input_folder = r"D:\datastore\TESTS\JPEGImages-512" # 替换为你的图片文件夹路径
    # 输出文件夹（存放增强后的图片，可自定义路径）
    output_folder = r"D:\datastore\TESTS\JPEGImages-512-contract" # 建议与输入文件夹区分开

    # 执行批量处理
    batch_process_folder(input_folder, output_folder)
    #原图