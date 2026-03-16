import os
import numpy as np
from PIL import Image
import cv2  # 用OpenCV做更精细的处理


def custom_gray_conversion(img):
    # 转换为RGB模式（确保处理一致性）
    img_rgb = img.convert('RGB')
    r, g, b = img_rgb.split()

    # 转换为numpy数组（0-255）
    r = np.array(r, dtype=np.float32)
    g = np.array(g, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    # 自定义权重（可根据边界颜色调整，这里示例突出红色和蓝色的差异）
    gray = gray = 0.01*r + 0.5*g + 0.29*b

    # 归一化到0-255
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return Image.fromarray(gray)


def local_contrast_enhance(gray_img, clip_limit=2.0, grid_size=(8, 8)):
    # 转换为OpenCV格式
    gray_cv = np.array(gray_img)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    # 应用增强
    enhanced_cv = clahe.apply(gray_cv)
    return Image.fromarray(enhanced_cv)


def batch_convert_to_gray(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            try:
                with Image.open(input_path) as img:
                    # 步骤1：用自定义公式转换灰度，保留边界差异
                    gray_img = custom_gray_conversion(img)

                    # 步骤2：局部对比度增强，让边界更清晰
                    enhanced_img = local_contrast_enhance(gray_img)

                    # 保存结果
                    output_filename = f"{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}"
                    output_path = os.path.join(output_folder, output_filename)
                    enhanced_img.save(output_path)
                    print(f"已处理: {filename} -> {output_filename}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

    print("处理完成！结果保存到:", output_folder)


if __name__ == "__main__":
    input_folder = r"D:\dachuang\TESTS\JPEGImages-512"
    output_folder = r"D:\dachuang\TESTS\wb-512"

    batch_convert_to_gray(input_folder, output_folder)