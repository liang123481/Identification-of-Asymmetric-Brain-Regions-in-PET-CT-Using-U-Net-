import os
import numpy as np
from PIL import Image
import cv2


def custom_gray_conversion(img, weights=(0.03, 0.7, 0.25)):
    """
    自定义灰度转换，通过调整RGB通道权重突出特定颜色边界
    """
    # 确保图像为RGB模式
    img_rgb = img.convert('RGB')
    r, g, b = img_rgb.split()

    # 转换为numpy数组进行计算
    r = np.array(r, dtype=np.float32)
    g = np.array(g, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    # 应用自定义权重计算灰度值
    r_weight, g_weight, b_weight = weights
    gray = r_weight * r + g_weight * g + b_weight * b

    # 确保灰度值在0-255范围内
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return Image.fromarray(gray)


def adaptive_thresholding(gray_img, block_size=11, c=2):
    """
    自适应阈值处理，根据局部区域调整阈值
    增强不同光照条件下的对比度
    """
    gray_cv = np.array(gray_img)
    # 自适应阈值处理，将图像转换为黑白二值
    thresh = cv2.adaptiveThreshold(
        gray_cv, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 基于高斯加权和的自适应阈值
        cv2.THRESH_BINARY,
        block_size, c
    )
    return Image.fromarray(thresh)


def local_contrast_enhance(gray_img, clip_limit=2.0, grid_size=(8, 8)):
    """
    使用CLAHE算法进行局部对比度增强
    传统图像处理方法，不涉及机器学习
    """
    gray_cv = np.array(gray_img)
    # 创建CLAHE对象（对比度受限的自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced_cv = clahe.apply(gray_cv)
    return Image.fromarray(enhanced_cv)


def edge_enhancement(gray_img, strength=1.0):
    """
    使用传统边缘检测算法增强图像边界
    基于Sobel算子的边缘检测
    """
    gray_cv = np.array(gray_img)

    # Sobel边缘检测
    sobelx = cv2.Sobel(gray_cv, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_cv, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)

    # 归一化边缘强度
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 融合原图与边缘信息
    enhanced = cv2.addWeighted(gray_cv, 1.0, sobel, strength, 0)
    return Image.fromarray(enhanced)


def process_image(img, weights=(0.03, 0.7, 0.25), clip_limit=2.0, enhance_edges=False):
    """
    完整图像处理流程：灰度转换 -> 对比度增强 -> 可选边缘增强
    全部使用传统方法，无机器学习
    """
    # 步骤1：自定义灰度转换
    gray_img = custom_gray_conversion(img, weights)

    # 步骤2：局部对比度增强
    enhanced_img = local_contrast_enhance(gray_img, clip_limit)

    # 可选步骤：边缘增强
    if enhance_edges:
        enhanced_img = edge_enhancement(enhanced_img)

    return enhanced_img


def batch_process_images(input_folder, output_folder,
                         weights=(0.03, 0.7, 0.25),
                         clip_limit=2.0,
                         enhance_edges=False):
    """批量处理文件夹中的所有图像"""
    os.makedirs(output_folder, exist_ok=True)
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            try:
                with Image.open(input_path) as img:
                    # 处理图像
                    processed_img = process_image(
                        img,
                        weights=weights,
                        clip_limit=clip_limit,
                        enhance_edges=enhance_edges
                    )

                    # 保存结果
                    output_path = os.path.join(output_folder, filename)
                    processed_img.save(output_path)
                    print(f"已处理: {filename}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")

    print(f"所有图像处理完成，结果保存至: {output_folder}")


if __name__ == "__main__":
    # 输入输出文件夹路径
    input_folder = r"D:\datastore\TESTS\JPEGImages-512"
    output_folder = r"D:\datastore\TESTS\JPEGImages-512-wb"

    # 可调整的参数
    color_weights = (0.03, 0.7, 0.25)  # RGB通道权重，可根据需要调整
    contrast_limit = 2.0  # 对比度限制，值越大增强效果越强
    enhance_edges = False  # 是否增强边缘

    # 执行批量处理
    batch_process_images(
        input_folder=input_folder,
        output_folder=output_folder,
        weights=color_weights,
        clip_limit=contrast_limit,
        enhance_edges=enhance_edges
    )
