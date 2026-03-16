import argparse
import logging
import os
from pathlib import Path
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
#加载你训练好的 U-Net 模型权重，对单张图片或整个文件夹里的所有图片自动进行分割预测，并将生成的掩码（Mask）保存为 PNG 图片。

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images or folder')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', required=True,
                        help='Filenames of input images or path to a folder containing images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Directory for output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--gray', '-g', action='store_true', help='Convert input images to grayscale before processing')
    parser.add_argument('--extensions', '-e', metavar='EXT', nargs='+', default=['jpg', 'jpeg', 'png', 'bmp'],
                        help='Image file extensions to process (default: jpg, jpeg, png, bmp)')

    return parser.parse_args()


def get_image_paths(input_path):
    """获取输入路径中的所有图片文件路径"""
    if os.path.isdir(input_path):
        # 如果是文件夹，收集所有指定格式的图片
        image_paths = []
        for ext in args.extensions:
            # 匹配所有带指定扩展名的文件，不区分大小写
            pattern = os.path.join(input_path, f'*.{ext.lower()}')
            image_paths.extend(glob.glob(pattern))
            pattern = os.path.join(input_path, f'*.{ext.upper()}')
            image_paths.extend(glob.glob(pattern))
        # 去重并排序
        image_paths = sorted(list(set(image_paths)))
        return image_paths
    elif os.path.isfile(input_path):
        # 如果是文件，直接返回包含该文件的列表
        return [input_path]
    else:
        logging.error(f"Input path not found: {input_path}")
        return []


def get_output_filenames(args, image_paths):
    """根据输入生成输出文件名"""
    # 如果没有指定输出，使用输入文件所在目录
    if not args.output:
        return [f'{os.path.splitext(fn)[0]}.png' for fn in image_paths]

    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)

    # 为每个输入文件在输出目录中生成文件名
    return [os.path.join(args.output, f'{os.path.splitext(os.path.basename(fn))[0]}.png') for fn in image_paths]


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def process_images(net, device, args, mask_values):
    """处理图片的主函数"""
    # 获取所有图片路径
    image_paths = get_image_paths(args.input)

    if not image_paths:
        logging.error("No images found for processing")
        return

    out_files = get_output_filenames(args, image_paths)

    logging.info(f"Found {len(image_paths)} images to process...")

    for i, filename in enumerate(image_paths):
        try:
            logging.info(f'Processing image {i + 1}/{len(image_paths)}: {filename}')

            # 打开图片，根据参数决定是否转为灰度图
            img = Image.open(filename)
            if args.gray:
                img = img.convert('L')  # 转为灰度图

            # 预测掩码
            mask = predict_img(
                net=net,
                full_img=img,
                scale_factor=args.scale,
                out_threshold=args.mask_threshold,
                device=device
            )

            # 保存结果
            if not args.no_save:
                out_filename = out_files[i]
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            # 可视化
            if args.viz:
                logging.info(f'Visualizing results for image {filename}, close to continue...')
                plot_img_and_mask(img, mask)

        except Exception as e:
            logging.error(f"Error processing image {filename}: {str(e)}")
            continue

    logging.info("Processing completed!")


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 初始化模型
    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    # 处理图片
    process_images(net, device, args, mask_values)
    #python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model "D:\dachuang\goodmodel\lunkuo.pth" --input D:\\PythonProject11\\Pytorch-UNet\\data\\imgs --output D:\\PythonProject11\\Pytorch-UNet\\predict_output
    #python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model "D:\dachuang\goodmodel\lunkuo.pth" --input "D:\datastore\TESTS\JPEGImages-512" --output "D:\datastore\TESTS\round"
    # python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model D:\\PythonProject11\\Pytorch-UNet\\checkpoints\\checkpoint_epoch12.pth --input D:\\PythonProject11\\Pytorch-UNet\\data\\imgs --output D:\\PythonProject11\\Pytorch-UNet\\predict_output --gray
    #python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model D:\\PythonProject11\\Pytorch-UNet\\checkpoints\\checkpoint_epoch16.pth --input "D:\\dachuang\\TESTS\\wb-512" --output "D:\\dachuang\\TESTS\\huidu-predict"
#python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model D:\PythonProject11\Pytorch-UNet\checkpoints\checkpoint_epoch16.pth --input "D:\dachuang\TESTS\wb-512"" --output "D:\dachuang\TESTS\huidu-predict"
    #python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model D:\PythonProject11\Pytorch-UNet\checkpoints\checkpoint_epoch12.pth --input D:\PythonProject11\Pytorch-UNet\data\imgs
#python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model D:\PythonProject11\Pytorch-UNet\checkpoints\checkpoint_epoch15.pth --input D:\PythonProject11\Pytorch-UNet\data\imgs --output D:\PythonProject11\Pytorch-UNet\predict_output  # 自定义输出目录
#python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model "C:\Users\LENOVO\Desktop\大创\goodmodel\1015.pth" --input "D:\datastore\917\all_enhanced" --output "D:\datastore\917\out" --gray
#python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model D:\\PythonProject11\\Pytorch-UNet\\checkpoints\\checkpoint_epoch12.pth --input "D:\\datastore\\test\\512" --output "D:\\datastore\\test\\mcmask"
#python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model "D:\dachuang\goodmodel\1018.pth" --input "D:\datastore\goodph\ori" --output D:\\PythonProject11\\Pytorch-UNet\\predict_output --gray
#python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model "D:\dachuang\goodmodel\yuan-11-8-2.pth" --input "D:\datastore\TESTS\JPEGImages-512" --output "D:\\datastore\\TESTS\\yuan-predict"
#python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model "D:\dachuang\goodmodel\yuan-11-7-hui.pth" --input "D:\datastore\TESTS\JPEGImages-512-wb" --output "D:\\datastore\\TESTS\\hui-predict"
#python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model "D:\dachuang\goodmodel\yuan-11-8-1.pth" --input D:\\PythonProject11\\Pytorch-UNet\\data\\imgs --output D:\\PythonProject11\\Pytorch-UNet\\predict_output

#D:\datastore\goodph\wb
#"D:\dachuang\goodmodel\yuan-11-8-1.pth"
# python "D:\\PythonProject11\\Pytorch-UNet\\grouppredict.py" --model "D:\dachuang\goodmodel\11-2-hui.pth" --input D:\\datastore\\goodph\\wb --output D:\\PythonProject11\\Pytorch-UNet\\predict_output-hui --gray
