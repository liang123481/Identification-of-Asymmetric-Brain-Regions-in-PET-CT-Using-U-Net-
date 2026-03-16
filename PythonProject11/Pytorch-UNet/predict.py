import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from utils.dice_score import dice_loss, dice_coeff

def predict_img(net,#输入的训练好的模型
                full_img,
                device,
                scale_factor=1,#不缩放
                out_threshold=0.5):#预测值大于0.5为真
    net.eval()#Unet修改成评估模式
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))#将输入的原始图像周转成模型可理解的图像
    img = img.unsqueeze(0)#在第一个维度之前插入一个维度
    img = img.to(device=device, dtype=torch.float32)#改变图片格式符合预测使用

    with torch.no_grad():#不能使用梯度计算
        output = net(img).cpu()#将图像数据输入然后输出得到转到cpu
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')#回归原始大小
        if net.n_classes > 1:
            mask = output.argmax(dim=1)#多分类，去概率最高的来使用
        else:
            mask = torch.sigmoid(output) > out_threshold#先转到0-1，大于阈值就是1

    return mask[0].long().squeeze().numpy()#去除批次很通道维度


def get_args():#获取参数函数
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'#提取文件名（去掉前缀）但是加上OUT

    return args.output or list(map(_generate_name, args.input))#指定输出就不用，不然就是按照输入来


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):#检查mask——values第一项是不是列表，彩色掩码[[0,0,0], [255,0,0], [0,255,0]]
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)#设置形状
    elif mask_values == [0, 1]:#2值掩码[0,1]
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:#灰度图【0-255】[0, 128, 255]
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)#三维是指（类别数，长，宽）在数组 mask 的第 0 个维度上，找到每个位置的最大值对应的索引，并将这些索引组成一个新的数组赋值给 mask。

    for i, v in enumerate(mask_values):
        out[mask == i] = v#将掩码数组变成彩色

    return Image.fromarray(out)#将数值变成可见的图片对象


# python Pytorch-UNet\\predict.py -i D:\PythonProject11\Pytorch-UNet\data\imgs\1.jpg -o output.jpg -m D:\\PythonProject11\\Pytorch-UNet\\checkpoints\\checkpoint_epoch5.pth
# python Pytorch-UNet\\predict.py -i D:\\PythonProject11\\Pytorch-UNet\\data\\imgs\\2007_002376.jpg -o output.jpg -m D:\\PythonProject11\\Pytorch-UNet\\checkpoints\\checkpoint_epoch5.pth

# D:\PythonProject11\\Pytorch-UNet\\predict.py
# D:\PythonProject11\Pytorch-UNet\try\0test.png
if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    # net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)#加载训练的模型
    mask_values = state_dict.pop('mask_values', [0, 1])#就是从模型字典提取关键字是key的值·，否就用默认的
    net.load_state_dict(state_dict)#将预设参数加入到模型里面

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        # 显式转为灰度图
        #img = Image.open(filename).convert('L')  # 解决RGBA/RCB转灰度的关键步骤
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
            #$ python Pytorch-UNet\\predict.py -i D:\\PythonProject11\\Pytorch-UNet\\test\\img.png -o output.jpg -m D:\\PythonProject11\\Pytorch-UNet\\checkpoints\\checkpoint_epoch14.pth
#python Pytorch-UNet\\predict.py -i D:\PythonProject11\Pytorch-UNet\data\imgs\0-1.png -o output.jpg -m D:\\PythonProject11\\Pytorch-UNet\\checkpoints\\checkpoint_epoch14.pth
#python Pytorch-UNet\\predict.py -i D:\\PythonProject11\\Pytorch-UNet\\data\\imgs\\4-4.jpg -o output.jpg -m D:\\PythonProject11\\Pytorch-UNet\\checkpoints\\checkpoint_epoch5.pth
#python D:\PythonProject11\Pytorch-UNet\data\imgs\4-4.jpg -o output.jpg -m C:\\Users\\LENOVO\\Desktop\\大创\\goodmodel\\10.13.pth
#"C:\Users\LENOVO\Desktop\大创\goodmodel\10.13.pth"

#python Pytorch-UNet\\predict.py -i D:\PythonProject11\Pytorch-UNet\data\imgs\0-1.png -o output.jpg -m "D:\dachuang\goodmodel\lunkuo.pth"