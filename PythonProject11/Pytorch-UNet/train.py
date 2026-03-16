import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')
os.environ["WANDB_API_KEY"] = 'fc06a621e7cf988cfb1a80b303f37baadc7247c1'
os.environ["WANDB_MODE"] = "offline"


def train_model(#模型训练函数
        model,#Unet类
        device,#训练工具
        epochs: int = 20,#轮数
        batch_size: int = 2,#次送入模型训练的样本数量,默认1
        learning_rate: float = 1e-5,#学习率，默认1e-5
        val_percent: float = 0.1,#可以尝试在这个上面做文章
        save_checkpoint: bool = True,
        img_scale: float = 0.5,#图像缩放比例
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,#动量,防止震荡
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)#验证的个数
    n_train = len(dataset) - n_val#训练个数
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))#固定训练集和验证集的随机数序列

    # 3. Create data loaders，调用CPU核心进行操作
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)#一次加载样本数量，调用进程数，是都启用3GPU
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)#训练集打乱顺序，
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)#测试集不用打乱顺序

    # (Initialize logging)记录训练日志
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)#优化器配置，让学习误差最小化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score，最大化dice系数，五轮没有显著提升之后就调整学习率
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)#加速训练速度
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()#定义损失函数
    global_step = 0
    #python Pytorch-UNet\\predict.py -i D:\\PythonProject11\\Pytorch-UNet\\data\\imgs\\2010_002310.jpg -o output.jpg -m D:\\PythonProject11\\Pytorch-UNet\\checkpoints\\checkpoint_epoch16.pth

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:#创建一个可视化的训练进度
            for batch in train_loader:#对训练数据按照批次输入
                images, true_masks = batch['image'], batch['mask']#提取张量输入模型
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
#上面这个是检测输入图片的通道数是否符合预期
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)#让图像数据满足模型计算的要求，让数据在位置、类型、存储格式
                true_masks = true_masks.to(device=device, dtype=torch.long)#让真实掩码数据在存储设备和数据类型上与模型及输入数据保持一致

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    #混合精度训练
                    masks_pred = model(images)
                    #将图片传入模型训练，得到预测的分割掩码
                    if model.n_classes == 1:#二分类
                        #将 交叉熵损失 和 Dice 损失 相加
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        #多分类任务
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                #反向传播与参数更新的核心流程
                optimizer.zero_grad(set_to_none=True)#清空模型所有参数的梯度
                grad_scaler.scale(loss).backward()#对损失进行缩放后，执行反向传播计算梯度。
                grad_scaler.unscale_(optimizer)#将放大的梯度恢复到原始比例。
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)#梯度裁剪，防止梯度爆炸。
                grad_scaler.step(optimizer)#使用处理后的梯度更新模型参数
                grad_scaler.update()#更新梯度缩放器的缩放比例。

                pbar.update(images.shape[0])#更新进度条
                global_step += 1#每处理完一个批次，步数 + 1
                epoch_loss += loss.item()#计算损失
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})#在进度条的末尾动态显示当前批次的loss

                # Evaluation round
                division_step = (n_train // (5 * batch_size))#训练步数每5步进行一轮验证
                if division_step > 0:
                    if global_step % division_step == 0:#是否达到验证轮
                        histograms = {}#开一个空字典
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():#检测参数并且将正常数据加入直方图
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():##检测参数梯度并且将正常数据加入直方图
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)#评估测试集的dice系数
                        scheduler.step(val_score)#根据现在的dice系数选择调整学习率

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:#记录信息
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:#记录完整一轮的效果
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()#当前的所有可学习参数
            state_dict['mask_values'] = dataset.mask_values#记录掩码数量
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            val_score = evaluate(model, val_loader, device, amp)#对一整轮进行一个总结
            scheduler.step(val_score)#调整学习率

            # 新增：直接在控制台打印Dice系数（直观可见）
            print(f"Epoch {epoch} | Step {global_step} | Validation Dice Coefficient: {val_score:.4f}")
            # 保留原有日志（可选，让日志文件也记录）
            logging.info(f'Epoch {epoch} | Validation Dice score: {val_score:.4f}')


def get_args():#命令行
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = UNet(n_channels=1, n_classes=2, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)#设置模型内存存储格式的操作，目的是优化计算效率

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:#继续训练的
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)#将模型部署到指定计算设备
    try:
        train_model(
            model=model,
            epochs=20,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
        #cd /d/PythonProject11/Pytorch-UNet    python train.py -e 15 -b 2 --amp
        #

        # / --learning-rate 5e-6 --validation 0.1 --scale 0.5
        #https://github.com/4uiiurz1/pytorch-nested-unet.git
        #python train.py --epochs 10 --learning-rate 2e-5
        # python train.py --load D:/PythonProject11/Pytorch-UNet/checkpoints/checkpoint_epoch8.pth --epochs 15--lr 5e-6 --amp
