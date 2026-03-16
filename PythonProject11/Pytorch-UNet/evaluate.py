import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()# 将模型设为评估模式
    num_val_batches = len(dataloader)#批次大小
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):#启用自动混合精度,结束自动关闭
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):#用 tqdm 库为验证集的迭代过程添加一个进度条
            image, mask_true = batch['image'], batch['mask']#从字典取出图片和掩码的张量

            # move images and labels to correct device and type，就是将图片和掩码转换成可以评估和符合当前机器的格式
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)#mask_pred = net.forward(image)等价就是将图片输入得到结果输出
            if net.n_classes == 1:#二分类任务
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'#检测是否掩码在0-1
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()#将掩码值输入sigmoid，将大于0.5设置为目标1，其他设置0为背景
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)#这是一个计算 Dice 系数的函数，false是函数会先为批量中的每个样本单独计算 Dice 系数再去平均值
            else:#多分类任务
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()#转成one-hot编码格式，并且修改张量的格式符合要求
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)#去掉预测掩码和真实掩码的背景

    net.train()#调整为训练模式
    return dice_score / max(num_val_batches, 1)#计算平均dice系数
