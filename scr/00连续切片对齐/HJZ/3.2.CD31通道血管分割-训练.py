#%% 导入库
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split, Dataset
import random

#%% 1. 定义网络结构 (标准单头 U-Net)
class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )
    def forward(self, x): 
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, dropout=0.3):
        super().__init__()
        # --- Encoder ---
        self.down1 = DoubleConv(in_ch, base_ch, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_ch, base_ch*2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_ch*2, base_ch*4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base_ch*4, base_ch*8, dropout)
        self.pool4 = nn.MaxPool2d(2)

        # --- Bridge ---
        self.bridge = DoubleConv(base_ch*8, base_ch*16, dropout)

        # --- Decoder (单头) ---
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = DoubleConv(base_ch*8*2, base_ch*8, dropout)
        
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch*4*2, base_ch*4, dropout)
        
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch*2*2, base_ch*2, dropout)
        
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch*2, base_ch, dropout)
        
        # 输出层：1个通道 (血管概率)
        self.out = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        d4 = self.down4(p3); p4 = self.pool4(d4)
        
        # Bridge
        b = self.bridge(p4)
        
        # Decoder
        u = self.up4(b); u = torch.cat([u, d4], 1); u = self.dec4(u)
        u = self.up3(u); u = torch.cat([u, d3], 1); u = self.dec3(u)
        u = self.up2(u); u = torch.cat([u, d2], 1); u = self.dec2(u)
        u = self.up1(u); u = torch.cat([u, d1], 1); u = self.dec1(u)
        
        return self.out(u)

#%% 2. 定义损失函数 (Dice + BCE)
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # BCE 部分
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        
        # Dice 部分
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return bce_loss + dice_loss

# 评价指标：Dice系数
def dice_coeff(pred, target):
    smooth = 1e-5
    pred = (torch.sigmoid(pred) > 0.5).float()
    num = pred.size(0)
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

#%% 3. 数据集加载 (CD31 单通道)
class CD31Dataset(Dataset):
    def __init__(self, root, mode='train'):
        super().__init__()
        # 假设之前的代码生成的图片在 images 文件夹，标签在 labels 文件夹
        # 支持 png 和 tif
        self.imgs = sorted(glob.glob(os.path.join(root, 'images', '*.*')))
        self.lbls = [p.replace('images', 'labels') for p in self.imgs]
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        # 1. 读灰度图 (单通道)
        img = cv2.imread(self.imgs[i], cv2.IMREAD_GRAYSCALE)
        if img is None: # 防御性检查
            raise ValueError(f"无法读取图片: {self.imgs[i]}")
        img = img.astype(np.float32) / 255.0
        img = img[None]  # 增加通道维度 -> [1,H,W]

        # 2. 读标签 (二分类)
        mask = cv2.imread(self.lbls[i], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取标签: {self.lbls[i]}")
        
        # 强制二值化：0为背景，1为血管
        mask = (mask > 0).astype(np.float32)[None] # -> [1,H,W]

        # 3. 数据增强 (仅训练时)
        if self.mode == 'train':
            # 随机翻转
            if random.random() > 0.5:
                img = np.flip(img, axis=2).copy()
                mask = np.flip(mask, axis=2).copy()
            if random.random() > 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

        return torch.from_numpy(img), torch.from_numpy(mask)

#%% 4. 训练主程序
if __name__ == "__main__":
    # 配置
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 路径：指向上一步生成的 Balanced 数据集目录
    # 注意：这里需要指向包含 'images' 和 'labels' 的上一级目录
    # 例如：.../Version_256_CD31_Balanced/train
    train_dir = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/patch/train"
    val_dir   = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/patch/test"
    
    save_dir  = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/models"
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 参数
    epochs = 50
    lr = 1e-4
    batch_size = 4 # 建议调大一点，1太慢且不稳定，只要显存够
    patience = 10

    # 数据集
    train_ds = CD31Dataset(train_dir, mode='train')
    val_ds   = CD31Dataset(val_dir, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型初始化 (in_ch=1)
    net = UNet(in_ch=1).to(device)
    
    # 优化器与损失
    opt = optim.Adam(net.parameters(), lr=lr)
    crit = DiceBCELoss() # 使用混合损失
    scaler = GradScaler() # 混合精度训练

    # 记录
    train_losses = []
    val_dices = []
    best_dice = 0.0
    no_imp = 0

    print(f"开始训练... 训练集: {len(train_ds)}, 验证集: {len(val_ds)}")

    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train")
        for img, mask in pbar:
            img, mask = img.to(device), mask.to(device)
            
            opt.zero_grad()
            
            with autocast():
                pred = net(img)
                loss = crit(pred, mask)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证
        net.eval()
        val_dice_sum = 0.0
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} Val  "):
                img, mask = img.to(device), mask.to(device)
                with autocast():
                    pred = net(img)
                val_dice_sum += dice_coeff(pred, mask).item()
        
        avg_dice = val_dice_sum / len(val_loader)
        val_dices.append(avg_dice)

        print(f"Epoch {epoch} Result: Loss={avg_loss:.4f}, Val_Dice={avg_dice:.4f}")

        # 保存最佳模型
        if avg_dice > best_dice:
            best_dice = avg_dice
            no_imp = 0
            torch.save(net.state_dict(), os.path.join(save_dir, "best_cd31_unet.pth"))
            print(f"--> Model Saved (Dice: {best_dice:.4f})")
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stopping triggered.")
                break

    # 绘图
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.title("Training Loss (Dice+BCE)")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(val_dices, label="Val Dice")
    plt.title("Validation Dice Score")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))

