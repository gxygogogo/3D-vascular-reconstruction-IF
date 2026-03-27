import os
import re
import cv2
import torch
import tifffile
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


#%% --- 1. 定义网络结构 (必须与训练脚本完全一致) ---
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
    def __init__(self, in_ch=1, base_ch=64, dropout=0.3): # 如果你修改了 base_ch，这里也要改
        super().__init__()
        self.down1 = DoubleConv(in_ch, base_ch, dropout); self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_ch, base_ch*2, dropout); self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_ch*2, base_ch*4, dropout); self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base_ch*4, base_ch*8, dropout); self.pool4 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(base_ch*8, base_ch*16, dropout)
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2); self.dec4 = DoubleConv(base_ch*8*2, base_ch*8, dropout)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2); self.dec3 = DoubleConv(base_ch*4*2, base_ch*4, dropout)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2); self.dec2 = DoubleConv(base_ch*2*2, base_ch*2, dropout)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2); self.dec1 = DoubleConv(base_ch*2, base_ch, dropout)
        self.out = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        d4 = self.down4(p3); p4 = self.pool4(d4)
        b = self.bridge(p4)
        u = self.up4(b); u = torch.cat([u, d4], 1); u = self.dec4(u)
        u = self.up3(u); u = torch.cat([u, d3], 1); u = self.dec3(u)
        u = self.up2(u); u = torch.cat([u, d2], 1); u = self.dec2(u)
        u = self.up1(u); u = torch.cat([u, d1], 1); u = self.dec1(u)
        return self.out(u)

# --- 2. 预测函数 ---
def predict_whole_slide(model, img_path, save_path, patch_size=256, stride=128, device='cuda'):
    '''
    predict_whole_slide 的 Docstring
    
    :param model: 训练好模型路径
    :param img_path: 原图路径
    :param save_path: 预测掩码保存路径
    :param patch_size: patch大小
    :param stride: 窗口滑行的步长
    :param device: GPU编号
    '''
    print(f"正在读取大图: {img_path}")
    
    # 读取图片
    image = tifffile.imread(img_path)
    if image.ndim == 3: image = np.squeeze(image)
    
    # 预处理：归一化到 0-1 (与训练时一致)
    # 注意：训练时是 cv2.imread -> /255.0
    # 这里如果原图是 uint16，先转 uint8 再归一化，或者直接归一化
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 归一化为 float32 [0, 1]
    image_norm = image.astype(np.float32) / 255.0
    
    H, W = image_norm.shape
    print(f"图像尺寸: {H} x {W}")
    
    # 初始化概率图和计数图 (用于重叠平均)
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    
    # 生成 Patch 坐标列表
    patches_coords = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = y
            x1 = x
            # 如果超出边界，往回退
            if y1 + patch_size > H: y1 = H - patch_size
            if x1 + patch_size > W: x1 = W - patch_size
            patches_coords.append((y1, x1))
    
    print(f"总 Patch 数: {len(patches_coords)}, 开始推理...")
    
    model.eval()
    
    # 批量推理
    with torch.no_grad():
        for i in tqdm(range(0, len(patches_coords), batch_size)):
            batch_coords = patches_coords[i : i + batch_size]
            batch_imgs = []
            
            # 准备 Batch 数据
            for (y, x) in batch_coords:
                patch = image_norm[y : y+patch_size, x : x+patch_size]
                batch_imgs.append(patch)
            
            # 转 Tensor: [B, 1, H, W]
            batch_tensor = np.array(batch_imgs)[:, None, :, :]
            batch_tensor = torch.from_numpy(batch_tensor).to(device)
            
            # 预测
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy() # [B, 1, H, W]
            
            # 填回大图
            for idx, (y, x) in enumerate(batch_coords):
                prob = probs[idx, 0, :, :]
                prob_map[y : y+patch_size, x : x+patch_size] += prob
                count_map[y : y+patch_size, x : x+patch_size] += 1
                
    # 计算平均概率
    print("正在合并结果...")
    final_prob = prob_map / np.maximum(count_map, 1.0)
    
    # 二值化 (阈值 0.5)
    final_mask = (final_prob > 0.5).astype(np.uint8) * 255
    
    # 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"正在保存 Mask 到: {save_path}")
    tifffile.imwrite(save_path, final_mask, compression='zlib')
    print("完成！")

#%% --- 3. 主程序 ---

# 4. 参数设置
patch_size = 256     # 必须与训练时一致
stride = 128         # 滑动步长 (建议 patch_size 的 1/2，即 50% 重叠，效果最好)
batch_size = 4      # 推理时的批次大小 (显存够大可调大)
device_id = "0"      # 使用的显卡ID

# ===========================================

if __name__ == "__main__":
    roi_list = ['ROI-00004-07662-12501', 'ROI-00004-02772-17487']
    for roi in roi_list:
        file_list = os.listdir(f'/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/{roi}/Micro-registration/micro_registered_slides_CD31/')

        for i in file_list:

            print(i)

            # 1. 输入图片路径 (CD31 单通道大图)
            input_image_path = f"/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/{roi}/Micro-registration/micro_registered_slides_CD31/{i}"

            # 2. 模型权重路径 (你训练好的 .pth 文件)
            model_path = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/models/best_cd31_unet.pth"

            pat = re.compile(r'HJZ_\d+-\d+-\d+-\d+')
            m = pat.search(i)
            base = m.group(0)

            # 3. 输出路径
            output_mask_path = f"/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/prediction/{base}_pred_mask.tif"

            os.environ["CUDA_VISIBLE_DEVICES"] = device_id

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 加载模型
            print("加载模型...")
            # 注意：如果训练时修改了 base_ch=32，这里也要改
            net = UNet(in_ch=1, base_ch=64).to(device) 

            # 加载权重
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"找不到模型文件: {model_path}")

            state_dict = torch.load(model_path, map_location=device)
            net.load_state_dict(state_dict)

            # 开始预测
            predict_whole_slide(
                net, 
                input_image_path, 
                output_mask_path, 
                patch_size=patch_size, 
                stride=stride, 
                device=device
            )

