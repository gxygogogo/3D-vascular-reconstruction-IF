import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from tqdm import tqdm
from aicsimageio import AICSImage
import torch.nn.functional as F
import pandas as pd
from aicsimageio import AICSImage
from skimage.measure import regionprops
import tifffile
# 
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def search_nuclei(mask, x, y):

    h, w = mask.shape
    points = [
        (int(np.floor(y)), int(np.floor(x))),
        (int(np.floor(y)), int(np.ceil(x))),
        (int(np.ceil(y)), int(np.floor(x))),
        (int(np.ceil(y)), int(np.ceil(x)))
    ]
    labels = []
    for py, px in points:
        if 0 <= py < h and 0 <= px < w:
            label = mask[py, px]
            if label != 0:
                labels.append(label)

    return max(labels, key=labels.count) if labels else 0

def load_target_channel(image_path, channel=2):
    img = AICSImage(image_path)
    channel_data = img.get_image_data("YX", T=0, Z=0, C=channel)
    v_max = np.percentile(channel_data, 99.9)
    channel_norm = np.clip(channel_data, 0, v_max) / v_max
    return channel_norm.astype(np.float32)

class SubGraphDataset(Dataset):
    def __init__(self, image, nuclei_mask, fluo_dict, patch_size=128, stride=64):
        """
        fluo_dict: {cell_id: 1 (Pos) or 0 (Neg)}
        """
        self.image = image
        self.nuclei_mask = nuclei_mask
        self.fluo_dict = fluo_dict
        self.patch_size = patch_size
        
        H, W = image.shape
        self.coords = []
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                # 只保留包含细胞的 Patch，纯背景跳过以加速训练
                crop_mask = nuclei_mask[y:y+patch_size, x:x+patch_size]
                if np.sum(crop_mask) > 0:
                    self.coords.append((x, y))
                    
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        x, y = self.coords[idx]
        p_sz = self.patch_size
        
        img_patch = self.image[y:y+p_sz, x:x+p_sz]
        mask_patch = self.nuclei_mask[y:y+p_sz, x:x+p_sz]
        
        # 生成 Training Maps
        # Map 1: Label Map (有标注的地方是0/1，无标注是-1)
        # Map 2: Weight Map (用于监督Loss)
        
        label_map = np.full((p_sz, p_sz), -1.0, dtype=np.float32)
        
        unique_ids = np.unique(mask_patch)
        unique_ids = unique_ids[unique_ids > 0]
        
        has_labeled_data = 0
        
        for uid in unique_ids:
            cell_area = (mask_patch == uid)
            if uid in self.fluo_dict:
                val = self.fluo_dict[uid]
                label_map[cell_area] = val
                has_labeled_data = 1

        img_tensor = torch.from_numpy(img_patch).float().unsqueeze(0) # [1, H, W]
        mask_tensor = torch.from_numpy(mask_patch.astype(np.int32)) # [H, W] (Cell IDs)
        label_tensor = torch.from_numpy(label_map).float() # [H, W] (-1, 0, 1)
        
        return {
            'image': img_tensor,
            'nuclei_mask': mask_tensor, # 带有ID的Mask，用于一致性Loss
            'label_map': label_tensor,
            'has_label': has_labeled_data
        }

# Sparse Consistent Loss 
class SparseConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, pred_prob, label_map, nuclei_mask):
        """
        pred_prob: [B, 1, H, W] (0~1)
        label_map: [B, H, W] (-1: Unlabeled, 0: Neg, 1: Pos)
        nuclei_mask: [B, H, W] (Cell IDs)
        """
        pred = pred_prob.squeeze(1)
        
        # 已标注区域的损失
        valid_mask = (label_map != -1).float()
        # 将-1的label临时变为0计算 BCE，然后用valid_mask过滤
        safe_targets = torch.clamp(label_map, 0, 1) 
        loss_sup = (self.bce(pred, safe_targets) * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        
        # 背景损失：非细胞核区域接近0
        bg_mask = (nuclei_mask == 0).float()
        loss_bg = (pred * bg_mask).mean()
        
        # 未标注的损失：计算其内部预测的方差，使用Pooling近似（效率高一点）
        # 一致性：平滑度约束
        # 强迫预测图在细胞核内部平滑
        dx = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        dy = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
        cell_area = (nuclei_mask > 0).float()
        loss_smooth = (dx * cell_area[:, :, 1:]).mean() + (dy * cell_area[:, 1:, :]).mean()
        
        total_loss = 1.0 * loss_sup + 0.5 * loss_bg + 0.1 * loss_smooth
        return total_loss, loss_sup, loss_bg

# UNet Style
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU())
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec2 = nn.Conv2d(128+64, 64, 3, 1, 1)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec1 = nn.Conv2d(64+32, 32, 3, 1, 1)
        self.final = nn.Conv2d(32, out_ch, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.final(d1))

# =================== 预测全图

def predict_sliding_window_smooth(model, image, patch_size=256, stride=128, batch_size=16, device='cuda:0'):
    """
    使用样条权重进行全图平滑预测
    """
    model.eval()
    H, W = image.shape
    
    x = np.linspace(0, np.pi, patch_size)
    window_1d = np.sin(x)**2 
    window_2d = window_1d[:, None] * window_1d[None, :]
    window_2d = torch.from_numpy(window_2d).float().to(device)
    
    pad_h = patch_size // 2
    pad_w = patch_size // 2
    image_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    pH, pW = image_padded.shape
    
    accumulator = torch.zeros((pH, pW), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((pH, pW), dtype=torch.float32, device=device)
    
    patches_coords = []
    for y in range(0, pH - patch_size + 1, stride):
        for x in range(0, pW - patch_size + 1, stride):
            patches_coords.append((y, x))
            
    print(f"Total patches: {len(patches_coords)}")

    with torch.no_grad():
        for i in tqdm(range(0, len(patches_coords), batch_size), desc="Inferencing"):
            batch_coords = patches_coords[i : i + batch_size]
            batch_imgs = []
            
            for (y, x) in batch_coords:
                patch = image_padded[y:y+patch_size, x:x+patch_size]
                batch_imgs.append(patch)
            

            img_tensor = torch.from_numpy(np.array(batch_imgs)).unsqueeze(1).float().to(device) # [B, 1, H, W]

            preds = model(img_tensor).squeeze(1) # [B, H, W]
            for j, (y, x) in enumerate(batch_coords):
                pred_patch = preds[j]
                accumulator[y:y+patch_size, x:x+patch_size] += pred_patch * window_2d
                weight_sum[y:y+patch_size, x:x+patch_size] += window_2d
                
    final_prob = accumulator / (weight_sum + 1e-6)
    final_prob = final_prob[pad_h:pad_h+H, pad_w:pad_w+W]

    return final_prob.cpu().numpy()

def analyze_cells(prob_map, nuclei_mask, threshold=0.5):
    """
    基于预测概率图统计阳性细胞
    """
    props = regionprops(nuclei_mask)
    results = []
    
    for prop in tqdm(props, desc="Analyzing Cells"):
        coords = prop.coords
        cell_probs = prob_map[coords[:, 0], coords[:, 1]]
        mean_prob = np.mean(cell_probs)
        
        is_positive = mean_prob > threshold
        
        results.append({
            'Cell_ID': prop.label,
            'Centroid_X': prop.centroid[1],
            'Centroid_Y': prop.centroid[0],
            'Mean_Probability': mean_prob,
            'Is_Positive': int(is_positive)
        })
        
    df = pd.DataFrame(results)
    return df





# ############# CD31
config = {
        'output_path': '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/micro_registered_slides_CD31_denoise',
        'image_path': '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/micro_registered_slides_multichannel/HJZ_16-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
        'mask_path': '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/micro_registered_slides_cellpose/HJZ_16-00004-07662-12501.dapi.masks.npy',
        'sample_name': 'HJZ_16-00004-07662-12501',
        'patch_size': 128,  
        'stride': 32,       # 推理步长，patch_size的一半以保证重叠
        'batch_size': 32,
        'lr': 1e-5,
        'epochs': 100,
        'device': 'cuda:1',
        'seed': 315,
        'channel': 1
    }
os.makedirs(config['output_path'], exist_ok=True)
device = config['device']
set_seed(121)
# 
image = tifffile.imread(config['image_path'])
image = image[config['channel'],:, :]
v_max = np.percentile(image, 99.9)
image = np.clip(image, 0, v_max) / v_max
mask = np.load(config['mask_path'])
#

model = SimpleUNet().to(device)
model.load_state_dict(torch.load("/public2/chengrm/3D_TME/mh_data/test_results/CGL_16_FFPE_SubGraph.CD31/best_model.pth", map_location = device))
####################################
  
prob_map = predict_sliding_window_smooth(
    model, 
    image, 
    patch_size=config['patch_size'], 
    stride=config['patch_size']//2, 
    batch_size=config['batch_size'],
    device=device
)

# # 纯粹增强信号
# enhanced = image * (1 + 2 * prob_map)

# 增强信号 + 压制背景
enhanced = image * prob_map * 1.5 + image * 0.2
# enhanced = np.clip(enhanced, 0, 1)

# enhanced = image * prob_map 
enhanced = enhanced / enhanced.max()


# 细胞计数统计
df_results = analyze_cells(prob_map, mask, threshold=0.5)
pos_count = df_results['Is_Positive'].sum()
total_count = len(df_results)
print(f"Positive Cells: {pos_count} / {total_count} ({pos_count/total_count*100:.2f}%)")


# cv2.imwrite(f"{config['output_path']}/prob_map.tif", prob_map.astype(np.float32))
cv2.imwrite(f"{config['output_path']}/enhanced_image2.tif", (enhanced * 255).astype(np.uint8))








df_results.to_csv(f"{config['output_path']}/cell_stats.csv", index=False)

nuclei_mask = (mask > 0).astype(np.uint8) * 255
merged = cv2.merge([nuclei_mask, np.zeros_like(nuclei_mask, dtype=np.uint8), (enhanced * 255).astype(np.uint8)]) # BGR format
cv2.imwrite(f"{config['output_path']}/enhanced_image.with_dapi.jpg", merged)

merged = cv2.merge([np.zeros_like(nuclei_mask, dtype=np.uint8), np.zeros_like(nuclei_mask, dtype=np.uint8), (enhanced * 255).astype(np.uint8)]) # BGR format
cv2.imwrite(f"{config['output_path']}/enhanced_image.without_dapi.jpg", merged)

