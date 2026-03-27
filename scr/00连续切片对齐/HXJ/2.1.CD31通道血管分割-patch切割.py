import os
import cv2
import random
import pickle
import numpy as np
import tifffile
import shutil

#%% 1. 参数设置
patch_size = 256
rotations  = [0, 90, 180, 270] # 血管图进行旋转增强
sample = 'HJZ_5-00004-02772-17487'

# ================= 修改区 =================
# 原图路径 (CD31 单通道)
img_path  = '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-02772-17487/Micro-registration/micro_registered_slides_CD31/HJZ_5-00004-02772-17487_uint16_enhanced.tif' 

# 掩码图路径
mask_path = '/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/mask/HJZ_5-00004-02772-174878_CD31_mask.tif' 
# ==========================================

# 输出目录
base_out = f'/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/patch/{sample}_{patch_size}_CD31_Balanced'

out_img_dir  = os.path.join(base_out, 'images')
out_lbl_dir  = os.path.join(base_out, 'labels')
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

#%% 2. 读取数据
print(f"正在读取原图: {img_path}")
CD31_raw = tifffile.imread(img_path)
if CD31_raw.ndim == 3: CD31_raw = np.squeeze(CD31_raw)
CD31 = cv2.normalize(CD31_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

print(f"正在读取掩码: {mask_path}")
mask_raw = tifffile.imread(mask_path)
if mask_raw.ndim == 3: mask_raw = np.squeeze(mask_raw)

# 确保掩码是二值的 (0和1)
mask = (mask_raw > 0).astype(np.uint8)

H, W = CD31.shape
if mask.shape != (H, W):
    raise ValueError(f"尺寸不匹配! 原图: {CD31.shape}, Mask: {mask.shape}")

#%% 3. 第一遍扫描：统计坐标
print("正在扫描全图以统计分类...")
pos_locs = [] # 存放血管 Patch 的 (y, x)
neg_locs = [] # 存放背景 Patch 的 (y, x)

n_rows = H // patch_size
n_cols = W // patch_size

for i in range(n_rows):
    for j in range(n_cols):
        y0, x0 = i*patch_size, j*patch_size
        y1, x1 = y0+patch_size, x0+patch_size
        
        # 只需要看 Mask 就能判断类别
        mask_p = mask[y0:y1, x0:x1]
        
        if mask_p.any():
            pos_locs.append((y0, x0))
        else:
            neg_locs.append((y0, x0))

# 计算数量
n_pos_raw = len(pos_locs)
n_neg_raw = len(neg_locs)

# 血管样本扩增后数量 (x4)
n_pos_aug = n_pos_raw * len(rotations)

print(f"扫描结果：")
print(f"  - 原始血管 Patch 数: {n_pos_raw} (扩增后预计产生 {n_pos_aug} 张图片)")
print(f"  - 原始背景 Patch 数: {n_neg_raw}")

# 确定需要保留的背景数量 (目标 1:1)
target_neg = n_pos_aug

if n_neg_raw > target_neg:
    print(f"  -> 背景过多，将随机抽取 {target_neg} 个以保持 1:1 比例...")
    random.seed(42) # 固定种子保证可复现
    kept_neg_locs = set(random.sample(neg_locs, target_neg))
else:
    print(f"  -> 背景数量不足或刚好 ({n_neg_raw} vs {target_neg})，保留所有背景。")
    kept_neg_locs = set(neg_locs)

#%% 4. 第二遍处理：切图并保存
patch_info = []
idx = 0

print("开始生成 Patch 文件...")

for i in range(n_rows):
    for j in range(n_cols):
        y0, x0 = i*patch_size, j*patch_size
        y1, x1 = y0+patch_size, x0+patch_size
        
        # 检查当前坐标是否在我们的“保留名单”里
        coord = (y0, x0)
        
        # 判断类别
        mask_p = mask[y0:y1, x0:x1]
        has_vessel = mask_p.any()
        
        is_pos = has_vessel
        is_neg_kept = (not has_vessel) and (coord in kept_neg_locs)
        
        # 如果既不是血管，也不是被选中的背景，就跳过
        if not (is_pos or is_neg_kept):
            continue
            
        # 开始切图
        img_p = CD31[y0:y1, x0:x1]
        
        # 决定旋转角度
        # 血管：旋转4次；背景：不旋转 (0度)
        angles = rotations if is_pos else [0]
        
        for angle in angles:
            M = cv2.getRotationMatrix2D((patch_size/2, patch_size/2), angle, 1.0)
            im_r  = cv2.warpAffine(img_p,  M, (patch_size, patch_size))
            # 标签旋转 (保持0/1值)
            lbl_r = cv2.warpAffine(mask_p, M, (patch_size, patch_size), flags=cv2.INTER_NEAREST)

            # 保存
            fname = f"{sample}-{idx:05d}_{angle}.png"
            cv2.imwrite(os.path.join(out_img_dir, fname), im_r)
            cv2.imwrite(os.path.join(out_lbl_dir, fname), lbl_r)

            patch_info.append({
                'idx': idx,
                'angle': angle,
                'x0': x0, 'y0': y0,
                'img_path': os.path.join(out_img_dir, fname),
                'lbl_path': os.path.join(out_lbl_dir, fname),
                'img_array': im_r.copy(),
                'lbl_array': lbl_r.copy(),
                'subset': None,
                'class': 1 if is_pos else 0 # 记录类别方便后续统计
            })
        
        idx += 1

print(f"切图完成，共保存 {len(patch_info)} 张图片。")

#%% 5. 划分 Train/Test
groups = {}
for info in patch_info:
    groups.setdefault(info['idx'], []).append(info)

# 统计实际生成的类别数量
count_0 = sum(1 for info in patch_info if info['class'] == 0)
count_1 = sum(1 for info in patch_info if info['class'] == 1)
print(f"最终数据集统计: 背景图片 {count_0} 张, 血管图片 {count_1} 张 (比例 {count_0}:{count_1})")

# 准备划分
idx2cls = {idx_key: infos[0]['class'] for idx_key, infos in groups.items()}
class_to_idxs = {}
for idx_key, cls in idx2cls.items():
    class_to_idxs.setdefault(cls, []).append(idx_key)

# 每类抽 10% idx 做测试
random.seed(42)
test_ids = set()
for cls, ids in class_to_idxs.items():
    n_test = max(1, int(len(ids)*0.1))
    test_ids.update(random.sample(ids, n_test))

# 建目录并复制
for split in ('train','test'):
    os.makedirs(os.path.join(base_out, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_out, split, 'labels'), exist_ok=True)

print("正在划分并复制文件...")
for idx_key, infos in groups.items():
    split = 'test' if idx_key in test_ids else 'train'
    for info in infos:
        fn = os.path.basename(info['img_path'])
        shutil.copy(info['img_path'],  os.path.join(base_out, split, 'images', fn))
        shutil.copy(info['lbl_path'],  os.path.join(base_out, split, 'labels', fn))
        info['subset'] = split

#%% 6. 保存 info
with open(os.path.join(base_out, 'patch_info.pkl'), 'wb') as f:
    pickle.dump(patch_info, f)

print(f"处理完毕！输出目录: {base_out}")

