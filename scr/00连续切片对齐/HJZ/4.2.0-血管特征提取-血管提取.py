import os
import glob
import json
import numpy as np
import tifffile
import cv2
from skimage.measure import label, regionprops
from tqdm import tqdm

# ==========================================
# Part 1: 辅助类与核心功能函数
# ==========================================

class NumpyEncoder(json.JSONEncoder):
    """解决 Numpy 数据类型无法直接存入 JSON 的问题"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def step1_generate_json_and_visualize(mask_dir, output_vis_dir, save_json_path):
    """
    步骤1: 遍历 Mask 文件夹，生成包含 ID 位置信息的 JSON，并生成带有数字标注的可视化图。
    """
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
    
    if not mask_files:
        print(f"[Step 1] 未找到 TIF 文件: {mask_dir}")
        return {}

    os.makedirs(output_vis_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)

    dataset_info = {}
    print(f"[Step 1] 开始扫描并生成可视化图 ({len(mask_files)} 张)...")

    for file_path in tqdm(mask_files):
        filename = os.path.basename(file_path)
        
        try:
            mask = tifffile.imread(file_path)
            mask_bin = (mask > 0).astype(np.uint8)
            
            if np.sum(mask_bin) == 0:
                dataset_info[filename] = {}
                continue
            
            # 连通域标记
            label_img = label(mask_bin, connectivity=2)
            regions = regionprops(label_img)
            
            # 转为 RGB 以便绘制彩色文字
            vis_img = cv2.cvtColor(mask_bin * 255, cv2.COLOR_GRAY2BGR)
            
            slice_data = {}
            for region in regions:
                obj_id = region.label
                # bbox: (min_row, min_col, max_row, max_col) -> (y1, x1, y2, x2)
                bbox = region.bbox 
                centroid = region.centroid
                
                slice_data[obj_id] = {
                    "bbox": bbox,
                    "centroid": centroid,
                    "area": region.area
                }
                
                # 绘制 ID
                cy, cx = centroid
                cv2.putText(vis_img, str(obj_id), (int(cx)-5, int(cy)+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 保存可视化图
            tifffile.imwrite(os.path.join(output_vis_dir, f"vis_{filename}"), vis_img, compression='zlib')
            dataset_info[filename] = slice_data
            
        except Exception as e:
            print(f"处理 {filename} 出错: {e}")

    # 保存 JSON
    with open(save_json_path, 'w') as f:
        json.dump(dataset_info, f, cls=NumpyEncoder, indent=4)
    print(f"[Step 1] 完成！JSON 已保存至: {save_json_path}")


def step2_extract_masks_fullsize(select_dict, json_path, input_mask_dir, output_dir):
    """
    步骤2: 根据 JSON 坐标加速，提取选定的血管 Mask (保持原图尺寸，非选定区域全黑)。
    """
    print(f"[Step 2] 加载 JSON: {json_path}")
    with open(json_path, 'r') as f:
        location_data = json.load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Step 2] 开始提取 Mask ({len(select_dict)} 张)...")
    
    for vis_filename, target_ids in tqdm(select_dict.items()):
        # 文件名处理: 去掉 'vis_' 前缀以匹配原始 Mask
        raw_filename = vis_filename.replace("vis_", "")
        file_path = os.path.join(input_mask_dir, raw_filename)
        
        if raw_filename not in location_data:
            print(f"跳过: JSON 中无 {raw_filename} 信息")
            continue
            
        try:
            # 读取原始 Mask 并重新 Label
            mask = tifffile.imread(file_path)
            if np.sum(mask) == 0: continue
            
            label_img = label(mask > 0, connectivity=2)
            
            # 创建空白底图 (全黑)
            final_mask = np.zeros_like(label_img, dtype=np.uint8)
            slice_info = location_data[raw_filename]
            
            # 利用 bbox 加速提取
            for vid in target_ids:
                if str(vid) in slice_info:
                    y1, x1, y2, x2 = slice_info[str(vid)]['bbox']
                    
                    # 仅在 bbox 范围内操作
                    crop_label = label_img[y1:y2, x1:x2]
                    crop_final = final_mask[y1:y2, x1:x2]
                    
                    # 仅保留目标 ID 的像素
                    crop_final[crop_label == vid] = 255
            
            # 保存
            save_name = f"extracted_{raw_filename}"
            tifffile.imwrite(os.path.join(output_dir, save_name), final_mask, compression='zlib')
            
        except Exception as e:
            print(f"处理 {raw_filename} 出错: {e}")

    print(f"[Step 2] Mask 提取完成！保存在: {output_dir}")


def step3_crop_signal_patches(select_dict, json_path, signal_dir, output_dir):
    """
    步骤3: 根据 JSON 坐标，从 Signal 图像中裁剪出特定的血管区域 (小图 Patch)。
    """
    print(f"[Step 3] 加载 JSON: {json_path}")
    with open(json_path, 'r') as f:
        location_data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Step 3] 开始裁剪 Signal ({len(select_dict)} 张)...")

    for vis_filename, target_ids in tqdm(select_dict.items()):
        # 1. 获取 JSON 键 (原始 Mask 名)
        json_key = vis_filename.replace("vis_", "")
        
        if json_key not in location_data:
            continue
            
        # 2. 推断 Signal 文件名
        # 你的 Mask 是 C15-1-36-mask0000.tif
        # 你的 Signal 是 C15-1-36-signal0000.tif
        # 逻辑：将文件名中的 "mask" 替换为 "signal"
        signal_filename = json_key.replace("mask", "signal")
        signal_path = os.path.join(signal_dir, signal_filename)
        
        if not os.path.exists(signal_path):
            print(f"警告: 找不到 Signal 文件 {signal_filename}")
            continue
        
        try:
            signal_img = tifffile.imread(signal_path)
            slice_info = location_data[json_key]
            
            for vid in target_ids:
                if str(vid) in slice_info:
                    y1, x1, y2, x2 = slice_info[str(vid)]['bbox']
                    
                    # 裁剪 Signal
                    crop = signal_img[y1:y2, x1:x2]
                    
                    # 保存: 原始名_ID_X.tif
                    save_name = f"{os.path.splitext(signal_filename)[0]}_ID_{vid}.tif"
                    tifffile.imwrite(os.path.join(output_dir, save_name), crop, compression='zlib')
                    
        except Exception as e:
            print(f"处理 {signal_filename} 出错: {e}")

    print(f"[Step 3] Signal 裁剪完成！保存在: {output_dir}")


#%% ====================================== 提取 ======================================

 
# --- 1. 路径配置 (请在此处修改) ---
BASE_DIR = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/Single-vascular-reconstruction"
    
# 输入文件夹
MASK_DIR_INPUT = os.path.join(BASE_DIR, "C15-1-36-mask")
SIGNAL_DIR_INPUT = os.path.join(BASE_DIR, "C15-1-36-signal")
    
# 输出文件/文件夹
JSON_PATH = os.path.join(BASE_DIR, "C15-1-36_Locations_Map.json")
VIS_DIR_OUTPUT = os.path.join(BASE_DIR, "C15-1-36-mask-Visualized")       # 存放带ID的可视化图
MASK_Selected_OUTPUT = os.path.join(BASE_DIR, "C15-1-36-mask-Selected")   # 存放提取后的完整Mask
SIGNAL_Patch_OUTPUT = os.path.join(BASE_DIR, "C15-1-36-signal-Cropped-Patches") # 存放裁剪的Signal小图

# --- 2. 筛选字典 (ID 列表) ---
select_vessels = {
    'vis_C15-1-36-mask0000.tif': [1, 3],
    'vis_C15-1-36-mask0001.tif': [1],
    'vis_C15-1-36-mask0002.tif': [2],
    'vis_C15-1-36-mask0003.tif': [1],
    'vis_C15-1-36-mask0004.tif': [1],
    'vis_C15-1-36-mask0005.tif': [1],
    'vis_C15-1-36-mask0006.tif': [3, 6],
    'vis_C15-1-36-mask0007.tif': [2],
    'vis_C15-1-36-mask0008.tif': [2, 5],
    'vis_C15-1-36-mask0009.tif': [2, 3, 5, 8],
    'vis_C15-1-36-mask0010.tif': [3, 4, 8],
    'vis_C15-1-36-mask0011.tif': [2, 3, 7],
    'vis_C15-1-36-mask0012.tif': [3, 4, 7],
    'vis_C15-1-36-mask0013.tif': [3, 4, 7],
    'vis_C15-1-36-mask0014.tif': [2, 4, 6],
    'vis_C15-1-36-mask0015.tif': [4, 5, 7],
    'vis_C15-1-36-mask0016.tif': [7, 8, 10],
    'vis_C15-1-36-mask0017.tif': [5, 6, 8],
    'vis_C15-1-36-mask0018.tif': [4, 5, 7],
    'vis_C15-1-36-mask0019.tif': [3, 4, 6],
    'vis_C15-1-36-mask0020.tif': [4, 5],
    'vis_C15-1-36-mask0021.tif': [3, 4, 5],
    'vis_C15-1-36-mask0022.tif': [5, 6, 7],
    'vis_C15-1-36-mask0023.tif': [2, 3, 4, 6, 7],
    'vis_C15-1-36-mask0024.tif': [4, 5, 6, 8],
    'vis_C15-1-36-mask0025.tif': [5, 7, 8],
    'vis_C15-1-36-mask0026.tif': [2, 4],
    'vis_C15-1-36-mask0027.tif': [5, 8],
    'vis_C15-1-36-mask0028.tif': [5, 7],
    'vis_C15-1-36-mask0029.tif': [5, 7, 6],
    'vis_C15-1-36-mask0030.tif': [3, 4, 5],
    'vis_C15-1-36-mask0031.tif': [6, 7, 8],
    'vis_C15-1-36-mask0032.tif': [5, 7, 8],
    'vis_C15-1-36-mask0033.tif': [4, 5, 6],
    'vis_C15-1-36-mask0034.tif': [4, 7, 8],
    'vis_C15-1-36-mask0035.tif': [3, 4, 5]
}


# 步骤 1: 生成 JSON 和 可视化图 (如果已经做过，可以注释掉这一行)
# step1_generate_json_and_visualize(MASK_DIR_INPUT, VIS_DIR_OUTPUT, JSON_PATH)
    
# 步骤 2: 根据 JSON 提取 Mask (原图大小，仅保留选中血管)
step2_extract_masks_fullsize(select_vessels, JSON_PATH, MASK_DIR_INPUT, MASK_Selected_OUTPUT)
    
# 步骤 3: 根据 JSON 裁剪 Signal (小图 Patch)
step3_crop_signal_patches(select_vessels, JSON_PATH, SIGNAL_DIR_INPUT, SIGNAL_Patch_OUTPUT)
