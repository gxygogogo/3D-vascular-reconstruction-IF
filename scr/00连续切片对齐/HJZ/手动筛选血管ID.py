import os
import glob
import json
import numpy as np
import tifffile
import cv2  # 需要安装 opencv-python
from skimage.measure import label, regionprops
from tqdm import tqdm

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def process_masks_and_visualize(mask_dir, output_vis_dir, save_json_path=None):
    """
    生成血管位置字典，并同时生成带有ID标注的可视化TIF图。
    """
    # 1. 获取文件
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
    
    if not mask_files:
        print("未找到 TIF 文件，请检查路径。")
        return {}

    # 确保可视化输出目录存在
    os.makedirs(output_vis_dir, exist_ok=True)

    dataset_info = {}

    print(f"开始处理 {len(mask_files)} 个文件...")

    for file_path in tqdm(mask_files):
        filename = os.path.basename(file_path)
        
        try:
            # --- 读取与预处理 ---
            mask = tifffile.imread(file_path)
            
            # 转为二值 (0 或 1)
            mask_bin = (mask > 0).astype(np.uint8)
            
            if np.sum(mask_bin) == 0:
                dataset_info[filename] = {}
                continue
            
            # 连通域标记
            label_img = label(mask_bin, connectivity=2)
            regions = regionprops(label_img)
            
            # --- 准备可视化图像 (RGB) ---
            # 将二值图转为 BGR 格式 (OpenCV使用BGR)，方便画彩色字
            # 背景黑色，血管白色
            vis_img = cv2.cvtColor(mask_bin * 255, cv2.COLOR_GRAY2BGR)
            
            slice_data = {}
            
            for region in regions:
                obj_id = region.label
                bbox = region.bbox
                centroid = region.centroid # (y, x)
                area = region.area
                
                # 记录数据
                slice_data[obj_id] = {
                    "bbox": bbox,
                    "centroid": centroid,
                    "area": area
                }
                
                # --- 在图上绘制 ID ---
                # skimage 的 centroid 是 (row/y, col/x)，OpenCV 画图需要 (x, y)
                cy, cx = centroid
                
                # 绘制文字
                # 参数: 图片, 文字内容, 坐标(整数), 字体, 大小, 颜色(B,G,R), 粗细
                text_content = str(obj_id)
                
                # 为了防止字太小看不清，根据 bbox 大小动态调整一点字体大小（可选）
                # 这里简单设定为 0.5 到 1.0 之间
                font_scale = 0.5 
                color = (0, 0, 255) # 红色 (B, G, R)
                thickness = 1
                
                # 稍微偏移一点，让字写在中心点附近
                text_pos = (int(cx) - 5, int(cy) + 5) 
                
                cv2.putText(vis_img, text_content, text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            # --- 保存可视化结果 ---
            save_path = os.path.join(output_vis_dir, f"vis_{filename}")
            tifffile.imwrite(save_path, vis_img, compression='zlib') # 使用压缩节省空间
            
            dataset_info[filename] = slice_data
            
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 保存 JSON
    if save_json_path:
        with open(save_json_path, 'w') as f:
            json.dump(dataset_info, f, cls=NumpyEncoder, indent=4)
        print(f"JSON 已保存至: {save_json_path}")

    return dataset_info

# ================= 运行配置 =================

INPUT_MASK_DIR = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/Single-vascular-reconstruction/C15-1-36-mask"
# 新建一个文件夹存放画了ID的图，避免覆盖原图
OUTPUT_VIS_DIR = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/Single-vascular-reconstruction/C15-1-36-mask-Visualized"
OUTPUT_JSON_PATH = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/Single-vascular-reconstruction/C15-1-36_Locations_Map.json"

process_masks_and_visualize(INPUT_MASK_DIR, OUTPUT_VIS_DIR, OUTPUT_JSON_PATH)



#%% 根据ID获取血管
import os
import json
import numpy as np
import tifffile
from skimage.measure import label
from tqdm import tqdm

def extract_selected_vessels(select_dict, json_path, input_mask_dir, output_dir):
    """
    根据选择字典和JSON位置信息，提取特定ID的血管（保持原图尺寸）。
    
    参数:
        select_dict: {文件名: [ID列表]} 的字典
        json_path: 之前生成的包含bbox信息的JSON文件路径
        input_mask_dir: 原始二值掩码存放文件夹
        output_dir: 结果保存路径
    """
    # 1. 加载 JSON 数据
    print(f"正在加载坐标字典: {json_path}")
    with open(json_path, 'r') as f:
        location_data = json.load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始提取 {len(select_dict)} 张切片中的目标血管...")
    
    # 2. 遍历字典
    for vis_filename, target_ids in tqdm(select_dict.items()):
        # --- A. 文件名匹配 ---
        # 这里的键通常带 "vis_"，需要去掉以匹配 JSON 和 原始文件名
        key_filename = vis_filename.replace("vis_", "")
        
        # 拼接原始文件路径
        file_path = os.path.join(input_mask_dir, key_filename)
        
        if not os.path.exists(file_path):
            print(f"警告: 找不到原始文件 {key_filename}，跳过。")
            continue
            
        if key_filename not in location_data:
            print(f"警告: JSON中没有 {key_filename} 的位置信息，跳过。")
            continue

        try:
            # --- B. 读取原始掩码并重新标记 ---
            # 必须重新 label，因为原始图是二值图，无法区分 ID
            mask = tifffile.imread(file_path)
            mask_bin = mask > 0 
            
            if np.sum(mask_bin) == 0:
                continue
                
            # 这一步必须与生成 JSON 时完全一致 (connectivity=2)
            label_img = label(mask_bin, connectivity=2)
            
            # --- C. 创建空白底图 (原图尺寸) ---
            final_mask = np.zeros_like(label_img, dtype=np.uint8)
            
            # 获取该图的 JSON 信息
            slice_info = location_data[key_filename]
            
            # --- D. 利用 JSON bbox 加速提取 ---
            for vid in target_ids:
                str_id = str(vid)
                
                if str_id in slice_info:
                    # 从 JSON 获取边界框 [y1, x1, y2, x2]
                    y1, x1, y2, x2 = slice_info[str_id]['bbox']
                    
                    # 关键优化：只在 bbox 范围内操作，而不是全图扫描
                    # 1. 切出 label 局部
                    crop_label = label_img[y1:y2, x1:x2]
                    
                    # 2. 切出 结果图 局部
                    crop_final = final_mask[y1:y2, x1:x2]
                    
                    # 3. 仅将局部中等于目标 ID 的像素设为 255
                    # 这样即使 bbox 里包含了别的血管，也不会被误选
                    crop_final[crop_label == vid] = 255
                    
                    # 4. 因为 crop_final 是 final_mask 的切片引用（View），
                    # 修改 crop_final 会自动更新 final_mask，无需手动赋值回去
                else:
                    print(f"提示: ID {vid} 不在 {key_filename} 的 JSON 记录中")
            
            # --- E. 保存结果 ---
            save_name = f"extracted_{key_filename}"
            save_path = os.path.join(output_dir, save_name)
            
            tifffile.imwrite(save_path, final_mask, compression='zlib')
            
        except Exception as e:
            print(f"处理 {key_filename} 时出错: {e}")

    print(f"提取完成！结果保存在: {output_dir}")



def crop_signal_by_json_coordinates(select_dict, json_path, signal_dir, output_dir):
    """
    根据 JSON 中的 bbox 坐标，从 Signal 图像中裁剪出特定的血管区域（矩形）。
    """
    # 1. 加载 JSON 数据
    print(f"正在加载坐标字典: {json_path}")
    with open(json_path, 'r') as f:
        location_data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始裁剪 {len(select_dict)} 张图像中的目标...")

    # 2. 遍历你的选择字典
    for vis_filename, target_ids in tqdm(select_dict.items()):
        
        # --- A. 文件名匹配逻辑 ---
        # 1. 去掉 vis_ 前缀以匹配 JSON 的键 (假设 JSON 键是原始 mask 文件名)
        json_key = vis_filename.replace("vis_", "")
        
        # 2. 检查 JSON 中是否有这张图的信息
        if json_key not in location_data:
            print(f"跳过: JSON 中没有文件 {json_key} 的记录")
            continue
            
        # 3. 推断 Signal 文件名 (根据你之前的逻辑去掉 -mask)
        signal_filename = json_key.replace("-mask", "")
        signal_path = os.path.join(signal_dir, signal_filename)
        
        # 如果 signal 文件不存在，尝试一下原名
        if not os.path.exists(signal_path):
             if os.path.exists(os.path.join(signal_dir, json_key)):
                 signal_path = os.path.join(signal_dir, json_key)
             else:
                 print(f"警告: 找不到 Signal 文件 {signal_filename}")
                 continue
        
        try:
            # --- B. 读取 Signal 图像 (只读一次) ---
            signal_img = tifffile.imread(signal_path)
            
            # 获取该图的所有血管位置信息
            slice_info = location_data[json_key]
            
            # --- C. 遍历目标 ID 进行裁剪 ---
            for vid in target_ids:
                # JSON 中的键通常是字符串，确保转换
                str_id = str(vid)
                
                if str_id in slice_info:
                    # 获取边界框: [min_row, min_col, max_row, max_col] -> [y1, x1, y2, x2]
                    bbox = slice_info[str_id]['bbox']
                    y1, x1, y2, x2 = bbox
                    
                    # === 核心步骤：数组切片 ===
                    # 直接从原图中切出这一块
                    crop = signal_img[y1:y2, x1:x2]
                    
                    # === 保存 ===
                    # 命名格式: 原始名_ID_血管号.tif
                    # 例如: C15-1-36-0000_ID_1.tif
                    save_name = f"{os.path.splitext(signal_filename)[0]}_ID_{vid}.tif"
                    save_path = os.path.join(output_dir, save_name)
                    
                    tifffile.imwrite(save_path, crop, compression='zlib')
                else:
                    print(f"提示: ID {vid} 不在 {json_key} 的记录中")
                    
        except Exception as e:
            print(f"处理 {signal_filename} 时出错: {e}")

    print(f"裁剪完成！结果保存在: {output_dir}")



select = {'vis_C15-1-36-mask0000.tif':[1, 3],
          'vis_C15-1-36-mask0001.tif':[1],
          'vis_C15-1-36-mask0002.tif':[2],
          'vis_C15-1-36-mask0003.tif':[1],
          'vis_C15-1-36-mask0004.tif':[1],
          'vis_C15-1-36-mask0005.tif':[1],
          'vis_C15-1-36-mask0006.tif':[3, 6],
          'vis_C15-1-36-mask0007.tif':[2],
          'vis_C15-1-36-mask0008.tif':[2, 5],
          'vis_C15-1-36-mask0009.tif':[2, 3, 5, 8],
          'vis_C15-1-36-mask0010.tif':[3, 4, 8],
          'vis_C15-1-36-mask0011.tif':[2, 3, 7],
          'vis_C15-1-36-mask0012.tif':[3, 4, 7],
          'vis_C15-1-36-mask0013.tif':[3, 4, 7],
          'vis_C15-1-36-mask0014.tif':[2, 4, 6],
          'vis_C15-1-36-mask0015.tif':[4, 5, 7],
          'vis_C15-1-36-mask0016.tif':[7, 8, 10],
          'vis_C15-1-36-mask0017.tif':[5, 6, 8],
          'vis_C15-1-36-mask0018.tif':[4, 5, 7],
          'vis_C15-1-36-mask0019.tif':[3, 4, 6],
          'vis_C15-1-36-mask0020.tif':[4, 5],
          'vis_C15-1-36-mask0021.tif':[3, 4, 5],
          'vis_C15-1-36-mask0022.tif':[5, 6, 7],
          'vis_C15-1-36-mask0023.tif':[2, 3, 4, 6, 7],
          'vis_C15-1-36-mask0024.tif':[4, 5, 6, 8],
          'vis_C15-1-36-mask0025.tif':[5, 7, 8],
          'vis_C15-1-36-mask0026.tif':[2, 4],
          'vis_C15-1-36-mask0027.tif':[5, 8],
          'vis_C15-1-36-mask0028.tif':[5, 7],
          'vis_C15-1-36-mask0029.tif':[5, 7, 6],
          'vis_C15-1-36-mask0030.tif':[3, 4, 5],
          'vis_C15-1-36-mask0031.tif':[6, 7, 8],
          'vis_C15-1-36-mask0032.tif':[5, 7, 8],
          'vis_C15-1-36-mask0033.tif':[4, 5, 6],
          'vis_C15-1-36-mask0034.tif':[4, 7, 8],
          'vis_C15-1-36-mask0035.tif':[3, 4, 5]}

def extract_vessel_signal(raw_img_path, mask_path, save_path):
    try:
        # 1. 读取原始 CD31 图片 (支持 uint16 高动态范围)
        raw_img = tifffile.imread(raw_img_path)
        if raw_img.ndim == 3: raw_img = np.squeeze(raw_img)
        
        # 2. 读取调整后的掩码
        mask = tifffile.imread(mask_path)
        if mask.ndim == 3: mask = np.squeeze(mask)
        
        # 3. 尺寸检查
        if raw_img.shape != mask.shape:
            print(f"Error: 尺寸不匹配! 原图 {raw_img.shape} vs 掩码 {mask.shape}")
            return

        # 4. 提取信号 (核心步骤)
        # 将掩码转为 0 和 1 的矩阵
        # 注意：astype(raw_img.dtype) 确保掩码的数据类型和原图一致，避免计算报错
        binary_mask = (mask > 0).astype(raw_img.dtype)
        
        # 矩阵相乘：Mask为0的地方变0，Mask为1的地方保留原值
        extracted_img = raw_img * binary_mask
        
        total_intensity = np.sum(raw_img[mask > 0]) # 只取掩码区域的像素
        pixel_count = np.count_nonzero(mask)

        if pixel_count > 0:
            mfi = total_intensity / pixel_count
            print(f"{os.path.basename(raw_img_path)} - MFI: {mfi:.2f}")
        else:
            print(f"{os.path.basename(raw_img_path)} - MFI: 0 (无血管)")

        # 5. 保存结果
        # 使用 zlib 压缩可以大大减小文件体积，且不丢失数据
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tifffile.imwrite(save_path, extracted_img, compression='zlib')
        
    except Exception as e:
        print(f"处理出错 {os.path.basename(raw_img_path)}: {e}")


# 2. 路径设置
# JSON 文件路径
JSON_PATH = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/Single-vascular-reconstruction/C15-1-36_Locations_Map.json"

# 原始 mask 所在的文件夹
INPUT_MASK_DIR = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/Single-vascular-reconstruction/C15-1-36-mask"

# 结果输出文件夹
OUTPUT_mask_DIR = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/Single-vascular-reconstruction/C15-1-36-mask-Selected"

# 3. 运行
extract_selected_vessels(select, JSON_PATH, INPUT_MASK_DIR, OUTPUT_mask_DIR)




# 2. 路径设置
# 之前生成的 JSON 文件路径
JSON_PATH = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/Single-vascular-reconstruction/C15-1-36_Locations_Map.json"

# Signal 原始图像文件夹
INPUT_SIGNAL_DIR = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/Single-vascular-reconstruction/C15-1-36-signal"

# 输出裁剪结果的文件夹
OUTPUT_DIR = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/Single-vascular-reconstruction/C15-1-36-signal-Cropped-Patches"

crop_signal_by_json_coordinates(select, JSON_PATH, INPUT_SIGNAL_DIR, OUTPUT_DIR)


#%%
signal_new = '/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/Single-vascular-reconstruction/C15-1-36-signal-2'
mask_files = [f for f in os.listdir(OUTPUT_mask_DIR) if f.endswith(('.tif', '.tiff'))]
    
# ================= 修正后的代码 =================
print(f"找到 {len(mask_files)} 个掩码文件，开始提取信号...")

for mask_file in tqdm(mask_files):
    raw_filename = mask_file.replace('_pred_mask_processed.tif', '.tif').replace('_processed.tif', '.tif')
    
    # 先把所有可能的后缀都剥离，拿到纯 ID (如 HJZ_104-00004...)
    base_name = mask_file.replace('_pred_mask_processed.tif', '')\
                         .replace('_processed.tif', '')\
                         .replace('_uint16_enhanced.tif', '')\
                         .replace('_mask.tif', '')\
                         .replace('.tif', '')
    base_name_signal = base_name.replace('extracted_', '').replace('mask', 'signal')
    # 然后按照原图目录的规则重新组装
    raw_filename = f"{base_name_signal}.tif"
    
    raw_path = os.path.join(INPUT_SIGNAL_DIR, raw_filename)
    mask_path = os.path.join(OUTPUT_mask_DIR, mask_file)
    out_path = os.path.join(signal_new, raw_filename.replace('.tif', '_signal.tif'))

    if os.path.exists(raw_path):
        extract_vessel_signal(raw_path, mask_path, out_path)
    else:
        # print(f"DEBUG: 正在寻找 -> {raw_path}")
        print(f"跳过: 找不到对应的原图 {raw_filename}")
