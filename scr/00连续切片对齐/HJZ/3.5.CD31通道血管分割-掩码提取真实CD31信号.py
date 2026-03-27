import os
import numpy as np
import tifffile
from tqdm import tqdm

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


#%% ================= 路径配置 =================
roi = 'ROI-00004-02772-17487'


# 1. 原始 CD31 图片文件夹
raw_dir = f"/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/{roi}/Micro-registration/micro_registered_slides_CD31"
    
# 2. 处理后的掩码文件夹 (Post-processed Masks)
mask_dir = f"/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/prediction_postprocessed/{roi}"
    
# 3. 输出文件夹 (Extraction)
output_dir = f"/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/{roi}"
os.makedirs(output_dir, exist_ok = True)

# ===========================================

# 获取所有掩码文件
mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.tif', '.tiff'))]
    
# ================= 修正后的代码 =================
print(f"找到 {len(mask_files)} 个掩码文件，开始提取信号...")

for mask_file in tqdm(mask_files):
    # 【修改点】：只去除掩码特有的后缀，保留原图可能有的 _uint16_enhanced
    # 假设掩码名是: HJZ_xxx_uint16_enhanced_pred_mask_processed.tif
    # 第一步 replace 后变成: HJZ_xxx_uint16_enhanced.tif (这正是我们要的！)
    
    # 这一行根据你的实际掩码后缀来定，如果你之前的掩码后缀是 _processed.tif，就改用那个
    # 这里为了保险，我用 split 提取核心 ID，然后重新组装
    
    # 方法 A：如果你确定掩码文件名里包含 _uint16_enhanced
    raw_filename = mask_file.replace('_pred_mask_processed.tif', '.tif').replace('_processed.tif', '.tif')
    
    # 方法 B (更稳健)：不管掩码叫什么，强制加上 _uint16_enhanced
    # 先把所有可能的后缀都剥离，拿到纯 ID (如 HJZ_104-00004...)
    base_name = mask_file.replace('_pred_mask_processed.tif', '')\
                         .replace('_processed.tif', '')\
                         .replace('_uint16_enhanced.tif', '')\
                         .replace('_mask.tif', '')\
                         .replace('.tif', '')
    
    # 然后按照原图目录的规则重新组装
    raw_filename = f"{base_name}_uint16_enhanced.tif"
    
    raw_path = os.path.join(raw_dir, raw_filename)
    mask_path = os.path.join(mask_dir, mask_file)
    out_path = os.path.join(output_dir, raw_filename.replace('.tif', '_signal.tif'))

    if os.path.exists(raw_path):
        extract_vessel_signal(raw_path, mask_path, out_path)
    else:
        # 调试信息：打印一下到底在找什么文件，方便排查
        # print(f"DEBUG: 正在寻找 -> {raw_path}")
        print(f"跳过: 找不到对应的原图 {raw_filename}")

