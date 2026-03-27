import cv2
import numpy as np
import tifffile
import os

def enhance_image_intensity(image_data):
    # 1. 维度压缩
    if image_data.ndim == 3:
        image_data = np.squeeze(image_data)
        
    # 2. 鲁棒归一化
    vmin, vmax = np.percentile(image_data, (1, 99.99)) # 稍微调高上限，避免过曝
    normalized = np.clip((image_data - vmin) / (vmax - vmin), 0, 1)
    img_uint8 = (normalized * 255).astype(np.uint8)
    
    # 3. CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img_uint8)
    
    return enhanced_img

def save_overlay_native_resolution(raw_path, mask_path, save_path, color='cyan', alpha=0.3):
    """
    保留原始分辨率的叠加保存函数
    直接操作像素矩阵，不使用 matplotlib
    """
    print(f"处理中: {os.path.basename(raw_path)}")
    
    # --- 1. 读取数据 ---
    if raw_path.endswith(('.tif', '.tiff')):
        raw = tifffile.imread(raw_path)
    else:
        raw = cv2.imread(raw_path, -1)
        
    if mask_path.endswith(('.tif', '.tiff')):
        mask = tifffile.imread(mask_path)
    else:
        mask = cv2.imread(mask_path, 0)
    
    mask_bin = (mask > 0).astype(bool) # 转为布尔矩阵
    
    # 尺寸对齐 (以防万一)
    h, w = min(raw.shape[0], mask.shape[0]), min(raw.shape[1], mask.shape[1])
    raw = raw[:h, :w]
    mask_bin = mask_bin[:h, :w]

    # --- 2. 图像增强 (得到单通道灰度图) ---
    enhanced_gray = enhance_image_intensity(raw)

    # --- 3. 转换为 BGR 彩色空间 (OpenCV 默认使用 BGR 顺序) ---
    # 必须转为3通道才能叠加上色
    base_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    # --- 4. 定义颜色 (注意：OpenCV 是 B-G-R 顺序) ---
    # 格式: (Blue, Green, Red)
    colors_bgr = {
        'red':     (0, 0, 255),
        'green':   (0, 255, 0),
        'blue':    (255, 0, 0),
        'cyan':    (255, 255, 0),  # B=255, G=255, R=0
        'yellow':  (0, 255, 255),  # B=0, G=255, R=255
        'magenta': (255, 0, 255)
    }
    overlay_color = colors_bgr.get(color, (0, 0, 255)) # 默认红

    # --- 5. 像素级混合 (Alpha Blending) ---
    # 逻辑：只在 mask 为 True 的地方修改像素
    # New_Pixel = Old_Pixel * (1 - alpha) + Color * alpha
    
    # 创建结果副本
    final_img = base_bgr.copy()
    
    # 获取掩码区域的像素 (N, 3)
    roi = final_img[mask_bin]
    
    # 计算混合结果 (需要转为 float 防止溢出，算完转回 uint8)
    # 这一步利用广播机制，只计算 masked 区域，速度极快
    blended_roi = (roi.astype(float) * (1 - alpha) + np.array(overlay_color) * alpha)
    
    # 填回结果图像
    final_img[mask_bin] = blended_roi.astype(np.uint8)

    # --- 6. 保存 ---
    # cv2.imwrite 能够保证像素 1:1 输出，不改变分辨率
    # 自动创建目录
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    cv2.imwrite(save_path, final_img)
    print(f"图片已保存 (原始分辨率 {w}x{h}): {save_path}")

#%% 运行
sample = 'ETKO2_CD31_1'
raw_file = f"/public3/Xinyu/3D_tissue/IF/Vascular_stat/CD31_tif/{sample}.tif"
mask_file = f"/public3/Xinyu/3D_tissue/IF/Vascular_stat/prediction_postprocessed/{sample}_pred_mask_processed.tif"
save_file = f"/public3/Xinyu/3D_tissue/IF/Vascular_stat/{sample}_overlay_native.png"

# 使用新函数
save_overlay_native_resolution(raw_file, mask_file, save_file, color='cyan', alpha=0.3)
