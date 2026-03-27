import cv2
import numpy as np
import tifffile
import os
from scipy import ndimage

# ================= 参数控制区 (修改这里!) =================
CONFIG = {
    # 1. 连接断裂血管的力度 (越断裂，这两个数越大)
    'close_kernel_size': (7, 7),  # 闭运算核大小，建议 (5,5) 到 (9,9)
    'close_iter': 6,              # 闭运算迭代次数，建议 3 到 8

    # 2. 去除噪点的力度 (不想丢失信号，就把这两个数调小)
    'open_kernel_size': (3, 3),   # 开运算核大小，越小越能保留细血管
    'open_iter': 1,               # 开运算次数，设为 0 则不进行去噪，保留所有信号

    # 3. 最终过滤的门槛 (保留多小的碎片)
    'min_area': 30                # 连通域最小像素数，越小保留的碎片越多
}
# ========================================================

def create_filled_vessel_mask(image_path, output_path, config):
    try:
        # 读取
        img = tifffile.imread(image_path)
        
        # 归一化
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 增强对比度 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)

        # 阈值分割 (Otsu)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, wall_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- 关键步骤1：连接断裂 (闭运算) ---
        if config['close_iter'] > 0:
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config['close_kernel_size'])
            # 使用 iterate 使得连接能力成倍增加
            connected_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel_close, iterations=config['close_iter'])
        else:
            connected_mask = wall_mask

        # --- 关键步骤2：填充孔洞 (Fill Holes) ---
        filled_mask_bool = ndimage.binary_fill_holes(connected_mask > 0)
        filled_mask = (filled_mask_bool * 255).astype(np.uint8)

        # --- 关键步骤3：去噪 (开运算) ---
        # 如果你的细血管被删掉了，请把 config['open_iter'] 设为 0
        if config['open_iter'] > 0:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config['open_kernel_size'])
            clean_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_open, iterations=config['open_iter'])
        else:
            clean_mask = filled_mask

        # --- 关键步骤4：面积过滤 ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
        final_mask = np.zeros_like(clean_mask)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= config['min_area']:
                final_mask[labels == i] = 255

        # 保存
        cv2.imwrite(output_path, final_mask)
        print(f"处理完成: {os.path.basename(output_path)}")
        
    except Exception as e:
        print(f"ERROR {image_path}: {e}")

# 执行代码
input_dir = '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/ROI-1304-2892_CD31'
output_dir = '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/ROI-1304-2892-CD31-mask'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = [f for f in os.listdir(input_dir) if f.endswith(('.tif', '.tiff'))]

print(f"使用参数: {CONFIG}")
for filename in files:
    path_in = os.path.join(input_dir, filename)
    path_out = os.path.join(output_dir, os.path.splitext(filename)[0] + '_mask.png')
    create_filled_vessel_mask(path_in, path_out, CONFIG)





import cv2
import numpy as np
import tifffile
from scipy import ndimage

# 1. 读取图片
path = "/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-02772-17487/Micro-registration/micro_registered_slides_CD31_denoise/enhanced_image.tif"
img = tifffile.imread(path)

# 2. 预处理：归一化到 8-bit
if img.dtype != np.uint8:
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# 如果是多通道（防止意外），转灰度
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 图像增强 (CLAHE) - 突出血管结构
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(img)

# 4. 分割 (Otsu)
blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
_, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 5. 形态学连接 (闭运算)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

# 6. 填充孔洞 (Fill Holes) - 让血管变成实心
filled_bool = ndimage.binary_fill_holes(closed)
filled = (filled_bool * 255).astype(np.uint8)

# 7. 过滤噪点 (保留面积 > 100 的区域)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
final_mask = np.zeros_like(filled)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= 100:
        final_mask[labels == i] = 255

# 8. 生成可视化图 (原图 + 绿色轮廓)
overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 在原图上画绿色轮廓 (0, 255, 0)，线宽 2
cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

# 保存
cv2.imwrite("/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-02772-17487/Micro-registration/micro_registered_slides_CD31_denoise/vessel_mask.png", final_mask)
cv2.imwrite("/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-02772-17487/Micro-registration/micro_registered_slides_CD31_denoise/vessel_identified_overlay.png", overlay)


