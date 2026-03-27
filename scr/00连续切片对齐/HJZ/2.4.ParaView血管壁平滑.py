import numpy as np
import tifffile as tif
from scipy.ndimage import zoom, gaussian_filter

# 1. 读取堆叠的 TIF 文件 (假设顺序是 Z, Y, X)
# 你的原始数据
img_stack = tif.imread('/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/ROI-1304-2892.tif') 

# 2. 计算缩放因子 (关键步骤)
# 假设你的显微镜参数如下：
z_spacing = 5.0  # Z轴层间距 (比如 5微米)
xy_pixel = 0.5   # XY平面像素大小 (比如 0.5微米)

# 为了让 Z 轴和 XY 轴分辨率一致，Z 轴需要拉伸的倍数
z_factor = z_spacing / xy_pixel 
# XY 轴保持不变 (倍数为 1)
zoom_factors = [z_factor, 1, 1]

print(f"正在进行插值重采样，Z轴拉伸倍数: {z_factor}...")

# 3. 使用三次样条插值 (order=3) 重采样
# 这会填补层与层之间的空隙，消除台阶效应
resampled_img = zoom(img_stack, zoom_factors, order=3)

# 4. (可选) 3D 高斯平滑
# 在提取表面前稍微模糊一下，可以去除噪点，让血管壁更圆润
# sigma 控制模糊程度，通常 1.0 - 2.0 就够了
smoothed_img = gaussian_filter(resampled_img, sigma=1.0)

# 5. 保存处理后的图像
tif.imwrite('/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/smoothed_isotropic_stack.tif', smoothed_img.astype(np.uint16))

