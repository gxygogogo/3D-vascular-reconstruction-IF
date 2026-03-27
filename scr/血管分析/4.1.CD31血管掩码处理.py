import os
import cv2
import numpy as np
import tifffile
from scipy import ndimage
from tqdm import tqdm

def post_process_mask(mask_path, save_path, min_area=100, close_ksize=7):
    '''
    post_process_mask 的 Docstring
    
    :param mask_path: 掩码路径
    :param save_path: 保存处理好后的掩码
    :param min_area: 过滤阈值
    :param close_ksize: 连接阈值
    '''
    # 1. 读取预测的掩码图
    # 预测图通常是 0和255 (或0和1)
    mask = tifffile.imread(mask_path)
    
    # 维度处理：如果是 (1, H, W) -> (H, W)
    if mask.ndim == 3: mask = np.squeeze(mask)
    
    # 确保是二值图 (0 和 255)
    # 如果原图是 0-1，这里会变成 0-255
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # --- 步骤 1: 闭运算 (连接断裂) ---
    # 原理：先膨胀后腐蚀。能弥合缝隙，连接临近的断裂处。
    # kernel_size 越大，能连接的裂缝越宽，但也越容易把两条不相干的血管粘在一起
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    # 建议多做几次迭代 (iterations)，效果通常比单纯增大 kernel 更好
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- 步骤 2: 填充孔洞 (补齐内部空缺) ---
    # 原理：找到所有被白色包围的黑色区域，把它填白
    # 必须先转成 bool 类型 (True/False)
    filled_mask_bool = ndimage.binary_fill_holes(closed_mask > 0)
    filled_mask = (filled_mask_bool * 255).astype(np.uint8)

    # --- 步骤 3: 面积过滤 (去除小噪点) ---
    # 原理：计算连通域面积，小于阈值的扔掉
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)
    
    final_mask = np.zeros_like(filled_mask)
    
    # stats[:, 4] 是面积列 (cv2.CC_STAT_AREA)
    # 0号 label 是背景，所以从 1 开始遍历
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            final_mask[labels == i] = 255

    # 4. 保存
    # 使用压缩保存，减小体积
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tifffile.imwrite(save_path, final_mask, compression='zlib')
    # print(f"处理完成: {os.path.basename(save_path)}")



if __name__ == "__main__":
    # ================= 参数配置 =================
    # 输入：预测出来的原始 Mask 路径
    input_dir = "/public3/Xinyu/3D_tissue/IF/Vascular_stat/prediction"
    
    # 输出：处理后的 Mask 路径
    output_dir = "/public3/Xinyu/3D_tissue/IF/Vascular_stat/prediction_postprocessed"
    
    # 参数微调
    MIN_AREA = 800       # 过滤阈值：小于 500 像素的碎块会被删掉 (根据你的图分辨率调整)
    CLOSE_KERNEL = 7     # 连接力度：数值越大(必须是奇数)，连接能力越强，但细节丢失越多
    # ===========================================

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有 .tif 文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    
    print(f"开始处理 {len(files)} 张图片...")
    
    for f in tqdm(files):
        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f.replace('.tif', '_processed.tif'))
        
        try:
            post_process_mask(in_path, out_path, min_area=MIN_AREA, close_ksize=CLOSE_KERNEL)
        except Exception as e:
            print(f"处理 {f} 失败: {e}")

    print("全部完成！")



