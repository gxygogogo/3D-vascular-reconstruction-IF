import os
import numpy as np
import tifffile
import pandas as pd
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from skimage.segmentation import watershed
from scipy import ndimage
from skan import Skeleton, summarize
import matplotlib.pyplot as plt
import seaborn as sns

# --- 绘图函数 (保持不变) ---
def plot_vessel_overlay(raw_img, mask_bin, skeleton_img, save_prefix):
    # ... (为了节省篇幅，这里复用你之前的绘图代码) ...
    # 只需要确保上面的 plot_vessel_overlay 函数在你的脚本里定义过即可
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_img, cmap='gray', interpolation='none')
    overlay_red = np.zeros((*skeleton_img.shape, 4)) 
    overlay_red[skeleton_img > 0] = [1, 0, 0, 1] 
    plt.imshow(overlay_red, interpolation='none')
    plt.axis('off')
    plt.savefig(f"{save_prefix}_Raw_Skeleton.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_bin, cmap='gray', interpolation='none')
    overlay_blue = np.zeros((*skeleton_img.shape, 4))
    overlay_blue[skeleton_img > 0] = [0, 1, 1, 1] 
    plt.imshow(overlay_blue, interpolation='none')
    plt.axis('off')
    plt.savefig(f"{save_prefix}_Mask_Skeleton.png", dpi=150, bbox_inches='tight')
    plt.close()

# --- 分形维数计算 (保持不变) ---
def fractal_dimension(Z, threshold=0.9):
    assert(len(Z.shape) == 2)
    Z = (Z < threshold)
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(((Z[:Z.shape[0]//size*size:size, :Z.shape[1]//size*size:size] > 0).sum()))
    if len(counts) < 2: return 0
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

# --- 🔥 核心修改后的分析函数 ---
def analyze_vessel_comprehensive(mask_path, raw_img_path, save_csv_path, pixel_size=1.0, min_len_px=5):
    print(f"正在处理: {os.path.basename(mask_path)}")
    
    # 1. 读取与预处理
    if mask_path.endswith(('.tif', '.tiff')):
        mask = tifffile.imread(mask_path)
    else:
        mask = cv2.imread(mask_path, 0)
    
    if raw_img_path.endswith(('.tif', '.tiff')):
        raw_img = tifffile.imread(raw_img_path)
    else:
        raw_img = cv2.imread(raw_img_path, 0)
        
    if mask.ndim == 3: mask = np.squeeze(mask)
    if raw_img.ndim == 3: raw_img = np.squeeze(raw_img)
    
    if mask.shape != raw_img.shape:
        print(f"警告: 尺寸不匹配，裁剪中...")
        h = min(mask.shape[0], raw_img.shape[0])
        w = min(mask.shape[1], raw_img.shape[1])
        mask = mask[:h, :w]
        raw_img = raw_img[:h, :w]

    mask_bin = (mask > 0).astype(np.uint8)
    mask_bin = remove_small_objects(mask_bin.astype(bool), min_size=10).astype(np.uint8)
    
    # 图像总像素数 (用于标准化)
    total_image_pixels = mask_bin.size 
    
    # --- 骨架提取 ---
    skeleton_img = skeletonize(mask_bin)
    
    if np.sum(skeleton_img) == 0:
        return None, None
        
    skel_obj = Skeleton(skeleton_img, spacing=pixel_size)
    branch_data = summarize(skel_obj)
    
    # ==========================================================
    # 核心步骤: 像素级面积分配 (Watershed)
    # 目标: 统计属于每一段骨架的 mask 像素数量
    # ==========================================================
    print("   -> 正在进行像素分配 (Watershed)...")
    
    # 1. 创建标记图 (Markers)：把骨架上的像素标记为 Segment ID
    # skel_obj.n_paths 是分支数量
    # 我们创建一个全是0的图，把骨架像素填上 1, 2, 3... (ID)
    markers = np.zeros_like(mask_bin, dtype=np.int32)
    
    # 建立一个 ID 映射表 (DataFrame Index -> Marker ID)
    # branch_data 的 index 就是 skel_obj 的 path index
    for i in range(len(branch_data)):
        path_inds = skel_obj.path(i)
        coords = np.unravel_index(path_inds, mask_bin.shape)
        # 注意：markers 的 ID 必须 > 0，所以用 i + 1
        markers[coords] = i + 1 
        
    # 2. 距离变换 (作为分水岭的地形/Basin)
    # 血管中心距离背景越远，值越大。取负数作为地形，水往低处(中心)流
    dist = ndimage.distance_transform_edt(mask_bin)
    
    # 3. 分水岭算法
    # 让 markers (骨架) 扩张，填满 mask_bin (血管掩码)
    # 结果是一个 label map，每个像素的值等于它所属的骨架 ID
    vessel_labels = watershed(-dist, markers, mask=mask_bin)
    
    # 4. 统计每个 ID 的像素数
    # np.bincount 统计每个整数出现的次数
    # 最大的 ID 是 len(branch_data)，所以 bin 数量要够
    area_counts = np.bincount(vessel_labels.ravel())
    
    # ==========================================================
    
    # --- 循环提取其他指标 ---
    radius_map = ndimage.distance_transform_edt(mask_bin) * pixel_size
    mean_radii = []
    mean_intensities = []
    std_intensities = []
    pixel_areas = []      # 存储每个血管的像素数
    norm_areas = []       # 存储标准化后的面积
    
    for i in range(len(branch_data)):
        path_inds = skel_obj.path(i)
        coords = np.unravel_index(path_inds, mask_bin.shape)
        
        mean_radii.append(np.mean(radius_map[coords]))
        mean_intensities.append(np.mean(raw_img[coords]))
        std_intensities.append(np.std(raw_img[coords]))
        
        # 获取该段血管的像素面积
        # 对应的 Marker ID 是 i + 1
        # 只有当 Watershed 成功分配时才有值，否则设为 0
        marker_id = i + 1
        if marker_id < len(area_counts):
            count = area_counts[marker_id]
        else:
            count = 0
        
        pixel_areas.append(count)
        # 标准化: 该血管像素数 / 原图总像素数
        norm_areas.append(count / total_image_pixels)

    # --- 全局指标 ---
    vessel_pixels = raw_img[mask_bin > 0]
    fd_value = fractal_dimension(skeleton_img)
    
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = ndimage.convolve(skeleton_img.astype(np.uint8), kernel, mode='constant', cval=0) * skeleton_img
    num_junctions = np.sum(filtered >= 13)
    num_endpoints = np.sum(filtered == 11)
    
    image_area_mm2 = (mask_bin.shape[0] * mask_bin.shape[1] * pixel_size * pixel_size) / 1e6

    # --- 保存 Global Summary ---
    summary_dict = {
        'Sample': os.path.basename(mask_path),
        'Vessel_Density_Area_Fraction': np.sum(mask_bin) / mask_bin.size,
        'Total_Vessel_Area_mm2': np.count_nonzero(mask_bin) * (pixel_size**2) / 1e6,
        'Fractal_Dimension': fd_value,
        'Total_Junctions': num_junctions,
        'Junction_Density_per_mm2': num_junctions / image_area_mm2,
        'Total_Endpoints': num_endpoints,
        'Global_Signal_Mean': np.mean(vessel_pixels),
        'Global_Signal_Std': np.std(vessel_pixels),
        'Global_Signal_CV': np.std(vessel_pixels) / (np.mean(vessel_pixels) + 1e-6)
    }

    # --- 保存 Segment Details ---
    df_segments = pd.DataFrame({
        'Sample': os.path.basename(mask_path),
        'Segment_ID': branch_data.index,
        'Length_um': branch_data['branch-distance'],
        'Mean_Radius_um': mean_radii,
        'Mean_Intensity': mean_intensities,
        'Intensity_Std': std_intensities,
        'Tortuosity': branch_data['branch-distance'] / branch_data['branch-type'],
        'Segment_Pixel_Count': pixel_areas,           # 绝对像素数
        'Segment_Area_Normalized': norm_areas         # 标准化面积 (Area/ImageSize)
    })
    
    # 修复无穷大和过滤
    df_segments.loc[np.isinf(df_segments['Tortuosity']), 'Tortuosity'] = np.nan
    df_segments = df_segments[df_segments['Length_um'] > (min_len_px * pixel_size)]

    global_csv = save_csv_path.replace('.csv', '_Global_Summary.csv')
    pd.DataFrame([summary_dict]).to_csv(global_csv, index=False)
    
    segment_csv = save_csv_path.replace('.csv', '_Segment_Details.csv')
    df_segments.to_csv(segment_csv, index=False)

    print(f"全能分析完成！")
    print(f"   --> 细节统计: {segment_csv}")

    # 绘图
    save_img_prefix = save_csv_path.rsplit('.', 1)[0]
    plot_vessel_overlay(raw_img, mask_bin, skeleton_img, save_img_prefix)
    
    return summary_dict, df_segments

# ================= 运行 =================
sample_list = ['PSKO2_CD31_1', 'ETPSWT5_CD31_1', 'ETKO2_CD31_1', 'ETPSKO3_CD31_1']

for sample in sample_list:
    mask_file = f"/public3/Xinyu/3D_tissue/IF/Vascular_stat/prediction_postprocessed/{sample}_pred_mask_processed.tif"
    raw_file = f"/public3/Xinyu/3D_tissue/IF/Vascular_stat/CD31_tif/{sample}.tif"
    analyze_vessel_comprehensive(mask_file, 
                                 raw_file, 
                                 f"/public3/Xinyu/3D_tissue/IF/Vascular_stat/{sample}_result.csv", 
                                 pixel_size=1.0)

#%% 简单的统计分析
save_plot_dir = "/public3/Xinyu/3D_tissue/IF/Vascular_stat/plots"
os.makedirs(save_plot_dir, exist_ok=True)

# 设置 seaborn 主题风格，让图表更适合出版
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
# 定义颜色板 (可以根据需要修改)
palette = sns.color_palette("Set2", n_colors=4)

# 定义通用的绘图函数，避免重复代码
def save_current_plot(filename):
    filepath = os.path.join(save_plot_dir, filename)
    # 保存为矢量图(pdf)和高清位图(png)
    plt.savefig(filepath + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(filepath + ".pdf", bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {filename}")


ETPSWT5_detail_data = pd.read_csv('/public3/Xinyu/3D_tissue/IF/Vascular_stat/ETPSWT5_CD31_1_result_Segment_Details.csv')
PSKO2_detail_data = pd.read_csv('/public3/Xinyu/3D_tissue/IF/Vascular_stat/PSKO2_CD31_1_result_Segment_Details.csv')
ETKO2_detail_data = pd.read_csv('/public3/Xinyu/3D_tissue/IF/Vascular_stat/ETKO2_CD31_1_result_Segment_Details.csv')
ETPSKO3_detail_data = pd.read_csv('/public3/Xinyu/3D_tissue/IF/Vascular_stat/ETPSKO3_CD31_1_result_Segment_Details.csv')

ETPSWT5_global_data = pd.read_csv('/public3/Xinyu/3D_tissue/IF/Vascular_stat/ETPSWT5_CD31_1_result_Global_Summary.csv')
PSKO2_global_data = pd.read_csv('/public3/Xinyu/3D_tissue/IF/Vascular_stat/PSKO2_CD31_1_result_Global_Summary.csv')
ETKO2_global_data = pd.read_csv('/public3/Xinyu/3D_tissue/IF/Vascular_stat/ETKO2_CD31_1_result_Global_Summary.csv')
ETPSKO3_global_data = pd.read_csv('/public3/Xinyu/3D_tissue/IF/Vascular_stat/ETPSKO3_CD31_1_result_Global_Summary.csv')

## 长度分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(
    data=ETPSKO3_detail_data,
    x="Length_um",
    #hue="Group",      # 按组上色
    palette=palette,
    kde=True,         # 显示核密度曲线
    element="step",   # 直方图样式
    stat="density",   # Y轴显示密度，方便不同数量级的样本比较
    common_norm=False # 每组单独归一化
)
plt.title("Distribution of Vessel Segment Lengths")
plt.xlabel("Length (μm)")
plt.ylabel("Density")
# 血管长度通常呈长尾分布，如果太偏，可以考虑用对数坐标轴
# plt.xscale('log') 
save_current_plot("ETPSKO3_Length_Histogram")

## 不同样本长度boxplot
ETPSWT5_global_data['Group'] = 'ETPSWT5 (WT)'
PSKO2_global_data['Group']   = 'PSKO2'
ETKO2_global_data['Group']   = 'ETKO2'
ETPSKO3_global_data['Group'] = 'ETPSKO3'

# 同样的，给 detail 表也加上 (如果后面也要画 detail 的图)
ETPSWT5_detail_data['Group'] = 'ETPSWT5 (WT)'
PSKO2_detail_data['Group']   = 'PSKO2'
ETKO2_detail_data['Group']   = 'ETKO2'
ETPSKO3_detail_data['Group'] = 'ETPSKO3'

all_global_df = pd.concat([
    ETPSWT5_detail_data,
    PSKO2_detail_data,
    ETKO2_detail_data,
    ETPSKO3_detail_data
], ignore_index=True)

plt.figure(figsize=(8, 6))
# 画箱线图
ax = sns.boxplot(
    data=all_global_df,
    x="Group",
    y="Length_um",
    palette=palette,
    showfliers=False # 不显示异常值点，因为后面要叠加散点
)
# 叠加散点图 (Stripplot)，展示真实数据分布和样本量
# sns.stripplot(
#     data=all_global_df,
#     x="Group",
#     y="Length_um",
#     color="black",
#     alpha=0.3, # 透明度
#     size=2,    # 点的大小
#     jitter=True # 抖动显示，防止点重叠
# )
plt.title("Vessel Length Comparison")
plt.ylabel("Length (μm)")
plt.xlabel("") # 移除X轴标签，因为图例已经说明了
plt.axis(ymin=0, ymax=200) # Y轴从0开始
sns.despine(offset=10, trim=True) # 去掉顶部和右侧边框，更美观
save_current_plot("Comp_Length_Boxplot")

## 扭曲度分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(
    data=ETPSKO3_detail_data, 
    x="Tortuosity", 
    #hue="Group",
    palette=palette, kde=True, element="step", stat="density", common_norm=False,
    binrange=(1, 3) # 限制显示范围，因为有些极端的扭曲度会拉伸坐标轴
)
plt.axvline(x=1.0, color='gray', linestyle='--') # 标记 1.0 (笔直)
plt.title("Distribution of Vessel Tortuosity")
plt.xlabel("Tortuosity (Path/Euclidean)")
save_current_plot("ETPSKO3_Tortuosity_Histogram")

## 不同样本扭曲图boxplot
plt.figure(figsize=(8, 6))
# 画箱线图
ax = sns.boxplot(
    data=all_global_df,
    x="Group",
    y="Tortuosity",
    palette=palette,
    showfliers=False # 不显示异常值点，因为后面要叠加散点
)
# 叠加散点图 (Stripplot)，展示真实数据分布和样本量
sns.stripplot(
    data=all_global_df,
    x="Group",
    y="Tortuosity",
    color="black",
    alpha=0.3, # 透明度
    size=2,    # 点的大小
    jitter=True # 抖动显示，防止点重叠
)
plt.title("Vessel Length Comparison")
plt.ylabel("Length (μm)")
plt.xlabel("") # 移除X轴标签，因为图例已经说明了
sns.despine(offset=10, trim=True) # 去掉顶部和右侧边框，更美观
save_current_plot("Comp_Tortuosity_Boxplot")


## 平均强度分布直方图

## 强度标准差分布直方图

