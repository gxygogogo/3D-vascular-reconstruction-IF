import os
import re
import numpy as np
import tifffile
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects 
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects

#%% 指标统计
def analyze_cross_sections_full(mask_path, raw_img_path, save_csv_path, pixel_size_um=1.0, min_area_px=5):
    """
    分析独立的血管二维切面 (Cross-sections)。
    包含：面积标准化、长宽测量 (拟合椭圆)、信号强度统计。
    
    参数:
        mask_path: 二值化掩码路径
        raw_img_path: 原始荧光信号图路径
        save_csv_path: 结果保存路径 (.csv)
        pixel_size_um: 像素物理尺寸 (例如 0.65 um/px)
        min_area_px: 过滤掉小于多少像素的噪点
    """
    sample_name = os.path.basename(mask_path).split('.')[0]
    print(f"正在处理样本: {sample_name}")
    
    # --- 1. 读取数据 ---
    if mask_path.endswith(('.tif', '.tiff')):
        mask = tifffile.imread(mask_path)
    else:
        mask = cv2.imread(mask_path, 0)
    
    if raw_img_path.endswith(('.tif', '.tiff')):
        raw_img = tifffile.imread(raw_img_path)
    else:
        raw_img = cv2.imread(raw_img_path, 0)
        
    # 降维处理 (防止 3D 数据混入)
    if mask.ndim == 3: mask = np.squeeze(mask)
    if raw_img.ndim == 3: raw_img = np.squeeze(raw_img)
    
    # 尺寸对齐 (裁剪到最小尺寸)
    h, w = min(mask.shape[0], raw_img.shape[0]), min(mask.shape[1], raw_img.shape[1])
    mask = mask[:h, :w]
    raw_img = raw_img[:h, :w]

    # 计算整张图片的总像素数 (用于面积标准化)
    total_image_pixels = h * w
    print(f"   -> 图像尺寸: {w} x {h} (总像素: {total_image_pixels})")

    # 确保掩码是二值 (bool)
    mask_bin = (mask > 0)
    
    # --- 2. 预处理：清理微小噪点 ---
    mask_clean = remove_small_objects(mask_bin, min_size=min_area_px)
    
    # --- 3. 连通域标记 (Labeling) ---
    # connectivity=2 表示 8 邻域连接 (对角线也算连通)
    label_image = label(mask_clean, connectivity=2)
    num_objects = label_image.max()
    print(f"   -> 检测到 {num_objects} 个独立的血管切面")

    if num_objects == 0:
        print("未检测到有效对象，跳过。")
        return None

    # --- 4. 提取指标 (Regionprops) ---
    print("   -> 正在计算几何与信号指标...")
    properties = [
        'label',           # ID
        'area',            # 像素面积
        'mean_intensity',  # 平均强度
        'centroid',        # 中心坐标
        'equivalent_diameter', # 等效直径
        'major_axis_length',   # 长轴 (Length)
        'minor_axis_length',   # 短轴 (Width)
        'eccentricity'         # 离心率
    ]
    
    # 计算指标字典
    props_dict = regionprops_table(label_image, intensity_image=raw_img, properties=properties)
    df = pd.DataFrame(props_dict)
    
    # --- 5. 数据转换与整理 ---
    
    # 重命名基础列
    df = df.rename(columns={
        'label': 'Object_ID',
        'area': 'Area_pixels',
        'mean_intensity': 'Mean_Intensity',
        'equivalent_diameter': 'Equivalent_Diameter_px',
        'major_axis_length': 'Length_Major_Axis_px',
        'minor_axis_length': 'Width_Minor_Axis_px',
        'centroid-0': 'Center_Y',
        'centroid-1': 'Center_X'
    })
    
    # 插入样本名
    df.insert(0, 'Sample', sample_name)
    
    # --- 关键计算：标准化与物理单位 ---
    
    # 1. 物理面积 (um^2)
    df['Area_um2'] = df['Area_pixels'] * (pixel_size_um ** 2)
    
    # 2. 标准化面积 (Normalized Area Ratio)
    # 公式：单个血管像素 / 全图总像素
    # 结果是一个 0~1 之间的小数 (例如 0.0005)
    df['Area_Normalized_Ratio'] = df['Area_pixels'] / total_image_pixels
    
    # 3. 物理长度与宽度 (um)
    df['Length_um'] = df['Length_Major_Axis_px'] * pixel_size_um
    df['Width_um']  = df['Width_Minor_Axis_px'] * pixel_size_um
    df['Equivalent_Diameter_um'] = df['Equivalent_Diameter_px'] * pixel_size_um

    # --- 6. 整理列顺序 (把重要的放前面) ---
    cols_order = [
        'Sample', 'Object_ID', 
        'Area_um2', 'Area_Normalized_Ratio',  # 面积相关
        'Length_um', 'Width_um', 'Equivalent_Diameter_um', # 尺寸相关
        'Mean_Intensity', # 信号相关
        'Center_X', 'Center_Y' # 位置相关
    ]
    # 把剩下的列补在后面
    existing_cols = [c for c in cols_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + remaining]

    # --- 7. 保存与可视化 ---
    df.to_csv(save_csv_path, index=False)
    print(f"=================== 结果已保存: {save_csv_path} ===================")
    print(df[['Object_ID', 'Area_um2', 'Length_um', 'Width_um', 'Area_Normalized_Ratio']].head())

    # ==============================================================================
    # 可视化部分 (修改点：添加 Label 标注)
    # ==============================================================================
    print("   -> 正在生成带有标签的可视化图...")
    # 1. 创建画布，稍微大一点以便看清文字
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 2. 绘制底图 (背景透明，对象彩色)
    masked_label = np.ma.masked_where(label_image == 0, label_image)
    # 使用 'nipy_spectral' 或 'jet' 等鲜艳的色谱
    ax.imshow(masked_label, cmap='nipy_spectral', interpolation='nearest')
    
    # 3. 循环遍历 DataFrame，在每个对象的中心点绘制 ID
    # 如果对象太多(例如超过2000个)，标注可能会挤在一起看不清，可以考虑加个判断只标大对象
    for _, row in df.iterrows():
        obj_id = int(row['Object_ID'])
        cx = row['Center_X']
        cy = row['Center_Y']
        
        # 在中心点添加文字
        # color='yellow': 黄色文字在深色背景上比较显眼
        # ha='center', va='center': 文字居中对齐
        # fontsize=8: 字号适中，根据实际情况调整
        txt = ax.text(cx, 
                      cy, 
                      str(obj_id), 
                      color='black', 
                      fontsize=1, 
                      fontweight='bold',
                      ha='center', 
                      va='center')
        
        # (可选) 给文字加一个黑色描边，确保在任何颜色的背景上都能看清
        # txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    ax.set_title(f"Detected Objects with ID Labels\nSample: {sample_name} (Total: {num_objects})")
    ax.axis('off')
    
    # 保存图片
    save_img_path = save_csv_path.replace('.csv', '_LabelMap_Annotated.png')
    # 提高 dpi 让文字更清晰
    plt.savefig(save_img_path, dpi = 800, bbox_inches='tight')
    plt.close()
    print(f"可视化图已保存: {save_img_path}")
    
    return df

# ================= 运行部分 =================

# 1. 设置文件路径
# sample_list = ['ETPSWT5_CD31_1', 'PSKO2_CD31_1', 'ETKO2_CD31_1', 'ETPSKO3_CD31_1']
ROI = 'ROI-00004-07662-12501'
mask_path = f'/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/prediction_postprocessed/{ROI}'
raw_path = f'/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/{ROI}'
output_path = f'/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/cross_section_analysis/{ROI}'
os.makedirs(output_path, exist_ok = True)
sample_list = os.listdir(mask_path)

for sample in sample_list:

    base = sample.replace('_pred_mask_processed.tif','')

    print(f'======================= Processing Sample: {base} =======================')

    mask_file = f"{mask_path}/{base}_pred_mask_processed.tif"
    raw_file = f"{raw_path}/{base}_uint16_enhanced_signal.tif"
    output_csv = f"{output_path}/{sample}_CrossSection_Stats.csv"
    # 2. 运行分析
    # 注意：请务必修改 pixel_size_um 为你实际的像素尺寸 (例如 0.65)
    df_result = analyze_cross_sections_full(
        mask_path=mask_file, 
        raw_img_path=raw_file, 
        save_csv_path=output_csv, 
        pixel_size_um=1.0,  # <--- 在这里修改分辨率
        min_area_px=10      # 过滤小于10个像素的杂点
    )

#%% 指标比较
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects 

save_plot_dir = "/public3/Xinyu/3D_tissue/IF/Vascular_stat/plots"
os.makedirs(save_plot_dir, exist_ok=True)

# 设置 seaborn 主题风格，让图表更适合出版
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
# 定义颜色板 (可以根据需要修改)
palette = sns.color_palette("Set2", n_colors=4)

# 定义通用的绘图函数，避免重复代码
def save_current_plot(save_plot_dir, filename):
    filepath = os.path.join(save_plot_dir, filename)
    # 保存为矢量图(pdf)和高清位图(png)
    plt.savefig(filepath + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(filepath + ".pdf", bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {filename}")

'''
Object_ID: 血管截面的ID
Area_um2: 血管截面的真实物理面积, 像素数 X 像素尺寸的平方
Area_Normalized_Ratio: 标准化血管截面的面积比例, 血管截面像素数 / 图像总像素数
Length_um: 血管长轴的长度(最长直径)
Width_um: 血管最短直径
Equivalent_Diameter_um: 平均直径
Mean_Intensity: 原始信号平均强度
Center_X: 血管质心的坐标
Center_Y: 血管质心的坐标
Area_pixels: 原始像素点数量
Equivalent_Diameter_px: 
Length_Major_Axis_px: 
Width_Minor_Axis_px: 
eccentricity: 离心率，描述血管扁平程度
'''
# 老鼠
ETPSWT5 = pd.read_csv('/public3/Xinyu/3D_tissue/IF/Vascular_stat/ETPSWT5_CD31_1_CrossSection_Stats.csv')
PSKO2 = pd.read_csv('/public3/Xinyu/3D_tissue/IF/Vascular_stat/PSKO2_CD31_1_CrossSection_Stats.csv')
ETKO2 = pd.read_csv('/public3/Xinyu/3D_tissue/IF/Vascular_stat/ETKO2_CD31_1_CrossSection_Stats.csv')
ETPSKO3 = pd.read_csv('/public3/Xinyu/3D_tissue/IF/Vascular_stat/ETPSKO3_CD31_1_CrossSection_Stats.csv')

ETPSWT5['Group'] = 'ETPSWT5'
PSKO2['Group']   = 'PSKO2'
ETKO2['Group']   = 'ETKO2'
ETPSKO3['Group'] = 'ETPSKO3'

all_global_df = pd.concat([ETPSWT5, PSKO2, ETKO2, ETPSKO3], ignore_index = True)


plt.figure(figsize=(8, 6))
# 画箱线图
ax = sns.boxplot(
    data=all_global_df,
    x="Group",
    y="eccentricity",
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
plt.title("Vessel Eccentricity Comparison")
plt.ylabel("Eccentricity")
plt.xlabel("") # 移除X轴标签，因为图例已经说明了
#plt.axis(ymin=0, ymax=200) # Y轴从0开始
sns.despine(offset=10, trim=True) # 去掉顶部和右侧边框，更美观
save_current_plot(save_plot_dir, "Comp_Eccentricity_Boxplot")

#%% HJZ
save_plot_dir = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/cross_section_analysis/plots"
os.makedirs(save_plot_dir, exist_ok = True)

HJZ_1_ROI_00004_02772_17487 = pd.read_csv('/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/cross_section_analysis/ROI-00004-02772-17487/HJZ_1-00004-02772-17487_pred_mask_processed.tif_CrossSection_Stats.csv')
HJZ_1_ROI_00004_07662_12501 = pd.read_csv('/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/cross_section_analysis/ROI-00004-07662-12501/HJZ_1-00004-07662-12501_pred_mask_processed.tif_CrossSection_Stats.csv')
HJZ_1_ROI_00004_02772_17487['Group'] = 'ROI-00004-02772-17487'
HJZ_1_ROI_00004_07662_12501['Group'] = 'ROI-00004-07662-12501'

HJZ_20_ROI_00004_02772_17487 = pd.read_csv('/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/cross_section_analysis/ROI-00004-02772-17487/HJZ_20-00004-02772-17487_pred_mask_processed.tif_CrossSection_Stats.csv')
HJZ_20_ROI_00004_07662_12501 = pd.read_csv('/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/cross_section_analysis/ROI-00004-07662-12501/HJZ_20-00004-07662-12501_pred_mask_processed.tif_CrossSection_Stats.csv')
HJZ_20_ROI_00004_02772_17487['Group'] = 'ROI-00004-02772-17487'
HJZ_20_ROI_00004_07662_12501['Group'] = 'ROI-00004-07662-12501'

HJZ_41_ROI_00004_02772_17487 = pd.read_csv('/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/cross_section_analysis/ROI-00004-02772-17487/HJZ_41-00004-02772-17487_pred_mask_processed.tif_CrossSection_Stats.csv')
HJZ_41_ROI_00004_07662_12501 = pd.read_csv('/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/cross_section_analysis/ROI-00004-07662-12501/HJZ_41-00004-07662-12501_pred_mask_processed.tif_CrossSection_Stats.csv')
HJZ_41_ROI_00004_02772_17487['Group'] = 'ROI-00004-02772-17487'
HJZ_41_ROI_00004_07662_12501['Group'] = 'ROI-00004-07662-12501'

df_1 = pd.concat([HJZ_1_ROI_00004_02772_17487, HJZ_1_ROI_00004_07662_12501], ignore_index = True)
df_20 = pd.concat([HJZ_20_ROI_00004_02772_17487, HJZ_20_ROI_00004_07662_12501], ignore_index = True)
df_41 = pd.concat([HJZ_41_ROI_00004_02772_17487, HJZ_41_ROI_00004_07662_12501], ignore_index = True)

# 血管面积
plt.figure(figsize=(8, 6))
# 画箱线图
ax = sns.boxplot(
    data=df_20,
    x="Group",
    y="Area_Normalized_Ratio",
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
plt.title("Vessel Area_Normalized_Ratio Comparison")
plt.ylabel("Area_Normalized_Ratio")
plt.xlabel("") # 移除X轴标签，因为图例已经说明了
#plt.axis(ymin=0, ymax=200) # Y轴从0开始
sns.despine(offset=10, trim=True) # 去掉顶部和右侧边框，更美观
save_current_plot(save_plot_dir, "HJZ_20_Comp_Area_Normalized_Ratio_Boxplot")

# 血管长度
plt.figure(figsize=(8, 6))
# 画箱线图
ax = sns.boxplot(
    data=df_20,
    x="Group",
    y="Equivalent_Diameter_um",
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
plt.title("Vessel Equivalent_Diameter_um Comparison")
plt.ylabel("Equivalent_Diameter_um")
plt.xlabel("") # 移除X轴标签，因为图例已经说明了
#plt.axis(ymin=0, ymax=200) # Y轴从0开始
sns.despine(offset=10, trim=True) # 去掉顶部和右侧边框，更美观
save_current_plot(save_plot_dir, "HJZ_20_Comp_Equivalent_Diameter_um_Boxplot")

# 信号强度
plt.figure(figsize=(8, 6))
# 画箱线图
ax = sns.boxplot(
    data=df_41,
    x="Group",
    y="Mean_Intensity",
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
plt.title("Vessel Mean_Intensity Comparison")
plt.ylabel("Mean_Intensity")
plt.xlabel("") # 移除X轴标签，因为图例已经说明了
#plt.axis(ymin=0, ymax=200) # Y轴从0开始
sns.despine(offset=10, trim=True) # 去掉顶部和右侧边框，更美观
save_current_plot(save_plot_dir, "HJZ_41_Comp_Mean_Intensity_Boxplot")

# 离心率
plt.figure(figsize=(8, 6))
# 画箱线图
ax = sns.boxplot(
    data=df_20,
    x="Group",
    y="eccentricity",
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
plt.title("Vessel eccentricity Comparison")
plt.ylabel("eccentricity")
plt.xlabel("") # 移除X轴标签，因为图例已经说明了
#plt.axis(ymin=0, ymax=200) # Y轴从0开始
sns.despine(offset=10, trim=True) # 去掉顶部和右侧边框，更美观
save_current_plot(save_plot_dir, "HJZ_20_Comp_eccentricity_Boxplot")

#%% 连续切片，所有的切片汇总
def aggregate_slice_statistics(df_list):
    """
    将切片指标列表汇总为一个总表。
    每一行代表一张切片，列是该切片中所有血管指标的均值。
    """
    summary_rows = []

    for df in df_list:
        if df is None or df.empty:
            continue
            
        # 1. 获取样本名称 (假设同一张表中 Sample 列都是一样的，取第一个即可)
        sample_name = df['Sample'].iloc[0]
        
        # 2. 选择数值型列进行统计
        # 排除 Object_ID, Center_X, Center_Y 这种求均值没意义的列
        # include='number' 会自动选出所有数字列
        numeric_df = df.select_dtypes(include=['number'])
        
        # 定义不需要求均值的列 (ID和坐标求均值通常无生物学意义)
        cols_to_exclude = ['Object_ID', 'Center_X', 'Center_Y']
        numeric_cols = [c for c in numeric_df.columns if c not in cols_to_exclude]
        
        # 3. 计算均值 (Mean)
        # 这代表：该切片内“平均一根血管”长什么样
        mean_stats = df[numeric_cols].mean()
        
        # --- 进阶：对于某些指标，也许“总和 (Sum)”比“均值”更有意义 ---
        # 例如：Area_Normalized_Ratio 的总和 = 该切片的总血管密度 (Total Density)
        # Area_um2 的总和 = 该切片的总血管面积 (Total Burden)
        # 这里我帮你额外算这两个很有用的 Sum 指标
        total_density = df['Area_Normalized_Ratio'].sum()
        total_area_um2 = df['Area_um2'].sum()
        total_count = len(df) # 血管数量
        
        # 4. 构建这一行的数据
        row_data = {'Sample': sample_name}
        
        # 把算好的均值加进去 (自动加上前缀 Mean_ 以示区分)
        for col, val in mean_stats.items():
            row_data[f"Avg_{col}"] = val
            
        # 把额外的总和指标加进去
        row_data['Total_Vessel_Density'] = total_density
        row_data['Total_Vessel_Area_um2'] = total_area_um2
        row_data['Vessel_Count'] = total_count
        
        summary_rows.append(row_data)

    # 5. 转为 DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    return summary_df


ROI = 'ROI-00004-02772-17487'

file_list = os.listdir(f'/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/cross_section_analysis/{ROI}')
sample_stat_list = [f for f in file_list if f.endswith('_CrossSection_Stats.csv')]

roi_00004_07662_12501_list = [pd.read_csv(f'/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/cross_section_analysis/{ROI}/{stat}') for stat in sample_stat_list]
roi_00004_02772_17487_list = [pd.read_csv(f'/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/cross_section_analysis/{ROI}/{stat}') for stat in sample_stat_list]

# 汇总
roi_00004_02772_17487_summary_df = aggregate_slice_statistics(roi_00004_02772_17487_list)
roi_00004_07662_12501_summary_df = aggregate_slice_statistics(roi_00004_07662_12501_list)
roi_00004_02772_17487_summary_df['Group'] = 'ROI-00004-02772-17487'
roi_00004_07662_12501_summary_df['Group'] = 'ROI-00004-07662-12501'
summary_df = pd.concat([roi_00004_02772_17487_summary_df, roi_00004_07662_12501_summary_df], ignore_index = True)

# 合并
roi_00004_02772_17487_combine_df = pd.concat(roi_00004_02772_17487_list, ignore_index = True)
roi_00004_07662_12501_combine_df = pd.concat(roi_00004_07662_12501_list, ignore_index = True)
roi_00004_02772_17487_combine_df['Group'] = 'ROI-00004-02772-17487'
roi_00004_07662_12501_combine_df['Group'] = 'ROI-00004-07662-12501'
combine_df = pd.concat([roi_00004_02772_17487_combine_df, roi_00004_07662_12501_combine_df], ignore_index = True)



plt.figure(figsize=(8, 6))
# 画箱线图
ax = sns.boxplot(
    data=summary_df,
    x="Group",
    y="Avg_eccentricity",
    palette=palette,
    showfliers=False # 不显示异常值点，因为后面要叠加散点
)
plt.title("Vessel Avg_eccentricity Comparison")
plt.ylabel("Avg_eccentricity")
plt.xlabel("") # 移除X轴标签，因为图例已经说明了
#plt.axis(ymin=0, ymax=200) # Y轴从0开始
sns.despine(offset=10, trim=True) # 去掉顶部和右侧边框，更美观
save_current_plot(save_plot_dir, "summary_Comp_Avg_eccentricity_Boxplot")


## 添加血管密度
summary_df['Vessel_Number_Density_per_mm2'] = 0

IMG_HEIGHT = 4676       # 图片高度 (像素)
IMG_WIDTH  = 6740       # 图片宽度 (像素)
PIXEL_SIZE_UM = 1.0     # 像素尺寸 (微米/像素)

# 1. 计算单张切片的视野总面积 (单位转为 mm^2)
# 公式: (高 * 宽 * 像素尺寸^2) / 1,000,000
total_fov_area_mm2 = (IMG_HEIGHT * IMG_WIDTH * (PIXEL_SIZE_UM ** 2)) / 1e6

# 2. 计算血管数目密度 (Number Density)
# 公式: 血管数量 / 视野面积
summary_df.loc[summary_df.Group == 'ROI-00004-02772-17487', 'Vessel_Number_Density_per_mm2'] = summary_df.loc[summary_df.Group == 'ROI-00004-02772-17487', 'Vessel_Count'] / total_fov_area_mm2

IMG_HEIGHT = 7112       # 图片高度 (像素)
IMG_WIDTH  = 5432       # 图片宽度 (像素)
PIXEL_SIZE_UM = 1.0     # 像素尺寸 (微米/像素)
total_fov_area_mm2 = (IMG_HEIGHT * IMG_WIDTH * (PIXEL_SIZE_UM ** 2)) / 1e6
summary_df.loc[summary_df.Group == 'ROI-00004-07662-12501', 'Vessel_Number_Density_per_mm2'] = summary_df.loc[summary_df.Group == 'ROI-00004-07662-12501', 'Vessel_Count'] / total_fov_area_mm2


def extract_slice_num(filename):
    # 寻找 HJZ_ 后面的数字
    match = re.search(r"HJZ_(\d+)", str(filename))
    if match:
        return int(match.group(1))
    else:
        return 0 # 如果没找到数字，返回0

# 应用函数，创建新列 Slice_Index
summary_df['Slice_Index'] = summary_df['Sample'].apply(extract_slice_num)

# ==========================================
# 2. 排序 (Sorting)
# ==========================================
# 按照 Group 和 Slice_Index 进行升序排列
# 这样画出来的线才是从第1张连到最后一张
plot_df = summary_df.sort_values(by=['Group', 'Slice_Index']).reset_index(drop=True)
print(plot_df[['Group', 'Slice_Index', 'Sample']].head())

# ==========================================
# 3. 绘制分组折线图
# ==========================================
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=plot_df,
    x="Slice_Index",          # 横坐标：切片深度
    y="Vessel_Number_Density_per_mm2", # 纵坐标：血管密度 (请确保列名正确)
    hue="Group",              # 分组颜色：自动区分两个 ROI
    marker="o",               # 显示数据点
    linewidth=2,
    palette="Set1",           # 颜色风格
    alpha=0.8
)

plt.title("Total Vessel count density Trend along Z-Axis", fontsize=14, fontweight='bold')
plt.xlabel("Slice Depth (Z-Index)", fontsize=12)
plt.ylabel("Total Vessel Count Density", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 如果想让 X 轴更紧凑
plt.xlim(plot_df['Slice_Index'].min() - 1, plot_df['Slice_Index'].max() + 1)

# 保存图片
save_path = f"{save_plot_dir}/Comparison_Vessel_count_density_Trend.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')


