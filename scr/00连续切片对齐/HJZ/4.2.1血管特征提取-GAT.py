'''
3D tissue project.IF血管重建流程.4.2.1血管特征提取-GAT 的 Docstring
'''
#%% 导入库
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import glob
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from tqdm import tqdm
import cv2
import pandas as pd
# 图像处理库
from skimage.morphology import skeletonize, binary_closing, disk
from scipy.ndimage import distance_transform_edt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from skimage.measure import label, regionprops
# 图神经网络库
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
import matplotlib.colors as mcolors
import matplotlib as mpl
from PIL import Image
import random
from sklearn.decomposition import PCA
from matplotlib.collections import LineCollection
import seaborn as sns


#%% 1. 图构建函数 (支持长程连接)
def build_long_range_graph(mask_path, signal_path, sample_step=3, signal_threshold=None):
    """
    构建包含局部连接和长程连接的图。
    """
    # --- 读取与预处理 ---
    mask = tifffile.imread(mask_path)
    signal = tifffile.imread(signal_path)
    
    if np.sum(mask) == 0: return None

    # 自动阈值 + 闭运算修复微小断裂
    if signal_threshold is None:
        # 取血管区域内信号的 75 分位数作为阈值
        signal_threshold = np.percentile(signal[mask > 0], 75)
    
    # 定义高信号的血管壁区域
    wall_mask = (signal > signal_threshold) & (mask > 0)
    # 使用闭运算物理连接微小的断点 (半径3)
    wall_mask = binary_closing(wall_mask, footprint=disk(3))
    
    if np.sum(wall_mask) == 0: return None
        
    # --- 骨架化与采样 ---
    wall_skeleton = skeletonize(wall_mask)
    y_all, x_all = np.where(wall_skeleton)
    
    if len(y_all) < 5: return None

    indices = np.arange(0, len(y_all), sample_step)
    y_nodes = y_all[indices]
    x_nodes = x_all[indices]
    num_nodes = len(y_nodes)
    
    # --- 提取节点特征 [Y, X, Radius, Signal] ---
    # 1. 归一化坐标
    coords = np.stack([y_nodes, x_nodes], axis=1).astype(np.float32)
    coords_norm = coords / np.array(mask.shape)
    
    # 2. 归一化信号强度
    signal_values = signal[y_nodes, x_nodes].astype(np.float32)
    signal_norm = signal_values / (signal.max() + 1e-5)
    
    # 3. 管径/距离特征
    dist_transform = distance_transform_edt(mask > 0)
    radius_values = dist_transform[y_nodes, x_nodes]
    
    features = np.column_stack([coords_norm, radius_values, signal_norm])
    x_tensor = torch.tensor(features, dtype=torch.float)
    
    # --- 核心：构建多尺度边 (KNN) ---
    # 我们搜索 20 个邻居，而不仅仅是 5 个
    k_neighbors = min(20, num_nodes-1)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree').fit(coords)
    dists, idxs = nbrs.kneighbors(coords)
    
    src_list = []
    dst_list = []
    edge_types = [] # 0: Local, 1: Long-range
    
    # 定义距离阈值
    dist_threshold_local = sample_step * 4.0   # 局部连接容忍度
    min_dist_global = sample_step * 5.0        # 长程连接最小距离 (避开局部)
    max_dist_global = sample_step * 20.0       # 长程连接最大距离 (不要连太远)
    
    for i in range(num_nodes):
        neighbors = idxs[i]
        distances = dists[i]
        
        for k_idx, (neighbor_node, dist) in enumerate(zip(neighbors, distances)):
            if i == neighbor_node: continue 
            
            # 策略 A: 局部连接 (前 5 个最近的邻居)
            if k_idx < 5 and dist < dist_threshold_local:
                src_list.append(i)
                dst_list.append(neighbor_node)
                edge_types.append(0) # 0 代表局部边
            
            # 策略 B: 长程连接 (第 5-20 个邻居)
            # 只有当距离足够远(跨越空间)但又没太远(保持关联)时才连接
            elif k_idx >= 5 and (dist > min_dist_global and dist < max_dist_global):
                src_list.append(i)
                dst_list.append(neighbor_node)
                edge_types.append(1) # 1 代表长程边

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    # edge_attr 可以用来在 GNN 中区分边的类型 (可选)
    edge_attr = torch.tensor(edge_types, dtype=torch.float) 
    
    return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)


#%% 2. GAT 模型定义 (Graph Attention Network)
class VesselGAT(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, embedding_dim=32, heads=4):
        super(VesselGAT, self).__init__()
        
        # GATv2Conv 是更强的注意力机制，比标准 GAT 收敛更好
        # heads=4: 多头注意力，模型会从 4 个不同角度观察连接
        # concat=True: 输出维度会变成 hidden_dim * heads
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        
        # Layer 2
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        
        # Layer 3 (输出层)
        # concat=False: 将多头结果取平均，变回 embedding_dim
        self.conv3 = GATv2Conv(hidden_dim * heads, embedding_dim, heads=1, concat=False)
        
    def forward(self, data, return_node_feats=False):
        x, edge_index = data.x, data.edge_index
        # 获取 batch 索引 (如果是单图处理，会自动生成全0 batch)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x) # GAT 通常配合 ELU 激活函数
        x = F.dropout(x, p=0.2, training=self.training) # 防止过拟合
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Layer 3 (得到节点级特征)
        node_embeddings = self.conv3(x, edge_index)
        
        if return_node_feats:
            return node_embeddings
        
        # Readout (全局池化：将节点特征聚合为整图特征)
        # 融合 Mean (平均水平) 和 Max (最显著特征)
        feat_mean = global_mean_pool(node_embeddings, batch)
        feat_max = global_max_pool(node_embeddings, batch)
        
        # 拼接得到最终的 64维 特征
        return torch.cat([feat_mean, feat_max], dim=1)


#%% 3. 可视化检查工具 (查看长程连接)
def visualize_long_range_check(mask_path, signal_path, output_path):
    print(f"正在生成图结构可视化: {output_path}")
    graph = build_long_range_graph(mask_path, signal_path)
    if graph is None:
        print("构图失败 (空图)")
        return

    signal = tifffile.imread(signal_path)
    h, w = signal.shape
    
    # 坐标反归一化
    pixel_y = graph.x[:, 0].numpy() * h
    pixel_x = graph.x[:, 1].numpy() * w
    
    edge_index = graph.edge_index.numpy()
    edge_types = graph.edge_attr.numpy()
    
    # 分离局部边和长程边
    local_mask = edge_types == 0
    global_mask = edge_types == 1
    
    segments_local = []
    for s, d in zip(edge_index[0][local_mask], edge_index[1][local_mask]):
        segments_local.append([(pixel_x[s], pixel_y[s]), (pixel_x[d], pixel_y[d])])

    segments_global = []
    for s, d in zip(edge_index[0][global_mask], edge_index[1][global_mask]):
        segments_global.append([(pixel_x[s], pixel_y[s]), (pixel_x[d], pixel_y[d])])

    fig, ax = plt.subplots(figsize=(10, 10))
    vmin, vmax = np.percentile(signal, [5, 99])
    ax.imshow(signal, cmap='gray', vmin=vmin, vmax=vmax, alpha=0.8)
    
    # 绘制局部边 (黄色)
    lc_local = LineCollection(segments_local, colors='yellow', linewidths=1.0, alpha=0.6, label='Local')
    ax.add_collection(lc_local)
    
    # 绘制长程边 (青色 - 这就是你要的"远点作用")
    lc_global = LineCollection(segments_global, colors='cyan', linewidths=0.6, alpha=0.5, label='Long-range')
    ax.add_collection(lc_global)
    
    ax.scatter(pixel_x, pixel_y, c='red', s=3, zorder=10)
    
    # 自定义图例
    custom_lines = [Line2D([0], [0], color='yellow', lw=2),
                    Line2D([0], [0], color='cyan', lw=2)]
    ax.legend(custom_lines, ['Local', 'Long-range (Global)'], loc='upper right')
    
    plt.title("GAT Graph Construction\n(Yellow: Neighbors, Cyan: Long-range Interaction)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


#%% 4. 池化之后特征映射绘图函数
def reconstruct_continuous_feature_map(original_binary_mask_path, valid_files_list, feature_values, output_path, feature_name="Feature_X"):
    """
    将连续的特征数值映射回血管图像 (热图模式)。
    """
    print(f"[Vis] 正在生成特征热图: {feature_name} ...")
    
    # 1. 归一化特征值到 0-1 之间 (用于颜色映射)
    f_min, f_max = feature_values.min(), feature_values.max()
    norm_values = (feature_values - f_min) / (f_max - f_min + 1e-6)
    
    # 2. 使用连续色谱 (推荐 magma, plasma, viridis)
    cmap = plt.get_cmap('magma') 
    
    # 3. 读取并 Label
    original_mask = tifffile.imread(original_binary_mask_path)
    labeled_img = label(original_mask > 0, connectivity=2)
    h, w = labeled_img.shape
    
    # 初始化黑色背景
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 4. 建立映射
    id_pattern = re.compile(r"_ID_(\d+)_mask\.tif")
    id_to_value = {}
    
    for filepath, val in zip(valid_files_list, norm_values):
        match = id_pattern.search(os.path.basename(filepath))
        if match:
            obj_id = int(match.group(1))
            if obj_id <= labeled_img.max():
                id_to_value[obj_id] = val

    # 5. 渲染
    for obj_id, val in tqdm(id_to_value.items(), desc=f"Rendering {feature_name}"):
        # 获取颜色 (R,G,B,A) -> (R,G,B) 0-255
        color = [int(c*255) for c in cmap(val)[:3]]
        vis_img[labeled_img == obj_id] = color
        
    # 6. 保存
    try:
        Image.fromarray(vis_img).save(output_path, dpi=(300,300))
        print(f"[Success] 保存成功: {output_path}")
    except Exception as e:
        print(e)


#%% 5. 聚类映射回原图绘图函数

Image.MAX_IMAGE_PIXELS = None 

def reconstruct_cluster_map(original_binary_mask_path, valid_files_list, cluster_labels, output_path, n_clusters, color_list_rgb):
    """
    Args:
        color_list_rgb (list): 一个包含 n_clusters 个颜色的列表，格式为 [[R, G, B], [R, G, B], ...], 范围 0-255。
                               这必须与 t-SNE 图使用的颜色完全一致。
    """
    print(f"\n[Process] 开始重建映射: {os.path.basename(output_path)}")

    # 1. 颜色生成 (直接使用传入的列表，确保一致性)
    print(f"[Process] 使用主程序传入的颜色列表...")
    cluster_colors = color_list_rgb
    
    # 打印颜色核对
    for i, c in enumerate(cluster_colors):
        print(f"  -> Cluster {i} Color: {c}")

    # 2. 读取并重新标记 (保持不变)
    if not os.path.exists(original_binary_mask_path):
        print(f"[Error] 文件不存在: {original_binary_mask_path}")
        return

    print("[Process] 读取原始大图掩码...")
    original_mask = tifffile.imread(original_binary_mask_path)
    mask_bin = original_mask > 0
    
    print("[Process] 重新计算连通域 (Labeling)...")
    labeled_img = label(mask_bin, connectivity=2)
    
    h, w = labeled_img.shape
    cluster_vis_img = np.zeros((h, w, 3), dtype=np.uint8)

    # 3. 建立 ID 映射 (保持不变)
    id_to_cluster_map = {}
    id_pattern = re.compile(r"_ID_(\d+)_mask\.tif")
    max_id_in_img = labeled_img.max()
    
    print("[Process] 解析文件名并匹配 ID...")
    for filepath, label_idx in zip(valid_files_list, cluster_labels):
        filename = os.path.basename(filepath)
        match = id_pattern.search(filename)
        if match:
            obj_id = int(match.group(1))
            if obj_id <= max_id_in_img:
                id_to_cluster_map[obj_id] = label_idx

    # 4. 绘图 (填充颜色) (保持不变)
    print(f"[Process] 正在渲染 {len(id_to_cluster_map)} 个血管对象...")
    for obj_id, cluster_idx in tqdm(id_to_cluster_map.items(), desc="Rendering"):
        mask_bool = (labeled_img == obj_id)
        cluster_vis_img[mask_bool] = cluster_colors[cluster_idx]

    # 5. 保存 (保持不变)
    print(f"[Process] 正在保存高分辨率 PNG (使用 PIL)...")
    try:
        img_pil = Image.fromarray(cluster_vis_img)
        img_pil.save(output_path, format='PNG', dpi=(300, 300), compress_level=1)
        print(f"[Success] 图片已保存至: {output_path}")
    except Exception as e:
        print(f"[Error] 保存失败: {e}")


#%% 6. 节点特征可视化函数
def visualize_node_features(model, mask_path, signal_path, output_path, mode='pca'):
    """
    可视化血管内部的微观特征分布。
    Args:
        mode: 'pca' (主成分, 区分不同区域) 或 'norm' (L2范数, 也就是特征激活强度)
    """
    print(f"[Vis] 正在生成节点级特征图: {os.path.basename(output_path)}")
    
    # 1. 建图
    graph = build_long_range_graph(mask_path, signal_path)
    if graph is None: return
    graph = graph.to(device)
    
    # 2. 获取节点特征 (不进行梯度计算)
    model.eval()
    with torch.no_grad():
        # [Num_Nodes, 32]
        node_feats = model(graph, return_node_feats=True).cpu().numpy()
        
    # 3. 特征降维 (32维 -> 1维标量)
    if mode == 'pca':
        # 使用 PCA 取包含信息量最大的第一主成分
        pca = PCA(n_components=1)
        # 归一化到 0-1 以便绘图
        node_scalars = pca.fit_transform(node_feats).flatten()
        # Min-Max 归一化
        node_scalars = (node_scalars - node_scalars.min()) / (node_scalars.max() - node_scalars.min())
        cmap_name = 'viridis' # PCA 适合用彩虹色看分布
        title_suffix = "PCA Component 1 (Pattern Distribution)"
    else:
        # 使用 L2 范数 (看哪里“反应”最剧烈)
        node_scalars = np.linalg.norm(node_feats, axis=1)
        node_scalars = (node_scalars - node_scalars.min()) / (node_scalars.max() - node_scalars.min())
        cmap_name = 'magma'   # 强度适合用火热色
        title_suffix = "Feature Activation Intensity (L2 Norm)"

    # 4. 准备绘图数据
    signal = tifffile.imread(signal_path)
    h, w = signal.shape
    
    # 坐标反归一化
    pixel_y = graph.x[:, 0].cpu().numpy() * h
    pixel_x = graph.x[:, 1].cpu().numpy() * w
    
    edge_index = graph.edge_index.cpu().numpy()
    edge_attr = graph.edge_attr.cpu().numpy() # 0=Local, 1=Global
    
    # 我们只画局部边 (Local Edges) 来重建血管形态
    # 长程边画出来会像蜘蛛网，干扰观察血管内部变化
    local_mask = edge_attr == 0
    src_nodes = edge_index[0][local_mask]
    dst_nodes = edge_index[1][local_mask]
    
    # 构建线段集合
    segments = []
    colors = []
    
    # 获取 colormap
    cmap = plt.get_cmap(cmap_name)
    
    for s, d in zip(src_nodes, dst_nodes):
        # 线段坐标
        p1 = (pixel_x[s], pixel_y[s])
        p2 = (pixel_x[d], pixel_y[d])
        segments.append([p1, p2])
        
        # 线段颜色：取两端节点特征的平均值
        val_mean = (node_scalars[s] + node_scalars[d]) / 2.0
        colors.append(cmap(val_mean))

    # 5. 绘图
    fig, ax = plt.subplots(figsize=(12, 12))
    # 背景画淡一点的原始信号
    ax.imshow(signal, cmap='gray', vmin=np.percentile(signal, 5), vmax=np.percentile(signal, 99), alpha=0.6)
    
    # 创建 LineCollection
    lc = LineCollection(segments, colors=colors, linewidths=2.5, alpha=0.9)
    ax.add_collection(lc)
    
    # 加上 Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Normalized Feature Value')
    
    ax.set_title(f"Node-Level Feature Visualization\n{title_suffix}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()




#%% 主流程 (特征提取与聚类)
## ======================== 随机种子设置 ========================
SEED = 42
random.seed(SEED)             # Python内置随机库
np.random.seed(SEED)          # Numpy (用于 sklearn, pandas 等)
torch.manual_seed(SEED)       # PyTorch CPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)      # PyTorch GPU
    torch.cuda.manual_seed_all(SEED)  # PyTorch Multi-GPU

print(f"随机种子已固定为: {SEED}")

## ======================== 配置路径 =======================
ORIGINAL_MASK_LARGE_FILE = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/prediction_postprocessed/ROI-00004-02772-17487/HJZ_1-00004-02772-17487_pred_mask_processed.tif"
mask_dir = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/GAT_sldies/Single_Masks/"
signal_dir = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/GAT_sldies/Single_Signals/"
output_vis_dir = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/GAT_sldies/"
    
mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
signal_files = sorted(glob.glob(os.path.join(signal_dir, "*.tif")))
    

# --- 初始化设备和模型 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
    
model = VesselGAT(input_dim=4, hidden_dim=64, embedding_dim=32, heads=4).to(device)
model.eval() # 推理模式
    
all_embeddings = []
valid_files = []

print(f"开始 GAT 特征提取 (共 {len(mask_files)} 张切片)...")
    
# --- 循环处理 ---
for i, (m_path, s_path) in enumerate(zip(mask_files, signal_files)):
    try:
        # 1. 抽查可视化 (只画第20张，看看长程连接效果)
        if i == 48: 
            visualize_long_range_check(m_path, s_path, os.path.join(output_vis_dir, f"GAT_Graph_Check_{i}.png"))
        
        # 2. 构建图
        graph = build_long_range_graph(m_path, s_path)
        if graph is None: continue
        
        graph = graph.to(device)
        
        # 3. GAT 推理
        with torch.no_grad():
            emb = model(graph) # Shape [1, 64]
            
        all_embeddings.append(emb.cpu().numpy().flatten())
        valid_files.append(m_path)
        
    except Exception as e:
        print(f"Error processing {os.path.basename(m_path)}: {e}")

# --- 聚类与结果 ---
feature_matrix = np.vstack(all_embeddings)
print(f"\n成功提取特征矩阵: {feature_matrix.shape}")
    
# 标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_matrix)
    
# K-Means 聚类
n_clusters = 3
print(f"正在进行 K-Means 聚类 (k={n_clusters})...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(features_scaled)
    
# DBSCAN 聚类 
# Leiden 聚类

## ======================== 聚类结果统计 ========================
for i in range(n_clusters):
    print(f"Cluster {i}: {np.sum(labels == i)} slices")

# t-SNE 可视化 (修改为离散颜色)
print("绘制 t-SNE (使用离散颜色)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
features_2d = tsne.fit_transform(features_scaled)
plt.figure(figsize=(10, 8))

# --- 1. 选择离散色卡 ---
# 推荐色卡: 'tab10' (最多10类), 'Set1' (最多9类), 'Dark2' (最多8类)
# 如果类别非常多 (>10)，可以使用 'hsv' 但难以区分
DISCRETE_CMAP_NAME = 'tab10' 
base_cmap = plt.get_cmap(DISCRETE_CMAP_NAME)

# 对于离散色卡(tab10, Set1)，直接按索引取前 n_clusters 个颜色，保证 distinct
# tab10 的顺序是: 0:蓝, 1:橙, 2:绿, 3:红 ...
color_list_0_1 = [base_cmap(i) for i in range(n_clusters)] 

# 创建用于 t-SNE 的 colormap 对象
discrete_cmap = mcolors.ListedColormap(color_list_0_1)

# 创建用于 Reconstruct 的 RGB (0-255) 列表
color_list_0_255 = [[int(c*255) for c in color[:3]] for color in color_list_0_1]

print("已生成统一颜色列表:")
for idx, rgb in enumerate(color_list_0_255):
    print(f"  Cluster {idx}: {rgb}")

# t-SNE 可视化
print("绘制 t-SNE (使用离散颜色)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
features_2d = tsne.fit_transform(features_scaled)

plt.figure(figsize=(10, 8))

# 使用上面生成的 discrete_cmap
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                      c=labels, 
                      cmap=discrete_cmap, 
                      vmin=-0.5, vmax=n_clusters-0.5, # 确保颜色对齐
                      alpha=0.8, s=40, edgecolor='k', linewidth=0.3)

ticks = np.arange(n_clusters)
cbar = plt.colorbar(scatter, ticks=ticks)
cbar.ax.set_yticklabels([f'Cluster {i}' for i in ticks])
cbar.set_label('Cluster ID', fontsize=12)  
plt.title(f"GAT-based Clustering\nTotal: {len(labels)} vessels")

tsne_output_path = os.path.join(output_vis_dir, "GAT_Clustering_tSNE_Discrete.png")
plt.savefig(tsne_output_path, dpi=300, bbox_inches='tight')
plt.close()

## ======================== 聚类结果映射回原图(保证颜色与聚类结果一致) ========================
reconstruct_output_path = os.path.join(output_vis_dir, "GAT_Cluster_Reconstruction_Map_Discrete.png")
reconstruct_cluster_map(
    original_binary_mask_path=ORIGINAL_MASK_LARGE_FILE,
    valid_files_list=valid_files,
    cluster_labels=labels,
    output_path=reconstruct_output_path,
    n_clusters=n_clusters,
    color_list_rgb=color_list_0_255  # <--- 传入这个列表，而不是色卡名
)


## ======================== 节点特征可视化 ========================
target_idx = 48 
if target_idx < len(mask_files):
    sample_mask = mask_files[target_idx]
    sample_signal = signal_files[target_idx]
    
    # 方式 A: 看“特征强度” (哪里最重要?)
    out_path_norm = os.path.join(output_vis_dir, f"NodeViz_Intensity_{target_idx}.png")
    visualize_node_features(model, sample_mask, sample_signal, out_path_norm, mode='norm')
    
    # 方式 B: 看“模式分布” (头和尾是不是不同?)
    out_path_pca = os.path.join(output_vis_dir, f"NodeViz_PCA_{target_idx}.png")
    visualize_node_features(model, sample_mask, sample_signal, out_path_pca, mode='pca')

    print("节点级可视化已完成，快去看看图吧！")

## ======================== 池化之后的特征映射 ========================
variances = np.var(feature_matrix, axis=0)
top_feature_indices = np.argsort(variances)[::-1][:3] # 取前3个最重要的特征

print(f"方差最大的特征维度索引: {top_feature_indices}")

for idx in top_feature_indices:
    # 取出这一列特征
    single_dim_feature = feature_matrix[:, idx]
    out_name = os.path.join(output_vis_dir, f"GAT_Feature_Dim_{idx}_Heatmap.png")
    reconstruct_continuous_feature_map(
        original_binary_mask_path=ORIGINAL_MASK_LARGE_FILE,
        valid_files_list=valid_files,
        feature_values=single_dim_feature,
        output_path=out_name,
        feature_name=f"Dim_{idx}"
    )

#%% 构架ID映射
original_mask = tifffile.imread(ORIGINAL_MASK_LARGE_FILE)
mask_bin = original_mask > 0
    
print("[Process] 重新计算连通域 (Labeling)...")
labeled_img = label(mask_bin, connectivity=2)
    
h, w = labeled_img.shape
cluster_vis_img = np.zeros((h, w, 3), dtype=np.uint8)

# 3. 建立 ID 映射 (保持不变)
id_to_cluster_map = {}
id_pattern = re.compile(r"_ID_(\d+)_mask\.tif")
max_id_in_img = labeled_img.max()
    
print("[Process] 解析文件名并匹配 ID...")
for filepath, label_idx in zip(valid_files, labels):
    filename = os.path.basename(filepath)
    match = id_pattern.search(filename)
    if match:
        obj_id = int(match.group(1))
        if obj_id <= max_id_in_img:
            id_to_cluster_map[obj_id] = label_idx

HJZ_1_stat_df = pd.read_csv('/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/cross_section_analysis/ROI-00004-02772-17487/HJZ_1-00004-02772-17487_pred_mask_processed.tif_CrossSection_Stats.csv')

HJZ_1_stat_df['Cluster_Class'] = HJZ_1_stat_df['Object_ID'].map(id_to_cluster_map)


def save_current_plot(save_plot_dir, filename):
    filepath = os.path.join(save_plot_dir, filename)
    # 保存为矢量图(pdf)和高清位图(png)
    plt.savefig(filepath + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(filepath + ".pdf", bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {filename}")

# 设置 seaborn 主题风格，让图表更适合出版
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
# 定义颜色板 (可以根据需要修改)
palette = sns.color_palette("Set2", n_colors=4)

plt.figure(figsize=(8, 6))
# 画箱线图
ax = sns.boxplot(
    data=HJZ_1_stat_df,
    x="Cluster_Class",
    y="Mean_Intensity",
    palette=color_list_0_1,
    showfliers=False # 不显示异常值点，因为后面要叠加散点
)
plt.title("Vessel Mean_Intensity Comparison")
plt.ylabel("Mean_Intensity")
plt.xlabel("") # 移除X轴标签，因为图例已经说明了
#plt.axis(ymin=0, ymax=200) # Y轴从0开始
sns.despine(offset=10, trim=True) # 去掉顶部和右侧边框，更美观
save_current_plot('/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/GAT_sldies', "HJZ_1_Comp_Mean_Intensity_Boxplot")
