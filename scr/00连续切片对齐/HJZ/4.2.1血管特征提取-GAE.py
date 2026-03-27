'''
3D tissue project.IF血管重建流程
4.2.1 血管特征提取 - Graph AutoEncoder (GAE)
'''
#%% ======================= 导入库 =======================
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
import random

# 图像处理
from skimage.morphology import skeletonize, binary_closing, disk
from scipy.ndimage import distance_transform_edt
from skimage.measure import label

# 机器学习
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# GNN
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv,GAE,global_mean_pool,global_max_pool
from torch_geometric.loader import DataLoader

# 可视化
import matplotlib.colors as mcolors
from PIL import Image
import seaborn as sns

#%% 图构建函数
def build_long_range_graph(mask_path, signal_path, sample_step=3, signal_threshold=None, node_sampling_mode='step'):
    """
    构建包含局部连接和长程连接的血管图。
    
    参数:
        node_sampling_mode (str): 
            - 'step': (默认) 按照索引间隔采样，速度快但可能受骨架走向影响。
            - 'knn_radius': 使用KNN半径抑制进行空间均匀采样，节点分布更合理。
    """
    # 1. 读取与预处理
    mask = tifffile.imread(mask_path)
    signal = tifffile.imread(signal_path)
    if np.sum(mask) == 0: return None

    if signal_threshold is None:
        signal_threshold = np.percentile(signal[mask > 0], 75)

    wall_mask = (signal > signal_threshold) & (mask > 0)
    wall_mask = binary_closing(wall_mask, footprint=disk(3))
    if np.sum(wall_mask) == 0: return None

    # 2. 骨架化
    skel = skeletonize(wall_mask)
    y_all, x_all = np.where(skel)
    if len(y_all) < 5: return None

    # ==========================================
    # [新功能] 节点选择策略 (Node Selection)
    # ==========================================
    if node_sampling_mode == 'step':
        # 原始方法：按索引间隔采样 (Raster scan order)
        idx = np.arange(0, len(y_all), sample_step)
        y_nodes, x_nodes = y_all[idx], x_all[idx]
        
    elif node_sampling_mode == 'knn_radius':
        # 新方法：基于空间的均匀采样 (Iterative Radius Suppression)
        # 1. 获取所有骨架点坐标
        all_coords = np.stack([y_all, x_all], axis=1)
        
        # 2. 初始化
        kept_indices = []
        # 使用 KNN 构建器来辅助查找 (radius search)
        # 这里为了效率，我们手动实现一个贪心策略
        # 先随机打乱，避免光栅顺序带来的偏差
        perm = np.random.permutation(len(all_coords))
        shuffled_coords = all_coords[perm]
        
        # 标记数组：0表示未处理，1表示保留，-1表示被抑制
        status = np.zeros(len(shuffled_coords), dtype=int)
        
        # 使用 sklearn 的 RadiusNeighbors 进行批量查询可能较慢，
        # 对于骨架化后的点，简单的网格下采样或 KDTree 是更优解。
        # 这里演示使用 NearestNeighbors 的 radius 逻辑：
        # 实际上，最快的方法是使用 Grid Subsampling (体素网格下采样) 模拟半径采样
        
        # --- 替代方案：网格下采样 (极快且效果等同于均匀 KNN) ---
        # 将坐标除以步长并取整，每个网格只保留一个点
        grid_coords = (shuffled_coords / sample_step).astype(int)
        # 使用 pandas 或 numpy unique 找到唯一网格的索引
        _, unique_indices = np.unique(grid_coords, axis=0, return_index=True)
        
        # 还原回原始顺序的坐标
        selected_coords = shuffled_coords[unique_indices]
        y_nodes, x_nodes = selected_coords[:, 0], selected_coords[:, 1]
        
    else:
        raise ValueError(f"Unknown mode: {node_sampling_mode}")

    num_nodes = len(y_nodes)
    if num_nodes < 5: return None

    # 3. 特征提取
    coords = np.stack([y_nodes, x_nodes], axis=1).astype(np.float32)
    coords_norm = coords / np.array(mask.shape)

    signal_norm = signal[y_nodes, x_nodes].astype(np.float32) / (signal.max() + 1e-5)
    dist_map = distance_transform_edt(mask > 0)
    radius = dist_map[y_nodes, x_nodes]

    x_tensor = torch.tensor(
        np.column_stack([coords_norm, radius, signal_norm]),
        dtype=torch.float
    )

    # 4. 构建边 (使用 KNN 查找邻居)
    # k_neighbors 决定了搜索范围
    k_neighbors = min(20, num_nodes - 1)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(coords)
    dists, idxs = nbrs.kneighbors(coords)

    src, dst, edge_type = [], [], []
    
    for i in range(num_nodes):
        for k, (j, d) in enumerate(zip(idxs[i], dists[i])):
            if i == j: continue
            
            # 策略：混合连接 (Fusion)
            # 0类边：局部紧密连接 (维持血管拓扑)
            if k < 5 and d < sample_step * 4:
                src.append(i); dst.append(j); edge_type.append(0)
            
            # 1类边：长程跳跃连接 (捕捉宏观形态，跨越局部弯曲)
            elif k >= 5 and sample_step * 5 < d < sample_step * 20:
                src.append(i); dst.append(j); edge_type.append(1)

    return Data(
        x=x_tensor,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_type, dtype=torch.long)
    )

#%% 模型
### [GAE CHANGE]
class VesselEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, embedding_dim=32, heads=4):
        super(VesselEncoder, self).__init__()
        
        # Layer 1
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads) # 加 BN
        
        # Layer 2
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads) # 加 BN
        
        # Layer 3
        self.conv3 = GATv2Conv(hidden_dim * heads, embedding_dim, heads=1, concat=False)
        
    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x) # BN
        x = F.elu(x)    # ELU
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x) # BN
        x = F.elu(x)    # ELU
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Layer 3 (Output Z)
        z = self.conv3(x, edge_index)
        return z
    
### [GAE CHANGE]
def train_gae(model, loader, device, epochs=15, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    # 外层进度条：显示 Epoch 进度
    epoch_pbar = tqdm(range(epochs), desc="Training GAE", unit="epoch")
    
    for ep in epoch_pbar:
        loss_all = 0
        batch_count = 0
        
        # 内层进度条：显示 Batch 进度 (leave=False 表示跑完当前Epoch后该条消失，保持界面整洁)
        batch_pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}", leave=False, unit="batch")
        
        for batch in batch_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 前向传播与计算 Loss
            z = model.encode(batch.x, batch.edge_index)
            loss = model.recon_loss(z, batch.edge_index)
            
            # 异常检测
            if torch.isnan(loss):
                print(f"\nError: Loss is NaN at Epoch {ep+1}!")
                return 

            loss.backward()
            optimizer.step()
            
            # 累计 Loss
            current_loss = loss.item()
            loss_all += current_loss
            batch_count += 1
            
            # 实时更新内层进度条的后缀 (显示当前 Batch 的 Loss)
            batch_pbar.set_postfix({"Batch Loss": f"{current_loss:.4f}"})
        
        # 计算当前 Epoch 的平均 Loss
        avg_loss = loss_all / len(loader)
        
        # 更新外层进度条的信息
        epoch_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

#%% 节点特征可视化
### [GAE CHANGE]
def visualize_node_features(model, mask_path, signal_path, output_path, mode='pca'):
    graph = build_long_range_graph(mask_path, signal_path)
    if graph is None:
        return
    graph = graph.to(device)

    model.eval()
    with torch.no_grad():
        node_feats = model.encode(graph.x, graph.edge_index).cpu().numpy()

    if mode == 'pca':
        node_scalars = PCA(1).fit_transform(node_feats).flatten()
        title_suffix = "PCA Component 1 (Pattern Distribution)"
    else:
        node_scalars = np.linalg.norm(node_feats, axis=1)
        title_suffix = "Feature Activation Intensity (L2 Norm)"

    node_scalars = (node_scalars - node_scalars.min()) / (node_scalars.max() - node_scalars.min() + 1e-6)

    signal = tifffile.imread(signal_path)
    h, w = signal.shape
    py = graph.x[:, 0].cpu().numpy() * h
    px = graph.x[:, 1].cpu().numpy() * w

    local_mask = graph.edge_attr.cpu().numpy() == 0
    src, dst = graph.edge_index.cpu().numpy()[:, local_mask]

    segments, colors = [], []
    cmap = plt.get_cmap('viridis' if mode == 'pca' else 'magma')

    for s, d in zip(src, dst):
        segments.append([(px[s], py[s]), (px[d], py[d])])
        colors.append(cmap((node_scalars[s] + node_scalars[d]) / 2))

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(signal, cmap='gray', vmin=np.percentile(signal, 5), vmax=np.percentile(signal, 99), alpha=0.6)
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=2, alpha=0.9))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Normalized Feature Value')
    
    ax.set_title(f"Node-Level Feature Visualization\n{title_suffix}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

#%% 聚类信息映射回原图
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

#%% 可视化检查工具
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


#%% 主程序
#%% ======================= 主流程 =======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

## ======================== 配置路径 =======================
ORIGINAL_MASK_LARGE_FILE = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/prediction_postprocessed/ROI-00004-02772-17487/HJZ_1-00004-02772-17487_pred_mask_processed.tif"
mask_dir = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/GAT_sldies/Single_Masks/"
signal_dir = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/GAT_sldies/Single_Signals/"
output_vis_dir = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/GAT_sldies/"
    
mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
signal_files = sorted(glob.glob(os.path.join(signal_dir, "*.tif")))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = VesselEncoder()
model = GAE(encoder).to(device)

graphs = []
valid_files = []

for m, s in zip(mask_files, signal_files):
    g = build_long_range_graph(m, s, node_sampling_mode='knn_radius')
    if g is not None:
        graphs.append(g)
        valid_files.append(m)

loader = DataLoader(graphs, batch_size=32, shuffle=True)

### [GAE CHANGE] 训练
train_gae(model, loader, device, epochs=20)

### [GAE CHANGE] 提取 graph-level embedding
model.eval()
all_embeddings = []

with torch.no_grad():
    for g in graphs:
        g = g.to(device)
        z = model.encode(g.x, g.edge_index)
        mean = global_mean_pool(z, torch.zeros(z.size(0), dtype=torch.long, device=device))
        maxp = global_max_pool(z, torch.zeros(z.size(0), dtype=torch.long, device=device))
        all_embeddings.append(torch.cat([mean, maxp], dim=1).cpu().numpy().flatten())

feature_matrix = np.vstack(all_embeddings)

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


## 节点特征可视化
for i, (m_path, s_path) in enumerate(zip(mask_files, signal_files)):
    visualize_long_range_check(m_path, s_path, os.path.join(f'{output_vis_dir}/Node_plot2', f"GAT_Graph_Check_{i}.png"))

    ## ======================== 节点特征可视化 ========================
    target_idx = i 
    if target_idx < len(mask_files):
        sample_mask = mask_files[target_idx]
        sample_signal = signal_files[target_idx]

        # 方式 A: 看“特征强度” (哪里最重要?)
        out_path_norm = os.path.join(f'{output_vis_dir}/Node_plot2', f"NodeViz_Intensity_{target_idx}.png")
        visualize_node_features(model, sample_mask, sample_signal, out_path_norm, mode='norm')

        # 方式 B: 看“模式分布” (头和尾是不是不同?)
        out_path_pca = os.path.join(f'{output_vis_dir}/Node_plot2', f"NodeViz_PCA_{target_idx}.png")
        visualize_node_features(model, sample_mask, sample_signal, out_path_pca, mode='pca')
        print("节点级可视化已完成")


#%% 构建ID映射，进行血管指标映射
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
    y="Area_um2",
    palette=color_list_0_1,
    showfliers=False # 不显示异常值点，因为后面要叠加散点
)
plt.title("Vessel Area Comparison")
plt.ylabel("Area_um2")
plt.xlabel("") # 移除X轴标签，因为图例已经说明了
#plt.axis(ymin=0, ymax=200) # Y轴从0开始
sns.despine(offset=10, trim=True) # 去掉顶部和右侧边框，更美观
save_current_plot('/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/GAT_sldies', "HJZ_1_Comp_Area_Boxplot")

