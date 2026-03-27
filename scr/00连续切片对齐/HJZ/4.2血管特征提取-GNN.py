import numpy as np
import tifffile
import torch
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

import numpy as np
import tifffile
import torch
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

def build_2d_vascular_graph_on_wall(mask_path, signal_path, sample_step=3, signal_threshold=None):
    """
    构建基于血管壁高信号区域的 2D 血管图。
    节点位于信号强度高于阈值的区域的骨架上。
    
    参数:
        signal_threshold: 信号阈值。高于此值的像素被认为是血管壁。
                          如果为 None，将自动计算一个阈值（例如 80 百分位）。
    """
    # 1. 读取数据
    mask = tifffile.imread(mask_path)
    signal = tifffile.imread(signal_path)
    
    # 异常检查
    if np.sum(mask) == 0:
        return None

    # 2. 确定用于生成节点的掩码 (血管壁区域)
    if signal_threshold is None:
        # 如果没有提供阈值，自动计算一个。
        # 这里取血管区域内信号的 80 百分位作为阈值，你可以根据实际情况调整。
        signal_threshold = np.percentile(signal[mask > 0], 80)
        print(f"未提供阈值，自动设定信号阈值为: {signal_threshold:.2f}")
    
    # 创建高信号区域的掩码，并且限定在原始血管掩码范围内
    # 这就是我们定义的“血管壁”区域
    wall_mask = (signal > signal_threshold) & (mask > 0)
    
    if np.sum(wall_mask) == 0:
        print("未找到高于阈值的信号区域，无法构建图。请尝试降低阈值。")
        return None
        
    # 3. 骨架化高信号区域以提取节点
    # 这将得到血管壁的“中心线”，确保节点分布在壁上
    wall_skeleton = skeletonize(wall_mask)
    
    # 4. 提取节点 (2D 坐标)
    y_all, x_all = np.where(wall_skeleton)
    
    # 如果骨架太小，跳过
    if len(y_all) < 5:
        return None

    # 采样：为了控制节点数量，进行间隔采样
    indices = np.arange(0, len(y_all), sample_step)
    y_nodes = y_all[indices]
    x_nodes = x_all[indices]
    num_nodes = len(y_nodes)
    # print(f"在血管壁上提取到 {num_nodes} 个节点。")
    
    # 5. 提取特征
    # A. 信号特征 (CD31强度) - 在这些节点上信号自然较高
    signal_values = signal[y_nodes, x_nodes].astype(np.float32)
    # 归一化
    signal_norm = signal_values / (signal.max() + 1e-5) 
    
    # B. 形态特征 (管径/距离)
    # 计算原始血管掩码的距离变换。
    # 对于血管壁上的点，这个值表示它们距离血管中心线（或背景）有多远。
    original_binary = mask > 0
    dist_transform = distance_transform_edt(original_binary)
    radius_values = dist_transform[y_nodes, x_nodes]
    
    # C. 空间特征 (归一化坐标)
    coords = np.stack([y_nodes, x_nodes], axis=1).astype(np.float32)
    coords_norm = coords / np.array(mask.shape)
    
    # 组合特征: [Y, X, Radius, Signal]
    features = np.column_stack([
        coords_norm, 
        radius_values, 
        signal_norm
    ])
    
    x_tensor = torch.tensor(features, dtype=torch.float)
    
    # 6. 构建边 (基于 2D 欧氏距离)
    # 使用 KNN 连接附近的节点
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(coords)
    dists, idxs = nbrs.kneighbors(coords)
    
    src_list = []
    dst_list = []
    # 连接阈值可能需要调整，因为壁上的点可能分布得更散
    connect_threshold = sample_step * 3.0 
    
    for i in range(num_nodes):
        for k, d in zip(idxs[i], dists[i]):
            if i != k and d < connect_threshold:
                src_list.append(i)
                dst_list.append(k)
                
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    
    return Data(x=x_tensor, edge_index=edge_index)


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class VesselGNN_2D(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, embedding_dim=32):
        super(VesselGNN_2D, self).__init__()
        
        # input_dim = 4 (Y, X, Radius, CD31)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Node Embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        node_embeddings = self.conv3(x, edge_index)
        
        # Graph Embedding (Pooling)
        # 融合 Mean (平均状态) 和 Max (极端状态)
        feat_mean = global_mean_pool(node_embeddings, batch)
        feat_max = global_max_pool(node_embeddings, batch)
        
        return torch.cat([feat_mean, feat_max], dim=1) # Output dim = 64

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VesselGNN_2D().to(device)


import glob
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. 获取文件列表 (使用你之前的路径)
mask_dir = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16_mask/"
signal_dir = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16/"

mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
signal_files = sorted(glob.glob(os.path.join(signal_dir, "*.tif")))

all_embeddings = []
valid_files = []

print(f"找到 {len(mask_files)} 个切面文件，开始提取特征...")

# 2. 循环提取特征
model.eval()
with torch.no_grad():
    for m_path, s_path in zip(mask_files, signal_files):
        try:
            # 构建 2D 图
            graph = build_2d_vascular_graph_on_wall(m_path, s_path)
            
            if graph is None:
                continue # 跳过空图或噪点
            
            graph = graph.to(device)
            
            # GNN 推理
            emb = model(graph) # Shape [1, 64]
            
            all_embeddings.append(emb.cpu().numpy().flatten())
            valid_files.append(m_path)
            
        except Exception as e:
            print(f"Skipping {os.path.basename(m_path)}: {e}")

# 3. 检查与聚类
if len(all_embeddings) > 0:
    feature_matrix = np.vstack(all_embeddings)
    print(f"成功提取特征矩阵: {feature_matrix.shape}")
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)
    
    # K-Means 聚类 (假设分 3 类：强信号、弱信号、中间态)
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    
    # 打印结果统计
    print("\n--- 聚类结果统计 ---")
    for i in range(n_clusters):
        print(f"Cluster {i}: {np.sum(labels == i)} 个切面")
        
    # --- 可视化 (t-SNE) ---
    # 将 64维特征降维到 2维进行展示
    print("\n正在绘制 t-SNE 可视化图...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    features_2d = tsne.fit_transform(features_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f"GNN-based Clustering of 2D Vascular Sections\n(Total: {len(labels)} slices)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")

    save_plot_path = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/vessel_clustering_tsne.png"
    plt.savefig(save_plot_path)
    print(f"t-SNE 图已保存至: {save_plot_path}")
    plt.show()

else:
    print("未提取到有效特征，请检查输入图片是否全黑。")




#%%
import numpy as np
import tifffile
import os
from tqdm import tqdm

def reconstruct_clustered_volume(all_mask_files, valid_files, labels, output_path):
    """
    将 2D 切面的聚类结果堆叠回 3D 空间。
    
    参数:
        all_mask_files: 原始所有 mask 文件的路径列表 (有序)
        valid_files:    参与聚类的 mask 文件路径列表 (对应 labels)
        labels:         聚类结果数组 (对应 valid_files)
        output_path:    保存路径 (.tif)
    """
    print(f"开始重建 3D 聚类体积...")
    
    # 1. 建立映射字典: {文件路径: 聚类ID}
    # 注意：labels 是 0,1,2... 我们存成 1,2,3... 以便 0 留给背景
    file_to_label = {f: l + 1 for f, l in zip(valid_files, labels)}
    
    # 2. 读取第一张图获取尺寸
    first_img = tifffile.imread(all_mask_files[0])
    h, w = first_img.shape
    z = len(all_mask_files)
    
    print(f"   目标尺寸: (Z={z}, Y={h}, X={w})")
    
    # 3. 初始化 3D 矩阵 (使用 uint8 节省内存，支持最多 255 类)
    volume_3d = np.zeros((z, h, w), dtype=np.uint8)
    
    # 4. 逐层填充
    print("   正在堆叠切面...")
    for z_idx, file_path in enumerate(tqdm(all_mask_files)):
        # 检查这个文件是否有聚类结果
        if file_path in file_to_label:
            # 读取 mask
            mask = tifffile.imread(file_path)
            
            # 获取对应的 Label ID
            cluster_id = file_to_label[file_path]
            
            # 将 mask > 0 的区域赋值为 cluster_id
            # 这里利用 mask 本身就是二值的特性
            volume_3d[z_idx][mask > 0] = cluster_id
            
    # 5. 保存结果
    print(f"保存 3D TIF 到: {output_path}")
    tifffile.imwrite(output_path, volume_3d, compression='zlib')
    print("重建完成！")
    return volume_3d

# ================= 运行示例 =================

# 假设你上面的聚类代码已经运行完毕，并且有以下变量：
# mask_files: 所有文件的列表
# valid_files: 成功提取特征的文件列表
# labels: K-Means 的结果

output_tif_path = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16_Clustered_3D_Stack.tif"

# 1. 生成 3D 数据
volume_clustered = reconstruct_clustered_volume(mask_files, valid_files, labels, output_tif_path)



#%%
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import rotate, zoom
import matplotlib.colors as mcolors
from tqdm import tqdm
import gc

def generate_flexible_rotation_gif(tif_path, output_gif_path, downscale_factor=0.2, clusters=10, mode='vertical', fps=10):
    """
    生成多模式 3D 旋转 GIF，支持自动调整画布比例。
    
    参数:
        tif_path: 输入 TIF 路径
        output_gif_path: 输出 GIF 路径
        downscale_factor: 降采样倍数 (0.2 推荐用于预览)
        clusters: 聚类数量 (用于颜色映射)
        mode: 旋转模式
            - 'vertical':   竖直自转 (默认，像陀螺)
            - 'horizontal': 水平滚动 (像烤肠，推荐用于长血管)
            - 'tumble':     前后翻滚 (像体操空翻)
            - 'turnstile':  左右摇摆 (像旋转门)
        fps: 帧率
    """
    print(f"正在读取数据: {tif_path}")
    try:
        volume = tifffile.imread(tif_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {tif_path}")
        return
    
    # 1. 降采样 (预览必备)
    print(f"   原始尺寸: {volume.shape}")
    if downscale_factor and downscale_factor < 1.0:
        print(f"  正在降采样 (Factor={downscale_factor})...")
        # Z轴保持 1.0 (通常层数较少)，XY轴缩放
        volume = zoom(volume, (1.0, downscale_factor, downscale_factor), order=0)
    
    # 2. 根据模式调整数据方向 (关键步骤!)
    # 我们通过交换轴 (Swap Axes) 或改变旋转平面来实现不同的视觉效果
    
    if mode == 'horizontal':
        print("模式: Horizontal Rolling (横着滚)")
        # 原始: (Z, Y, X) -> Z是层数(高), Y是高, X是宽
        # 我们把 Z 轴换到 X 轴的位置，让"高"变成"宽"
        # 这样血管就"躺"在了屏幕上
        volume_to_rotate = np.swapaxes(volume, 0, 2) 
        # 旋转轴: (1, 2) 即绕着新的长轴旋转
        rot_axes = (1, 2)
        proj_axis = 1 # 沿着 Y 轴投影看侧面
        
    elif mode == 'tumble':
        print("模式: Tumble (前后翻滚)")
        volume_to_rotate = volume
        # 在 Z-Y 平面旋转 (绕 X 轴)
        rot_axes = (0, 1)
        # 投影轴选 2 (X轴)，从侧面看它翻滚
        proj_axis = 2
        
    elif mode == 'turnstile':
        print("模式: Turnstile (左右转)")
        volume_to_rotate = volume
        # 在 Z-X 平面旋转 (绕 Y 轴)
        rot_axes = (0, 2)
        proj_axis = 1
        
    else: # 'vertical'
        print("模式: Vertical Spin (竖直自转)")
        volume_to_rotate = volume
        # 标准 Z 轴旋转 (在 Y-X 平面)
        rot_axes = (1, 2)
        proj_axis = 1 # 沿着 Y 轴投影，或者是 2
        
    # 3. 准备颜色
    cmap = plt.cm.get_cmap('tab10', clusters + 1)
    cmap.set_under('black')
    
    # 4. 生成帧
    print("正在渲染帧...")
    frames = []
    # 36 帧，每帧转 10 度
    angles = np.linspace(0, 360, 37)[:-1] 
    
    for angle in tqdm(angles):
        # reshape=False 保持画幅稳定，适合做 GIF
        # mode='constant', cval=0 填充黑色背景
        rotated_vol = rotate(volume_to_rotate, angle, axes=rot_axes, reshape=False, order=0, mode='constant', cval=0)
        
        # 最大密度投影
        projection = np.max(rotated_vol, axis=proj_axis)
        frames.append(projection)
        
        # 简单清理
        del rotated_vol

    # 5. 合成 GIF (自动调整画布比例)
    print(f"正在保存 GIF: {output_gif_path}")
    
    # 获取第一帧的尺寸
    h, w = frames[0].shape
    
    # 设定基准尺寸 (英寸)
    base_size = 6 
    
    # 智能计算 Figure 大小，防止图像变形或留白太多
    if w > h: 
        # 如果是宽图 (Horizontal模式)，宽度定死，高度自适应
        fig_w = base_size
        fig_h = base_size * (h / w)
    else:
        # 如果是高图 (Vertical模式)，高度定死，宽度自适应
        fig_h = base_size
        fig_w = base_size * (w / h)
        
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # 去除周围留白
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('off')
    
    first_frame = frames[0]
    masked_data = np.ma.masked_where(first_frame == 0, first_frame)
    im = ax.imshow(masked_data, cmap=cmap, vmin=1, vmax=clusters, animated=True, aspect='equal')
    
    # 添加色标 (可选，为了美观可以注释掉)
    # cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.0)
    
    # 标题放到底部或者不放
    # ax.text(0.5, 0.05, f"Mode: {mode}", transform=ax.transAxes, ha='center', color='white', fontsize=10)

    def update(frame_idx):
        data = frames[frame_idx]
        masked = np.ma.masked_where(data == 0, data)
        im.set_array(masked)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=True)
    ani.save(output_gif_path, writer='pillow', fps=fps)
    plt.close()
    
    # 最终清理
    del frames, volume, volume_to_rotate
    gc.collect()
    print("GIF 生成完成！")

# ================= 使用示例 =================

input_tif = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16_Clustered_3D_Stack.tif"

# 1. 横着滚 (推荐用于堆叠后的长血管)
generate_flexible_rotation_gif(input_tif, "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/View_Horizontal.gif", mode='vertical', downscale_factor=0.2)

# 2. 竖着转 (经典视角)
# generate_flexible_rotation_gif(input_tif, "View_Vertical.gif", mode='vertical', downscale_factor=0.2)

#%%
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import tifffile

# 假设你已经定义了 build_2d_vascular_graph 函数
# from your_script import build_2d_vascular_graph 

def visualize_graph_overlay(mask_path, signal_path, output_plot_path="Graph_Overlay.png", zoom_crop=None):
    """
    在原始 CD31 图像上叠加 GNN 的节点和边。
    
    参数:
        mask_path: Mask 文件路径 (用于构建图)
        signal_path: CD31 原始信号路径 (作为背景底图)
        output_plot_path: 保存路径
        zoom_crop: (可选) 元组 (y_start, y_end, x_start, x_end) 用于放大局部区域，
                   因为整张大图可能看不清细节。
    """
    print(f"正在准备叠加可视化: {mask_path}")
    
    # 1. 读取图像
    mask = tifffile.imread(mask_path)
    signal = tifffile.imread(signal_path)
    h, w = signal.shape
    
    # 2. 构建图 (获取节点和边)
    # 这一步会重新运行骨架化和采样
    graph = build_2d_vascular_graph_on_wall(mask_path, signal_path, sample_step=3)
    
    if graph is None:
        print("无法构建图 (可能是空掩码)")
        return

    # 3. 反归一化坐标
    # graph.x 的前两列是归一化的 (Y, X)
    norm_y = graph.x[:, 0].numpy()
    norm_x = graph.x[:, 1].numpy()
    
    # 还原回像素坐标
    pixel_y = norm_y * h
    pixel_x = norm_x * w
    
    # 4. 准备绘图
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # --- A. 绘制背景 (CD31 原始信号) ---
    # 使用 'gray' 或 'viridis' 色谱
    # vmin/vmax 用于增强对比度，过滤掉过暗的背景噪音
    vmin, vmax = np.percentile(signal, [5, 99]) 
    ax.imshow(signal, cmap='gray', vmin=vmin, vmax=vmax, alpha=0.9)
    
    # --- B. 绘制边 (Edges) ---
    print("   -> 正在绘制连接边...")
    edge_index = graph.edge_index.numpy()
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    # 构建线段列表: [[(x1, y1), (x2, y2)], ...]
    segments = []
    for s, d in zip(src_nodes, dst_nodes):
        p1 = (pixel_x[s], pixel_y[s])
        p2 = (pixel_x[d], pixel_y[d])
        segments.append([p1, p2])
        
    # 使用 LineCollection 高效绘制
    lc = LineCollection(segments, colors='yellow', linewidths=0.5, alpha=0.4)
    ax.add_collection(lc)
    
    # --- C. 绘制节点 (Nodes) ---
    print("   -> 正在绘制节点...")
    # 我们可以用节点特征来着色，比如 CD31 强度 (graph.x 第4列, index 3)
    # 或者直接用单一颜色 (如红色)
    node_signals = graph.x[:, 3].numpy()
    
    scatter = ax.scatter(pixel_x, pixel_y, c=node_signals, cmap='Reds', s=3, zorder=2, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label='Node Feature: CD31 Intensity', fraction=0.046, pad=0.04)
    
    # 5. 设置显示范围 (Zoom)
    if zoom_crop:
        ys, ye, xs, xe = zoom_crop
        ax.set_ylim(ye, ys) # 注意：imshow 的 Y 轴通常是反的，但在 set_ylim 中大的值在下
        ax.set_xlim(xs, xe)
        plt.title(f"GNN Graph Overlay (Crop: {zoom_crop})")
    else:
        plt.title("Vascular GNN Graph Overlay (Full View)")

    # 6. 保存
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    print(f"叠加图已保存: {output_plot_path}")
    # plt.show()
    plt.close()

# ================= 运行示例 =================

m_path = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16_mask/C16-1-44-mask0020.tif"
s_path = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16/C16-1-44-signal0020.tif"

# 1. 生成全景图 (可能点会比较小)
#visualize_graph_overlay(m_path, s_path, "Overlay_Full.png")

# 2. 生成局部放大图 (推荐！能看清细节)
# 假设图片很大，我们只看中间 500x500 的区域
# 格式: (y_start, y_end, x_start, x_end)

visualize_graph_overlay(m_path, s_path, "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/Overlay_Zoom.png")
