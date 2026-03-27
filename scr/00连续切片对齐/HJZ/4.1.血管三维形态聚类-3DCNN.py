#%% 3D特征提取并聚类分析血管结构
import os
import glob
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

# ================= 配置参数 =================
PATCH_SIZE = 16
LATENT_DIM = 64
BATCH_SIZE = 128
EPOCHS     = 10       
CLUSTERS   = 10       
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= 1. 模型定义 (Advanced Version) =================
class AdvancedAutoencoder3D(nn.Module):
    def __init__(self):
        super(AdvancedAutoencoder3D, self).__init__()
        
        # --- Encoder ---
        self.encoder = nn.Sequential(
            # L1: 16 -> 8
            nn.Conv3d(1, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            # L2: 8 -> 4
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),
            # L3: 4 -> 2
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d(2),
            # Flatten
            nn.Flatten(),
            nn.Linear(128 * 2 * 2 * 2, LATENT_DIM),
            nn.ReLU()
        )
        
        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128 * 2 * 2 * 2),
            nn.Unflatten(1, (128, 2, 2, 2)),
            # L3 Rev
            nn.ConvTranspose3d(128, 64, 2, stride=2), nn.BatchNorm3d(64), nn.ReLU(),
            # L2 Rev
            nn.ConvTranspose3d(64, 32, 2, stride=2), nn.BatchNorm3d(32), nn.ReLU(),
            # L1 Rev
            nn.ConvTranspose3d(32, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def extract_features(self, x):
        return self.encoder(x)

# ================= 2. 数据集定义 =================
class Vessel3D_Dataset(Dataset):
    def __init__(self, volume, coords):
        self.volume = volume
        self.coords = coords
        self.half_size = PATCH_SIZE // 2
        
    def __len__(self):
        return len(self.coords)
        
    def __getitem__(self, idx):
        z, y, x = self.coords[idx]
        cube = self.volume[
            z - self.half_size : z + self.half_size,
            y - self.half_size : y + self.half_size,
            x - self.half_size : x + self.half_size
        ]
        return torch.from_numpy(cube).float().unsqueeze(0)

# ================= 3. 主处理函数 =================
def process_folder_and_cluster(input_dir, output_tif_path, output_npy_path):
    print(f"开始处理文件夹: {input_dir}")
    
    # --- Step 1: 读取连续 TIF 序列 ---
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    
    if len(tif_files) == 0:
        raise ValueError(f"文件夹为空或未找到 .tif 文件: {input_dir}")
        
    print(f"   -> 找到 {len(tif_files)} 张切片，正在加载到内存...")
    stack_list = [tifffile.imread(f) for f in tif_files]
    stack = np.array(stack_list)
    
    if stack.ndim == 4: 
        print(f"   检测到多余维度 {stack.shape}，正在压缩...")
        stack = np.squeeze(stack)
    
    print(f"   -> 3D 数据构建完成，形状: {stack.shape}")

    # --- Step 2: 预处理 ---
    max_val = stack.max()
    print(f"   -> 数据最大值: {max_val} (用于归一化)")
    
    volume = stack.astype(np.float32) / (65535.0 if max_val > 255 else 255.0)
    pad = PATCH_SIZE // 2
    volume_padded = np.pad(volume, ((pad,pad), (pad,pad), (pad,pad)), mode='constant')

    # --- Step 3: 提取血管坐标 ---
    print("正在提取血管采样点...")
    vessel_mask = volume > 0.01 
    z_locs, y_locs, x_locs = np.where(vessel_mask)
    
    if len(z_locs) == 0:
        raise ValueError("未检测到任何血管信号，请检查图像是否全黑。")

    step = 5 
    coords_original = list(zip(z_locs[::step], y_locs[::step], x_locs[::step]))
    coords_padded = [(z+pad, y+pad, x+pad) for z, y, x in coords_original]
    
    print(f"   -> 提取到 {len(coords_original)} 个采样点 (Step={step})")
    
    dataset = Vessel3D_Dataset(volume_padded, coords_padded)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Step 4: 训练模型 ---
    print(f"开始训练 Advanced Autoencoder ({EPOCHS} Epochs)...")
    model = AdvancedAutoencoder3D().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in pbar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            _, reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    # --- Step 5: 提取特征 ---
    print("正在提取全量特征...")
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_features = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Extracting"):
            batch = batch.to(DEVICE)
            feats = model.extract_features(batch)
            all_features.append(feats.cpu().numpy())
    
    all_features = np.vstack(all_features)
    
    # 关键修改：使用 savez_compressed 解决 4GB 限制 
    output_npz_path = output_npy_path.replace('.npy', '.npz')
    print(f"   -> 保存特征到(压缩格式): {output_npz_path}")
    
    # 注意：此处不再使用 np.save，而是 savez_compressed
    np.savez_compressed(output_npz_path, 
                        features=all_features, 
                        coords_zyx=coords_original, 
                        shape=stack.shape)

    # --- Step 6: 聚类 ---
    print(f"正在进行 K-Means 聚类 (K={CLUSTERS})...")
    
    # 选项 A: 标准 KMeans (如果内存足够大 > 64GB)
    kmeans = KMeans(n_clusters=CLUSTERS, random_state=42)
    
    # 选项 B: MiniBatchKMeans (省内存模式，如果下面那行报错 OOM，请解开这行注释并注释掉上面那行)
    # kmeans = MiniBatchKMeans(n_clusters=CLUSTERS, batch_size=10000, random_state=42)
    
    labels = kmeans.fit_predict(all_features)
    
    # --- Step 7: 生成 3D 结果图 ---
    print("正在重建 3D 标签图...")
    result_volume = np.zeros(stack.shape, dtype=np.uint8)
    
    for (z, y, x), label in zip(coords_original, labels):
        result_volume[z, y, x] = label + 1
        
    print(f"   -> 保存 3D 堆叠图到: {output_tif_path}")
    tifffile.imwrite(output_tif_path, result_volume)
    
    print("\n所有任务完成！")
    print(f"1. 特征文件(NPZ): {output_npz_path}")
    print(f"2. 聚类结果(TIF): {output_tif_path}")

# ================= 运行示例 =================

input_folder = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16_mask"

output_tif = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16-mask-3D.tif"
# 这里即使你写 .npy，代码也会自动改成 .npz 保存
output_npy = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16-mask-3D.npy"

process_folder_and_cluster(input_folder, output_tif, output_npy)





#%% 三面俯视图绘制
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

def generate_static_3d_ortho_view(tif_path, output_plot_path, clusters=10):
    """
    生成大尺度 3D 标签数据的三视图彩色投影 (Orthogonal MIP)。
    使用内存映射 (memmap) 技术处理大文件。
    """
    print(f"正在准备可视化: {os.path.basename(tif_path)}")
    
    # --- 1. 使用内存映射打开大文件 (关键步骤) ---
    # 这不会把整个文件读入内存，而是建立一个映射
    try:
        volume_mmap = tifffile.memmap(tif_path, mode='r')
        print(f"   -> 数据形状 (Z, Y, X): {volume_mmap.shape}")
        print("   -> 已建立内存映射，开始计算投影...")
    except Exception as e:
        print(f"内存映射失败: {e}")
        return

    # --- 2. 准备离散色谱 ---
    # 使用 tab10 或 Set1 这种鲜明的离散色盘
    # 0 (背景) 将被映射为黑色
    base_cmap = plt.cm.get_cmap('tab10', clusters + 1)
    color_list = base_cmap(np.linspace(0, 1, clusters + 1))
    color_list[0] = [0, 0, 0, 1] # 将背景 (Label 0) 设为全黑
    custom_cmap = mcolors.ListedColormap(color_list)
    
    # 定义归一化器，确保 label 整数正确映射到颜色上
    norm = mcolors.BoundaryNorm(np.arange(-0.5, clusters + 1.5, 1), custom_cmap.N)


    # --- 3. 计算三个方向的投影 (MIP) ---
    # np.max 在内存映射数组上可能会比较慢，添加进度条提示
    
    print("   -> [1/3] 计算顶部视图 (XY Projection, Z-axis MIP)...")
    # axis=0 沿着 Z 轴压平
    proj_xy = np.max(volume_mmap, axis=0)

    print("   -> [2/3] 计算前视图 (XZ Projection, Y-axis MIP)...")
    # axis=1 沿着 Y 轴压平。对于大数组，这可能需要一点时间
    proj_xz = np.max(volume_mmap, axis=1)
    
    print("   -> [3/3] 计算侧视图 (YZ Projection, X-axis MIP)...")
    # axis=2 沿着 X 轴压平
    proj_yz = np.max(volume_mmap, axis=2)
    
    print("   -> 投影计算完成，开始绘图...")


    # --- 4. 绘制三视图 ---
    fig = plt.figure(figsize=(15, 10))
    # 使用 GridSpec 布局：顶部视图占大头，下面放前视图和侧视图
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])

    # 辅助绘图函数
    def plot_projection(ax, data, title, xlabel, ylabel, aspect='auto'):
        # 使用 masked_array 隐藏背景0值，但这在 MIP 中不是必须的，因为我们已经把0设为黑色了
        im = ax.imshow(data, cmap=custom_cmap, norm=norm, origin='lower', aspect=aspect)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return im

    # A. 顶部视图 (XY) - 左上角
    ax_xy = fig.add_subplot(gs[0, 0])
    # aspect='equal' 保证 XY 平面比例尺一致
    im = plot_projection(ax_xy, proj_xy, "Top View (XY Plane)", "X Axis", "Y Axis", aspect='equal')
    
    # B. 侧视图 (YZ) - 右上角 (与 Top View 的 Y 轴对齐)
    ax_yz = fig.add_subplot(gs[0, 1], sharey=ax_xy)
    # 注意：YZ视图需要转置一下才符合直观视觉
    plot_projection(ax_yz, proj_yz.T, "Side View (YZ Plane)", "Z Axis", "Y Axis (Shared)")
    plt.setp(ax_yz.get_yticklabels(), visible=False) # 隐藏重复的 Y 轴标签

    # C. 前视图 (XZ) - 左下角 (与 Top View 的 X 轴对齐)
    ax_xz = fig.add_subplot(gs[1, 0], sharex=ax_xy)
    plot_projection(ax_xz, proj_xz, "Front View (XZ Plane)", "X Axis (Shared)", "Z Axis")
    
    # D. 色标栏 (Colorbar) - 右下角区域
    ax_cb = fig.add_subplot(gs[1, 1])
    ax_cb.axis('off') # 不显示坐标轴
    cbar = plt.colorbar(im, ax=[ax_xy, ax_yz, ax_xz], fraction=0.02, pad=0.04)
    cbar.set_label('Cluster ID', fontsize=12)
    # 设置色标刻度只显示 1 到 10
    cbar.set_ticks(np.arange(1, clusters + 1))

    plt.suptitle(f"Static 3D Orthogonal View: {os.path.basename(tif_path)}", fontsize=16, y=0.98)
    
    # 保存高分辨率大图
    plt.savefig(output_plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"静止 3D 三视图已保存: {output_plot_path}")
    plt.show()
    
    # 清理内存映射引用 (虽然 Python 会自动回收，但显式删除是个好习惯)
    del volume_mmap
    del proj_xy, proj_xz, proj_yz

# ================= 运行示例 =================
input_tif = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C15-3D.tif"
output_plot = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C15-3D_ortho_view.png"

# 这里的 clusters 要和你之前 K-Means 设置的 K 值一致
generate_static_3d_ortho_view(input_tif, output_plot, clusters=10)


 




#%% GIF 图绘制
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import rotate, zoom
import matplotlib.colors as mcolors
from tqdm import tqdm
import gc  # 引入垃圾回收，这对全分辨率处理至关重要

def generate_flexible_3d_gif(tif_path, output_gif_path, downscale_factor=None, tilt_angle=30, clusters=10):
    """
    生成 3D 聚类结果的倾斜旋转 GIF，支持全分辨率或降采样模式。
    
    参数:
        tif_path: 输入 TIF 路径
        output_gif_path: 输出 GIF 路径
        downscale_factor: 
            - 输入小数 (例如 0.2): 进行降采样 (推荐用于预览)
            - 输入 None 或 1.0: 关闭降采样 (使用原始分辨率，内存需求极大)
        tilt_angle: 俯视倾斜角度 (推荐 30 度)，增加立体感
    """
    print(f"正在读取数据: {tif_path}")
    
    # 使用 memmap 读取，避免一开始就炸内存
    volume_mmap = tifffile.memmap(tif_path, mode='r')
    original_shape = volume_mmap.shape
    print(f"   原始数据形状: {original_shape}")
    
    # --- Step 1: 处理分辨率 (降采样 vs 全分辨率) ---
    if downscale_factor is None or downscale_factor == 1.0:
        print("[模式: 全分辨率] 正在加载完整数据... 请确保服务器有 >64GB 内存。")
        volume = np.array(volume_mmap) # 只有这里才真正读入内存
        scale_text = "Full-Res"
    else:
        print(f"[模式: 降采样] 正在将 XY 轴缩小至 {downscale_factor}倍...")
        # 为了节省内存，我们可以先切片读取再降采样，或者直接读入后降采样
        # 这里为了代码简单，先读入 (假设内存够读入原始数据)
        volume_raw = np.array(volume_mmap)
        # Z轴保持 1.0 (不压缩层数)，XY轴压缩
        volume = zoom(volume_raw, (1.0, downscale_factor, downscale_factor), order=0)
        del volume_raw # 立即释放原图
        gc.collect()
        scale_text = f"Scale={downscale_factor}"
        print(f"   降采样后形状: {volume.shape}")
        
    del volume_mmap # 释放 mmap 句柄

    # --- Step 2: 预先倾斜 (Tilt) ---
    # 这是产生“立体感”的关键步骤
    if tilt_angle > 0:
        print(f"正在应用俯视倾斜 ({tilt_angle}度)...")
        # reshape=True 允许体积变大以容纳旋转后的角
        volume = rotate(volume, tilt_angle, axes=(0, 1), reshape=True, order=0, cval=0)
        gc.collect()

    # 准备颜色
    cmap = plt.cm.get_cmap('tab10', clusters + 1)
    cmap.set_under('black')

    # --- Step 3: 生成旋转帧 (MIP) ---
    print(f"正在生成旋转帧...")
    frames = []
    
    # 减少帧数以避免全分辨率下耗时过长 (36帧 = 每10度一帧)
    angles = np.linspace(0, 360, 37)[:-1] 
    
    # 目标投影轴: 现在的 X 轴 (axis=2) 是视线方向
    # 也可以选 Y 轴 (axis=1)，取决于你觉得哪个角度好看
    target_axis = 3 
    
    for i, angle in enumerate(tqdm(angles)):
        # A. 旋转 (Spin)
        # 绕着现在的 Z 轴 (视觉垂直轴) 旋转
        # reshape=False 保持视窗大小一致，防止画面抖动
        rotated_vol = rotate(volume, angle, axes=(1, 2), reshape=False, order=0, cval=0)
        
        # B. 最大密度投影 (MIP)
        projection = np.max(rotated_vol, axis=target_axis)
        
        # C. 存入列表 (copy是必须的)
        frames.append(projection.copy())
        
        # D. 内存清理 (全分辨率模式下的救命稻草)
        del rotated_vol, projection
        gc.collect()

    # 清理巨大的 3D 数组
    del volume
    gc.collect()

    # --- Step 4: 合成 GIF ---
    print(f"正在合成 GIF: {output_gif_path}")
    
    # 动态计算图片尺寸
    h, w = frames[0].shape
    # 控制一下显示尺寸，不要太大
    fig_h = 10
    fig_w = fig_h * (w / h)
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.tight_layout()
    ax.axis('off')
    
    # 初始化
    first_frame = frames[0]
    masked_data = np.ma.masked_where(first_frame == 0, first_frame)
    im = ax.imshow(masked_data, cmap=cmap, vmin=1, vmax=clusters, animated=True)
    
    # 色标
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Cluster ID')
    ax.set_title(f"3D Stereoscopic View ({scale_text})\nTilt: {tilt_angle} deg")

    def update(frame_idx):
        data = frames[frame_idx]
        masked = np.ma.masked_where(data == 0, data)
        im.set_array(masked)
        return [im]

    # 保存
    # fps=12 比较流畅
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    ani.save(output_gif_path, writer='pillow', fps=12)
    plt.close()
    print("动图生成完成！")

# ================= 运行示例 =================
input_tif = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16-mask-3D.tif"

output_gif_preview = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16-mask-3D.gif"
generate_flexible_3d_gif(input_tif, output_gif_preview, downscale_factor=None, tilt_angle=45)


#%%
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import rotate, zoom
import matplotlib.colors as mcolors
from tqdm import tqdm

def generate_3d_rotating_gif(tif_path, output_gif_path, downscale_factor=0.2, clusters=10):
    """
    Generates a rotating 3D GIF from a 3D TIF image stack.
    This code creates a 360-degree rotation around the vertical axis.
    """
    print(f"Reading data: {tif_path}")
    try:
        volume = tifffile.imread(tif_path)
    except FileNotFoundError:
        print(f"Error: File not found at {tif_path}")
        return
    
    # 1. Downsampling (Critical step for large datasets)
    # Downsampling significantly reduces memory usage and computation time for rotation.
    # For a large volume (e.g., 4000x6000x90), downsampling is essential.
    scale_z = 1.0  # Keep Z-axis resolution or adjust slightly
    scale_yx = downscale_factor # Downscale XY plane
    
    print(f"   Original shape: {volume.shape}")
    print(f"   Downsampling (Factor={downscale_factor})... This might take a minute...")
    
    # Use order=0 (nearest-neighbor interpolation) to preserve integer cluster labels.
    volume_small = zoom(volume, (scale_z, scale_yx, scale_yx), order=0)
    print(f"   Downsampled shape: {volume_small.shape}")

    # 2. Prepare Color Map
    # Use a discrete color map like 'tab10' for distinct cluster visualization.
    # Set the background (label 0) to black.
    cmap = plt.cm.get_cmap('tab10', clusters + 1)
    cmap.set_under('black')
    
    # 3. Generate Rotating Frames
    print("Generating rotating frames (36 frames total)...")
    frames = []
    
    # Rotation angles: 0 to 360 degrees, in 10-degree steps
    angles = np.linspace(0, 360, 37)[:-1] 
    
    for angle in tqdm(angles):
        # A. Rotate Volume (Rotate around the vertical axis)
        # axes=(1, 2) rotates within the YX plane, effectively around the Z-axis (vertical).
        # reshape=False keeps the frame size constant.
        rotated_vol = rotate(volume_small, angle, axes=(1, 2), reshape=False, order=0, mode='constant', cval=0)
        
        # B. Maximum Intensity Projection (MIP) -> Create 2D Image
        # Project along the Y-axis (axis=1) to create a side-view rotation effect.
        # Alternatively, project along the X-axis (axis=2) for a different perspective.
        projection = np.max(rotated_vol, axis=1)
        
        # C. Store the frame
        frames.append(projection)

    # 4. Create and Save GIF
    print(f"Saving GIF to: {output_gif_path}")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.tight_layout()
    ax.axis('off') # Hide axes
    
    # Initialize with the first frame
    # Use masked_where to make the background (value 0) transparent, showing the black background color set in cmap.
    first_frame = frames[0]
    masked_data = np.ma.masked_where(first_frame == 0, first_frame)
    im = ax.imshow(masked_data, cmap=cmap, vmin=1, vmax=clusters, animated=True)
    
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Cluster ID')
    ax.set_title(f"3D Rotating Vessel\n(Downsampled {downscale_factor}x)")

    def update(frame_idx):
        data = frames[frame_idx]
        masked = np.ma.masked_where(data == 0, data)
        im.set_array(masked)
        return [im]

    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=150, blit=True)
    
    # Save using the Pillow writer
    ani.save(output_gif_path, writer='pillow', fps=10)
    plt.close()
    print("3D rotating GIF generated successfully!")

# ================= Run the function =================
# Replace with your actual file paths
input_tif = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16-mask-3D.tif"
output_gif = "/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/C16-mask-3D.gif"

# Adjust downscale_factor based on your available memory.
# 0.1 shrinks XY dimensions to 1/10th, fastest.
# 0.2 shrinks to 1/5th, better detail.
# Make sure the input TIF file exists.
generate_3d_rotating_gif(input_tif, output_gif, downscale_factor=0.2, clusters=10)
