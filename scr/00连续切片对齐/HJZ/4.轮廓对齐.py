# cd /public2/chengrm/3D_TME/code/MetaVision3D/
# conda activate metavision3d

from modules.setup import *
from modules.utils import *
from modules.MetaNorm3D import *
from modules.MetaAlign3D import *
from modules.MetaImpute3D import *
from modules.MetaInterp3D import *
from modules.MetaAtlas3D import *
from modules.visualize import *
from modules.evaluate import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.cm as cm
import tifffile
from scipy.ndimage import gaussian_filter
import cv2

def display_animation(matrix, grayscale=False, cmap=None, save_path=None, fps=3):
    """
    将3D矩阵逐帧显示为动画，并可保存为 gif/mp4

    Args:
        matrix: 3D numpy 数组 (frames, H, W)
        grayscale: 是否灰度显示
        cmap: 使用的 colormap
        save_path: 输出路径 (以 .gif 或 .mp4 结尾)
        fps: 帧率
    """
    plt.rcParams["animation.html"] = "jshtml"
    frames = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(4, 4))
    vmax = np.percentile(matrix[matrix != 0], 99)

    im = ax.imshow(matrix[0], origin='upper', vmax=vmax)
    if grayscale:
        plt.gray()
    elif cmap is not None:
        plt.set_cmap(cmap)
    ax.set_axis_off()

    def animate(i):
        im.set_array(matrix[i])
        ax.set_title(f"Slice {i+1}")
        return [im]

    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True)

    # 保存
    if save_path is not None:
        if save_path.endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=fps)
        elif save_path.endswith(".mp4"):
            ani.save(save_path, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("save_path 必须以 .gif 或 .mp4 结尾")

    return ani

df_all = pd.read_csv("/public2/chengrm/3D_TME/3D-DESI/1.continued_results/all_df.30.csv")
group = 'wt'
# 翻转
df_all = flip_axis(df_all,flipud=True)

cols = ["tissue_id", "x", "y"] + [c for c in df_all.columns if c not in ["tissue_id", "x", "y"]]
df_all = df_all[cols]
features = df_all.columns[4:17]

def robust_minmax(x, lower=0.01, upper=0.99):
    q_low, q_high = np.quantile(x, [lower, upper])
    return (x - q_low) / (q_high - q_low)

df_all[features] = df_all.groupby("z")[features].transform(robust_minmax)

# for feature_xqq in features:

feature_xqq = 'mz_282'
df_all, deleted_compound = delete_low_prevalence_compound(df_all, 0.1, first_feature=feature_xqq) # LPA.18.1 is the first compound
deleted_compound
preview_matrix = create_compound_matrix(df_all, compound=feature_xqq,reverse=True)
out_dir = f"/public2/chengrm/3D_TME/3D-DESI/4.tmp_mask/{feature_xqq}/"
os.makedirs(out_dir, exist_ok=True)

for i in range(preview_matrix.shape[0]):
    img = preview_matrix[i]

    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_norm)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    out_path = os.path.join(out_dir, f"mask_{i:02d}.{feature_xqq}.png")
    cv2.imwrite(out_path, mask)




# ###############################################
# import cv2
# import numpy as np
# import os
# from glob import glob


# img_dir = f"/public2/chengrm/3D_TME/3D-DESI/4.tmp_mask/{feature_xqq}"
# out_dir = f"/public2/chengrm/3D_TME/3D-DESI/4.tmp_mask/{feature_xqq}/aligned/"
# os.makedirs(out_dir, exist_ok=True)

# mask_files = sorted(glob(os.path.join(img_dir, "*.png")))

# ref_img = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
# ref_img = cv2.threshold(ref_img, 127, 255, cv2.THRESH_BINARY)[1] 

# cv2.imwrite(os.path.join(out_dir, os.path.basename(mask_files[0])), ref_img)

# all_mask = [ref_img.copy()] 

# for i in range(1, len(mask_files)):
#     img = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
#     img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

#     ref_float = ref_img.astype(np.float32) / 255.0
#     img_float = img.astype(np.float32) / 255.0

#     warp_matrix = np.eye(2, 3, dtype=np.float32)

#     number_of_iterations = 5000
#     termination_eps = 1e-6
#     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
#                 number_of_iterations, termination_eps)

#     # ECC算法做刚性对齐

#     cc, warp_matrix = cv2.findTransformECC(ref_float, img_float, warp_matrix,
#                                                cv2.MOTION_EUCLIDEAN, criteria)

#     aligned = cv2.warpAffine(img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]),
#                              flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

#     out_path = os.path.join(out_dir, os.path.basename(mask_files[i]))
#     cv2.imwrite(out_path, aligned)

#     ref_img = aligned.copy()
#     all_mask.append(aligned)

# all_df = np.stack(all_mask, axis=0)
# tifffile.imwrite(f"{out_dir}all_mask.{feature_xqq}.tif", all_df.astype("float32"))




####################################################

import cv2
import numpy as np
import os
import tifffile

# 主特征
feature_xqq = "mz_282"
# 迁移的特征
transfer_features = ["ADA", "AA", "DGLA", "LA"]
out_dir = f"/public2/chengrm/3D_TME/3D-DESI/4.tmp_mask/{feature_xqq}/transfer/"
os.makedirs(out_dir, exist_ok=True)

# --------------------------
# Step 1: 在 mz_282 上学习变换
# --------------------------
warp_matrices = []

ref_img = None
for i in range(preview_matrix.shape[0]):
    img = preview_matrix[i]
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if ref_img is None:
        ref_img = binary.copy()
        warp_matrices.append(np.eye(2, 3, dtype=np.float32))  # 第一层不变
    else:
        ref_float = ref_img.astype(np.float32) / 255.0
        img_float = binary.astype(np.float32) / 255.0
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
        try:
            cc, warp_matrix = cv2.findTransformECC(ref_float, img_float, warp_matrix,
                                                   cv2.MOTION_EUCLIDEAN, criteria)
        except cv2.error:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_matrices.append(warp_matrix)
        ref_img = cv2.warpAffine(binary, warp_matrix, (ref_img.shape[1], ref_img.shape[0]),
                                 flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        
np.save(f"{out_dir}warp_matrices.npy", warp_matrices)
# ------------------------------------------------------------------------------
# Step 2: 把学习到的变换迁移到其他特征
# ------------------------------------------------------------------------------
for feat in transfer_features:
    print(f"处理特征 {feat} ...")

    # 构建该特征的矩阵
    matrix = create_compound_matrix(df_all, compound=feat, reverse=True)

    aligned_stack = []
    for i in range(matrix.shape[0]):
        img = matrix[i]
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        warp_matrix = warp_matrices[i]
        aligned = cv2.warpAffine(img_norm, warp_matrix, (matrix.shape[2], matrix.shape[1]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        aligned_stack.append(aligned)

    aligned_stack = np.stack(aligned_stack, axis=0)
    tifffile.imwrite(f"{out_dir}/aligned_{feat}.tif", aligned_stack.astype("float32"))

# --------------------------
# Step 3: 高斯平滑
# --------------------------

aligned_ADA = tifffile.imread(f"{out_dir}/aligned_ADA.tif")
aligned_AA = tifffile.imread(f"{out_dir}/aligned_AA.tif")
aligned_DGLA = tifffile.imread(f"{out_dir}/aligned_DGLA.tif")
aligned_LA = tifffile.imread(f"{out_dir}/aligned_LA.tif")

aligned_ga_ADA = gaussian_filter(aligned_ADA, sigma=2)
aligned_ga_AA = gaussian_filter(aligned_AA, sigma=2)
aligned_ga_DGLA = gaussian_filter(aligned_DGLA, sigma=2)
aligned_ga_LA = gaussian_filter(aligned_LA, sigma=2)

tifffile.imwrite(f"{out_dir}/aligned_ADA.ga.tif",aligned_ga_ADA)
tifffile.imwrite(f"{out_dir}/aligned_AA.ga.tif",aligned_ga_AA)
tifffile.imwrite(f"{out_dir}/aligned_DGLA.ga.tif",aligned_ga_DGLA)
tifffile.imwrite(f"{out_dir}/aligned_LA.ga.tif",aligned_ga_LA)


# --------------------------
# Step 4: 画组织轮廓
# --------------------------

test = aligned_ga_ADA.copy()
test[test != 0] = 1  
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
processed = []
for i in range(test.shape[0]): 
    img = test[i].astype(np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    processed.append(img_erode)
processed = np.stack(processed, axis=0)

tifffile.imwrite(
    "/public2/chengrm/3D_TME/3D-DESI/4.tmp_mask/mz_282/transfer/tissue.morph.tif",
    processed.astype(np.float32)
)
