#%% 导入库
from valis import registration, valtils, micro_rigid_registrar, preprocessing, slide_io
from skimage.util import img_as_ubyte
from skimage import filters, io, exposure
from roifile import roiread
from pathlib import Path
import numpy as np
import os
import re
import imageio.v2 as imageio
import pyvips
from roifile import roiread
import tifffile

#%% 获取 ROI 图像
# sample_name = 'CMH-00003-16139-16967'
# 
# rigid_tif_dir = f'/public3/Xinyu/3D_tissue/IF/01.mIHC_rigid_registration/MIF_CMH/registered_slides'
# save_roi_dir = f'/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/{sample_name}/Micro-registration/{sample_name}_DAPI'
# roi_file = f'/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/{sample_name}.roi'  # FIJI 保存的 ROI
# os.makedirs(save_roi_dir, exist_ok=True)
# 
# # ======================
# # 读取 ROI
# # ======================
# roi_obj = roiread(roi_file)  # 单个 ROI 返回 ImagejRoi
# rois = [roi_obj] if not isinstance(roi_obj, list) else roi_obj
# 
# # ======================
# # 遍历 ROI 裁剪所有切片
# # ======================
# # 获取 ROI 的坐标（假设只取第一个 ROI）
# roi = rois[0]
# minr = roi.top - 20
# maxr = roi.bottom 
# minc = roi.left - 20
# maxc = roi.right + 20
# 
# # 遍历 C3_clean 文件夹下所有切片
# slice_files = sorted([f for f in os.listdir(rigid_tif_dir) if f.endswith('.tif')])
# 
# for slice_fname in slice_files:
#     print(slice_fname)
# 
#     slice_path = os.path.join(rigid_tif_dir, slice_fname)
#     
#     # 读取切片
#     img = io.imread(slice_path)
#     
#     # 裁剪 ROI 区域
#     roi_img = img[minr:maxr, minc:maxc]
#     
#     # 保存裁剪结果
#     save_path = os.path.join(save_roi_dir, slice_fname)
#     io.imsave(save_path, img_as_ubyte(roi_img))
#     print(f"保存 {save_path}")
# 
# print("所有切片的 ROI 已裁剪完成")


#%% OME-TIFF
'''
功能：根据原始的大图，保存ROI对应的图像，保存为PNG或者ome.tiff,进行对齐

'''
# --- 1. 图像增强与 PNG 保存函数 ---
def enhance_channel(image_data):
    """
    对单通道图像进行自适应对比度增强 (1%-99% 拉伸)
    """
    if image_data.size == 0: return image_data 
    
    img_float = image_data.astype(np.float32)
    p1, p99 = np.percentile(img_float, (1, 99))
    
    if p99 > p1:
        img_rescaled = exposure.rescale_intensity(img_float, in_range=(p1, p99), out_range=(0, 255))
    else:
        img_rescaled = img_float 
        
    return img_rescaled

def save_mixed_png(cropped_arr, out_path, channel_indices, mode="ALL"):
    """
    将通道映射为指定的 BGR 颜色:
    DAPI -> Blue (B)
    CD31 -> Red (R)
    LYVE1 -> Green (G)
    """
    existing_channels = cropped_arr.shape[0]

    # 检查索引是否越界
    for ch_name in ["DAPI", "CD31", "LYVE1"]:
        idx = channel_indices.get(ch_name)
        if idx is None or idx >= existing_channels:
            print(f"  -> Skip: Channel '{ch_name}' (Index {idx}) is missing.")
            return

    # 1. 提取并增强各个通道
    dapi_data  = enhance_channel(cropped_arr[channel_indices["DAPI"]])
    cd31_data  = enhance_channel(cropped_arr[channel_indices["CD31"]])
    lyve1_data = enhance_channel(cropped_arr[channel_indices["LYVE1"]])

    # 2. 映射到 RGB 空间
    # 注意：scikit-image/imageio 保存的是 RGB 格式
    # R = CD31, G = LYVE1, B = DAPI
    R = cd31_data
    G = lyve1_data
    B = dapi_data

    # 3. 合并通道并转为 uint8
    # stack 顺序为 (R, G, B)
    rgb = np.stack([R, G, B], axis=-1)
    rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
    
    imageio.imwrite(out_path, rgb_uint8)
    print(f"  -> Saved RGB PNG (R:CD31, G:LYVE1, B:DAPI): {out_path}")

# --- 2. OME-TIFF 保存相关函数 ---

def numpy_to_vips(arr):
    arr = np.moveaxis(arr, 0, -1)   # (C,H,W)->(H,W,C)
    h, w, c = arr.shape
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.uint16:
        fmt = 'ushort'; data = arr.tobytes()
    elif arr.dtype == np.uint8:
        fmt = 'uchar'; data = arr.tobytes()
    else:
        arr = arr.astype(np.uint16); fmt = 'ushort'; data = arr.tobytes()
    return pyvips.Image.new_from_memory(data, w, h, c, fmt)

def save_ome_tiff_multichannel(cropped_arr, out_ome_path, channel_names=None, pixel_size_um=None):
    assert cropped_arr.ndim == 3, f"Input array must be (C, H, W), got {cropped_arr.shape}"
    C, H, W = cropped_arr.shape
    vips_img = numpy_to_vips(cropped_arr)
    bf_dtype = slide_io.vips2bf_dtype(vips_img.format)

    ome_xml_obj = slide_io.create_ome_xml(
        shape_xyzct=slide_io.get_shape_xyzct((W, H), n_channels=C),
        bf_dtype=bf_dtype,
        is_rgb=False, 
        pixel_physical_size_xyu=pixel_size_um,
        channel_names=channel_names,
        colormap=None
    )

    slide_io.save_ome_tiff(
        img=vips_img,
        dst_f=str(out_ome_path),
        ome_xml=ome_xml_obj.to_xml(),
        tile_wh=512,
        compression="lzw",
        Q=100,
        pyramid=False 
    )
    print(f"  -> Saved OME-TIFF: {out_ome_path}")

def save_plain_tiff(cropped_arr, out_path):
    """
    保存为普通的 ImageJ 兼容多通道 TIFF 文件。
    输入: (C, H, W) numpy 数组
    """
    # 使用 tifffile 直接保存
    # imagej=True 会写入 metadata，使得 ImageJ 能够识别为 Hyperstack (C, Z, T)
    tifffile.imwrite(
        str(out_path),
        cropped_arr,
        photometric='minisblack', # 灰度图
        imagej=True,              # 兼容 ImageJ
        compression='zlib',       # 使用压缩以减小体积
        metadata={'axes': 'CYX'}  # 显式声明轴顺序：通道-Y-X
    )
    print(f"  -> Saved Plain TIFF: {out_path}")

# --- 3. 辅助工具函数 (裁剪 - 关键修复版) ---

def get_roi_crop_array(ome_path, roi_path, pad=0):
    ome_p = Path(ome_path)
    roi_obj = roiread(roi_path)
    rois = roi_obj if isinstance(roi_obj, list) else [roi_obj]
    if len(rois) == 0: raise RuntimeError("No ROI found.")
    roi = rois[0]

    left = max(0, int(getattr(roi, "left", 0)) - pad)
    top  = max(0, int(getattr(roi, "top", 0))  - pad)
    right = int(getattr(roi, "right", 0)) + pad
    bottom = int(getattr(roi, "bottom", 0)) + pad
    
    # 1. 使用 pyvips 打开 (随机访问)
    # 不使用 tifffile，因为对于多页 OME-TIFF 它可能只读第一页
    v = pyvips.Image.new_from_file(str(ome_p), access="random")
    
    # 2. 【关键修复】多页自动合并逻辑
    # 如果 pyvips 认为只有一个波段 (bands=1)，但实际上我们需要多个通道
    # 我们尝试读取文件的后续页面 (pages) 并将它们合并
    if v.bands == 1:
        pages_to_join = [v]
        # 尝试加载后续页 (假设最多检查 10 页，通常 4 通道够用了)
        for i in range(1, 10): 
            try:
                # 显式指定 page=i
                p = pyvips.Image.new_from_file(str(ome_p), page=i, access="random")
                # 只有尺寸一致的才认为是通道（排除金字塔层级）
                if p.width == v.width and p.height == v.height:
                    pages_to_join.append(p)
                else:
                    break 
            except:
                break 
        
        if len(pages_to_join) > 1:
            # print(f"    [Info] Merged {len(pages_to_join)} pages into multichannel image.")
            # 修正：使用第一页调用 bandjoin 实例方法
            v = pages_to_join[0].bandjoin(pages_to_join[1:])

    # 3. 检查边界
    img_w, img_h = v.width, v.height
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(img_w, right), min(img_h, bottom)
    
    if x1 <= x0 or y1 <= y0:
        return np.zeros((v.bands, 0, 0), dtype=np.uint16)
    
    # 4. 裁剪
    crop_v = v.extract_area(x0, y0, x1-x0, y1-y0)
    
    # 5. 转为 Numpy (C, H, W)
    arr = np.ndarray(buffer=crop_v.write_to_memory(),
                     dtype=np.uint16 if crop_v.format == 'ushort' else np.uint8,
                     shape=[crop_v.height, crop_v.width, crop_v.bands])
    return np.moveaxis(arr, -1, 0)

# --- 4. 主流程 ---

ROI_DIR = "/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration"
OUT_DIR = "/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/CMH-00003-09235-15387/Original-ROIs-PNG"
os.makedirs(OUT_DIR, exist_ok=True)

roi_list = ['CMH-00003-09235-15387.roi']

IMG_DIR = '/public3/Xinyu/3D_tissue/IF/01.mIHC_rigid_registration/MIF_CMH/registered_slides' 
file_list = os.listdir(IMG_DIR)

# === 配置区域 ==========================================

CHANNEL_INDICES = {
    'DAPI': 0, 
    'CD31': 1,
    'LYVE1': 2 
}

# 【关键修改】选择可视化模式
# 'CD31'  -> 只保存 DAPI(蓝) + CD31(红)
# 'LYVE1' -> 只保存 DAPI(蓝) + LYVE1(洋红)
# 'ALL'   -> 保存 DAPI(蓝) + CD31(红) + LYVE1(洋红)
VISUALIZATION_MODE = 'ALL'  

SAVE_PNG = True
SAVE_OME_TIFF = False 
SAVE_PLAIN_TIFF = False

# =======================================================

for n in file_list:
    if not n.endswith('.tiff') and not n.endswith('.tif'): continue
    
    print(f'=== Processing Image: {n} ===')
    pat_file = re.compile(r'CMH_\d+')
    m_file = pat_file.search(n)
    if not m_file: continue
    
    base_file = m_file.group(0)
    OME_TIF_PATH = os.path.join(IMG_DIR, n)

    #roi_list = os.listdir(ROI_DIR)
    roi_list = roi_list

    for i in roi_list:
        if not i.endswith('.roi'): continue 

        pat = re.compile(r'\d+-\d+-\d+')
        m = pat.search(i)
        if not m: continue

        base = m.group(0)
        roi_file_path = os.path.join(ROI_DIR, i)

        try:
            # 1. 获取裁剪数据 (含多页合并修复)
            cropped_data = get_roi_crop_array(OME_TIF_PATH, roi_file_path, pad=10)
            
            if cropped_data.size == 0 or cropped_data.shape[1] == 0:
                print(f"  [Skip] Empty crop for ROI {base}.")
                continue
            
            print(f"  ROI {base}: Valid crop {cropped_data.shape}")

            # 2. 保存 PNG (根据模式)
            if SAVE_PNG:
                out_png_path = os.path.join(OUT_DIR, f"{base_file}-{base}_{VISUALIZATION_MODE}.png")
                save_mixed_png(cropped_data, out_png_path, CHANNEL_INDICES, mode=VISUALIZATION_MODE)

            # 3. 保存 OME-TIFF
            if SAVE_OME_TIFF:
                out_ome_path = os.path.join(OUT_DIR, f"{base_file}-{base}.ome.tiff")
                ordered_channels = ['DAPI', 'CD31', 'LYVE1'] 
                
                if not os.path.exists(out_ome_path):
                    save_ome_tiff_multichannel(
                        cropped_arr=cropped_data,
                        out_ome_path=out_ome_path,
                        channel_names=ordered_channels,
                        pixel_size_um=None
                    )
            
            # 4. 保存普通 TIFF (ImageJ 兼容)
            if SAVE_PLAIN_TIFF:
                out_plain_path = os.path.join(OUT_DIR, f"{base_file}-{base}.tif")
                if not os.path.exists(out_plain_path):
                    save_plain_tiff(cropped_data, out_plain_path)

        except Exception as e:
            print(f"  Error processing ROI {i} for {n}: {e}")
            import traceback
            traceback.print_exc()


