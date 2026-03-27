#%% 导入库
from valis import registration, valtils, micro_rigid_registrar, preprocessing, slide_io
from skimage.util import img_as_ubyte
from skimage import filters, io
from roifile import roiread
from pathlib import Path
import numpy as np
import os
import re

#%% 获取 ROI 图像
rigid_tif_dir = '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/micro_registered_slides_DAPI'
save_roi_dir = '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/ROI-1304-2892_DAPI'
roi_file = '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/1304-2892.roi'  # FIJI 保存的 ROI
os.makedirs(save_roi_dir, exist_ok=True)

# ======================
# 读取 ROI
# ======================
roi_obj = roiread(roi_file)  # 单个 ROI 返回 ImagejRoi
rois = [roi_obj] if not isinstance(roi_obj, list) else roi_obj

# ======================
# 遍历 ROI 裁剪所有切片
# ======================
# 获取 ROI 的坐标（假设只取第一个 ROI）
roi = rois[0]
minr = roi.top - 20
maxr = roi.bottom 
minc = roi.left - 20
maxc = roi.right + 20

# 遍历 C3_clean 文件夹下所有切片
slice_files = sorted([f for f in os.listdir(rigid_tif_dir) if f.endswith('.tif')])

for slice_fname in slice_files:
    print(slice_fname)

    slice_path = os.path.join(rigid_tif_dir, slice_fname)
    
    # 读取切片
    img = io.imread(slice_path)
    
    # 裁剪 ROI 区域
    roi_img = img[minr:maxr, minc:maxc]
    
    # 保存裁剪结果
    save_path = os.path.join(save_roi_dir, slice_fname)
    io.imsave(save_path, img_as_ubyte(roi_img))
    print(f"保存 {save_path}")

print("所有切片的 ROI 已裁剪完成")


#%% OME-TIFF
'''
功能：根据原始的大图，保存ROI对应的图像，保存为PNG或者ome.tiff,进行对齐

'''

import os
import re
import numpy as np
from pathlib import Path
from skimage import exposure
import imageio.v2 as imageio
import pyvips
from roifile import roiread
from valis import slide_io

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
    安全生成伪彩 PNG。
    mode: "CD31" (蓝+红), "LYVE1" (蓝+洋红), "ALL" (蓝+红+洋红)
    """
    existing_channels = cropped_arr.shape[0]

    # 检查所需的通道索引是否在数据范围内
    def has(ch_name):
        idx = channel_indices.get(ch_name)
        if idx is None: return False
        return idx < existing_channels

    required = {
        "CD31": ["DAPI", "CD31"],
        "LYVE1": ["DAPI", "LYVE1"],
        "ALL": ["DAPI", "CD31", "LYVE1"],
    }

    # 如果缺少必要通道，跳过
    req_channels = required.get(mode, [])
    for ch in req_channels:
        if not has(ch):
            print(f"  -> Skip ({mode}) because channel '{ch}' (Index {channel_indices.get(ch)}) is missing/out of bounds.")
            return

    # 提取 DAPI (所有模式通用)
    idx_dapi = channel_indices["DAPI"]
    dapi = enhance_channel(cropped_arr[idx_dapi])

    H, W = dapi.shape
    R = np.zeros((H, W), np.float32)
    G = np.zeros((H, W), np.float32)
    B = np.zeros((H, W), np.float32)

    # 根据模式叠加颜色
    if mode == "CD31":
        # DAPI(蓝) + CD31(红)
        cd31 = enhance_channel(cropped_arr[channel_indices["CD31"]])
        R = cd31
        B = dapi

    elif mode == "LYVE1":
        # DAPI(蓝) + LYVE1(洋红 = 红+蓝)
        lyve1 = enhance_channel(cropped_arr[channel_indices["LYVE1"]])
        R = lyve1
        B = dapi + lyve1

    elif mode == "ALL":
        # DAPI(蓝) + CD31(红) + LYVE1(洋红)
        # 注意：这里红色通道叠加了 CD31 和 LYVE1，蓝色叠加了 DAPI 和 LYVE1
        cd31 = enhance_channel(cropped_arr[channel_indices["CD31"]])
        lyve1 = enhance_channel(cropped_arr[channel_indices["LYVE1"]])
        R = cd31 + lyve1
        B = dapi + lyve1

    # 合并并转为 uint8
    rgb = np.stack([R, G, B], axis=-1)
    rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
    
    imageio.imwrite(out_path, rgb_uint8)
    print(f"  -> Saved PNG ({mode}): {out_path}")

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

ROI_DIR = "/public3/Xinyu/3D_tissue/IF/registration_test/RoiSet"
OUT_DIR = "/public3/Xinyu/3D_tissue/IF/registration_test/RoiSet_ALL_png"
os.makedirs(OUT_DIR, exist_ok=True)

roi_list = ['00004-02772-17487.roi']

IMG_DIR = '/public3/Xinyu/3D_tissue/IF/registration_test/Rigid_slide' 
file_list = os.listdir(IMG_DIR)

# === 配置区域 ==========================================

CHANNEL_INDICES = {
    'DAPI': 0, 
    'MAP2': 1, 
    'CD31': 2,
    'LYVE1': 3 
}

# 【关键修改】选择可视化模式
# 'CD31'  -> 只保存 DAPI(蓝) + CD31(红)
# 'LYVE1' -> 只保存 DAPI(蓝) + LYVE1(洋红)
# 'ALL'   -> 保存 DAPI(蓝) + CD31(红) + LYVE1(洋红)
VISUALIZATION_MODE = 'ALL'  

SAVE_PNG = False
SAVE_OME_TIFF = False 
SAVE_PLAIN_TIFF = True

# =======================================================

for n in file_list:
    if not n.endswith('.tiff') and not n.endswith('.tif'): continue
    
    print(f'=== Processing Image: {n} ===')
    pat_file = re.compile(r'HJZ_\d+')
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
                ordered_channels = ['DAPI', 'MAP2', 'CD31', 'LYVE1'] 
                
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











#%% 使用DAPIGetter类进行ROI微对齐
'''
## 进入docker容器内部
docker run -it --memory=200g \
    -v "$HOME:$HOME" \
    -v /public3/Xinyu:/public3/Xinyu \
    cdgatenbee/valis-wsi bash
'''

from valis import registration, valtils, micro_rigid_registrar, preprocessing, slide_io
from skimage import filters
import os
import re
import numpy as np

class DAPIGetter(preprocessing.ChannelGetter):
    """
    Select DAPI channel from image
    """
    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level,
                         series=series, *args, **kwargs)

    def create_mask(self):
        dapi_img = self.process_image()

        fg_t = filters.threshold_otsu(dapi_img)
        fg_mask = 255*(dapi_img > fg_t).astype(np.uint8)

        fg_bbox_mask = preprocessing.mask2bbox_mask(fg_mask)

        return fg_bbox_mask


class FixedBioFormatsSlideReader(slide_io.BioFormatsSlideReader):
    """
    BioFormats reader that sets negative pyramid level to 0
    """
    def __init__(self, src_f, series=None, *args, **kwargs):
        super().__init__(src_f=src_f, series=series, *args, **kwargs)

    def slide2vips(self, level, series=None, xywh=None, tile_wh=None, z=0, t=0, *args, **kwargs):
        level = max(0, level)
        return super().slide2vips(level=level, series=series, xywh=xywh, tile_wh=tile_wh, z=z, t=t, *args, **kwargs)

    def slide2image(self, level, series=None, xywh=None, tile_wh=None, z=0, t=0, *args, **kwargs):
        level = max(0, level)
        return super().slide2image(level=level, series=series, xywh=xywh, tile_wh=tile_wh, z=z, t=t, *args, **kwargs)


## 预设路径
slide_src_dir = "/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501"
results_dst_dir = "/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration"
registered_slide_dst_dir = "/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/micro_registered_slides"

os.makedirs(results_dst_dir, exist_ok = True)
os.makedirs(registered_slide_dst_dir, exist_ok = True)

sorted_img_f_list = ['HJZ_1-DAPI-cropped', 'HJZ_2-DAPI-cropped', 'HJZ_3-DAPI-cropped', 'HJZ_5-DAPI-cropped', 'HJZ_6-DAPI-cropped']


# sorted_img_f_list = os.listdir(slide_src_dir)
# valtils.sort_nicely(sorted_img_f_list) # sorting is done in place


## 有顺序的
# registrar = registration.Valis(slide_src_dir, results_dst_dir, img_list= sorted_img_f_list, imgs_ordered=True)
# rigid_registrar, non_rigid_registrar, error_df = registrar.register()

## 刚性和非刚性对齐
# Create a Valis object and use it to register the slides in slide_src_dir
# registrar = registration.Valis(slide_src_dir, 
#                                results_dst_dir, 
#                                micro_rigid_registrar_cls = micro_rigid_registrar.MicroRigidRegistrar,
#                                # max_processed_image_dim_px = 2000,
#                                # max_non_rigid_registration_dim_px = 1500,
#                                norm_method = 'img_stats')
# 
# rigid_registrar, non_rigid_registrar, error_df = registrar.register()

registrar = registration.Valis(slide_src_dir, 
                               results_dst_dir, 
                               micro_rigid_registrar_cls = micro_rigid_registrar.MicroRigidRegistrar,
                               # max_processed_image_dim_px = 2000,
                               # max_non_rigid_registration_dim_px = 1500,
                               norm_method = 'img_stats')

rigid_registrar, non_rigid_registrar, error_df = registrar.register(if_processing_cls = DAPIGetter, reader_cls = FixedBioFormatsSlideReader)

registrar.warp_and_save_slides(registered_slide_dst_dir, crop="reference", interp_method = 'bicubic', level = 0)




#%% 保存对齐后的单通道ROI图片
'''
## 进入docker容器内部
docker run -it --memory=200g \
    -v "$HOME:$HOME" \
    -v /public3/Xinyu:/public3/Xinyu \
    cdgatenbee/valis-wsi bash
'''
from valis import registration, valtils, micro_rigid_registrar, preprocessing, slide_io
import numpy as np
import tifffile
import cv2
import os
import re
from roifile import roiread
try:
    from skimage.filters import threshold_otsu
    _has_otsu = True
except Exception:
    _has_otsu = False


registrar = registration.load_registrar('/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/ROI-00004-07662-12501/data/ROI-00004-07662-12501_registrar.pickle')

files = [entry.name for entry in os.scandir('/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501') if entry.is_file()]

pat = re.compile(r'HJZ_\d+-\d+-\d+-\d+')
for i in files:
    print(i)
    # 可调参数
    m = pat.search(i)
    base = m.group(0)

    print(base)

    CD31_index = 0
    out_file = '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/micro_registered_slides_DAPI'
    os.makedirs(out_file, exist_ok = True)
    out_base = f"{out_file}/{base}"
    out_chan = out_base + "_uint16_enhanced.tif"      # 输出 uint16 增强后图
    # out_mask = out_base + "_mask_uint8.tif"           # 若需要保留 mask 可保存
    # out_preview = out_base + "_preview_uint8.tif"     # 便于快速检查的 uint8 预览

    # 增强参数（可调）
    p_low = 0.1        # 低百分位 (percentile) ，作为下限
    p_high = 99.9      # 高百分位，作为上限（去掉极端亮点）
    gamma = 0.9        # gamma < 1 会增强暗部，>1 会压暗（可设为1不做）
    clip_low = None    # 可选固定最小值（None 表示使用 percentile）
    clip_high = None   # 可选固定最大值（None 表示使用 percentile）

    # ---- （前面部分保持不变）：从 pyvips -> numpy 得到 arr_cyx ----
    # 假定你已有 arr_cyx (C, Y, X)
    # 例如使用： buf = v.write_to_memory(); arr = np.frombuffer(buf, dtype=np.uint16)...
    # 我这里直接使用 arr_cyx
    slide_f = f'/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/{i}'
    slide_obj = registrar.get_slide(slide_f)
    slide = slide_obj.warp_slide(level=0, non_rigid=False, crop='reference', interp_method = 'bicubic')
    v = slide  # pyvips.Image

    buf = v.write_to_memory()  # bytes
    arr = np.frombuffer(buf, dtype=np.uint16)
    # pyvips stores (height, width, bands) memory order in write_to_memory
    arr = arr.reshape((v.height, v.width, v.bands))  # H, W, C
    arr_cyx = np.transpose(arr, (2, 0, 1))  # C, Y, X
    # 取出 CD31 通道（原始 ushort）
    ch = arr_cyx[CD31_index].astype(np.float32)  # 用 float 做运算

    # ---- 计算 clip 上下界（用 percentile） ----
    if clip_low is None:
        low = np.percentile(ch.reshape(-1), p_low)
    else:
        low = float(clip_low)
    if clip_high is None:
        high = np.percentile(ch.reshape(-1), p_high)
    else:
        high = float(clip_high)

    # 防止 low == high
    if high <= low:
        high = ch.max() if ch.max() > low else low + 1.0

    # ---- 线性缩放到 0..1（同时做截断） ----
    ch_clipped = np.clip(ch, low, high)
    ch_norm = (ch_clipped - low) / (high - low)   # 0..1

    # ---- 可选 gamma 校正 ----
    if gamma is not None and gamma != 1.0:
        # gamma < 1 拉亮暗区，gamma >1 压暗
        ch_norm = np.power(ch_norm, gamma)

    # ---- 再次 clip 确保 0..1，然后转为 uint16 ----
    ch_norm = np.clip(ch_norm, 0.0, 1.0)
    ch_u16 = (ch_norm * 65535.0).round().astype(np.uint16)

    # ---- 保存 uint16（不去背景） ----
    tifffile.imwrite(out_chan, ch_u16, bigtiff=True)
    print("Saved enhanced uint16 channel:", out_chan)


#%% 保存对齐后的三个通道的ROI文件
from valis import registration, valtils, micro_rigid_registrar, preprocessing, slide_io
import numpy as np
import tifffile
import cv2
import os
import re
from roifile import roiread
try:
    from skimage.filters import threshold_otsu
    _has_otsu = True
except Exception:
    _has_otsu = False

# 加载 Registrar
registrar_path = '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-02772-17487/Micro-registration/ROI-00004-02772-17487/data/ROI-00004-02772-17487_registrar.pickle'
registrar = registration.load_registrar(registrar_path)

# 获取文件列表
data_dir = '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-02772-17487'
files = [entry.name for entry in os.scandir(data_dir) if entry.is_file()]

# 输出目录
out_dir = '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-02772-17487/Micro-registration/micro_registered_slides_multichannel'
os.makedirs(out_dir, exist_ok=True)

# 配置目标通道 (名称: 索引)
# 假设顺序: 0:DAPI, 1:MAP2, 2:CD31, 3:LYVE1
TARGET_CHANNELS = {
    'DAPI': 0,
    'CD31': 2,
    'LYVE1': 3
}

# 增强参数（可调）
p_low = 0.1        # 低百分位
p_high = 99.9      # 高百分位
gamma = 0.9        # Gamma 校正
clip_low = None    
clip_high = None   

pat = re.compile(r'HJZ_\d+-\d+-\d+-\d+')

for i in files:
    if not i.endswith(('.vsi', '.tif', '.tiff')): 
        continue
        
    print(f"Processing: {i}")
    m = pat.search(i)
    if not m:
        print(f"  [Skip] Pattern not found in {i}")
        continue
        
    base = m.group(0)
    
    # 构造完整路径
    slide_f = os.path.join(data_dir, i)
    
    try:
        # 获取并 Warp 切片
        slide_obj = registrar.get_slide(slide_f)
        # 使用 bicubic 插值获得更好的质量，level=0 为最高分辨率
        slide = slide_obj.warp_slide(level=0, non_rigid=False, crop='reference', interp_method='bicubic')
        v = slide  # pyvips.Image

        # --- 将 Pyvips 转为 Numpy (C, Y, X) ---
        # 注意：这里会加载整个图像到内存，请确保内存足够
        buf = v.write_to_memory()
        arr = np.frombuffer(buf, dtype=np.uint16)
        arr = arr.reshape((v.height, v.width, v.bands)) # H, W, C
        arr_cyx = np.transpose(arr, (2, 0, 1)) # C, H, W
        
        # 准备列表存储处理后的通道
        processed_stack = []
        
        # --- 遍历目标通道进行增强 ---
        for ch_name, ch_idx in TARGET_CHANNELS.items():
            if ch_idx >= arr_cyx.shape[0]:
                print(f"  [Warn] Channel {ch_name} index {ch_idx} out of bounds. Using empty channel.")
                processed_stack.append(np.zeros_like(arr_cyx[0]))
                continue

            # 取出单通道数据 (float32 用于计算)
            ch = arr_cyx[ch_idx].astype(np.float32)

            # 1. 计算 clip 上下界
            if clip_low is None:
                low = np.percentile(ch.reshape(-1), p_low)
            else:
                low = float(clip_low)
            
            if clip_high is None:
                high = np.percentile(ch.reshape(-1), p_high)
            else:
                high = float(clip_high)

            if high <= low:
                high = ch.max() if ch.max() > low else low + 1.0

            # 2. 截断并归一化 (0..1)
            ch_clipped = np.clip(ch, low, high)
            ch_norm = (ch_clipped - low) / (high - low)

            # 3. Gamma 校正
            if gamma is not None and gamma != 1.0:
                ch_norm = np.power(ch_norm, gamma)

            # 4. 转回 uint16
            ch_norm = np.clip(ch_norm, 0.0, 1.0)
            ch_u16 = (ch_norm * 65535.0).round().astype(np.uint16)
            
            processed_stack.append(ch_u16)
            # print(f"    Enhanced {ch_name}")

        # --- 合并并保存为多通道 TIFF ---
        if processed_stack:
            # 堆叠为 (3, H, W)
            multi_channel_img = np.stack(processed_stack, axis=0)
            
            out_name = f"{out_dir}/{base}_DAPI_CD31_LYVE1_enhanced.tif"
            
            # 保存 (imagej=True 使其在 ImageJ 中被识别为 Stack)
            tifffile.imwrite(
                out_name, 
                multi_channel_img, 
                photometric='minisblack',
                imagej=True,
                metadata={'axes': 'CYX', 'Labels': list(TARGET_CHANNELS.keys())}, # 尝试写入通道标签
                compression='zlib'
            )
            print(f"  -> Saved Stack: {out_name}")

    except Exception as e:
        print(f"  [Error] Failed processing {i}: {e}")


