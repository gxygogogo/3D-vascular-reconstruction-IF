#%% 导入库
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


#%% 保存对齐后的单通道ROI图片
'''
## 进入docker容器内部
docker run -it --memory=200g \
    -v "$HOME:$HOME" \
    -v /public3/Xinyu:/public3/Xinyu \
    cdgatenbee/valis-wsi bash
'''
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


region = '00001-19954-13932'

registrar = registration.load_registrar(f'/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/HXJ-{region}/Micro-registration/HXJ-{region}/data/HXJ-{region}_registrar.pickle')

files = [entry.name for entry in os.scandir(f'/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/HXJ-{region}') if entry.is_file()]

pat = re.compile(r'HXJ_\d+-\d+-\d+-\d+')
for i in files:
    print(i)
    # 可调参数
    m = pat.search(i)
    base = m.group(0)

    print(base)

    CD31_index = 1
    out_file = f'/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/HXJ-{region}/Micro-registration/micro_registered_slides_CD31'
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
    slide_f = f'/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/HXJ-{region}/Micro-registration/micro_registered_slides/{i}'
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
