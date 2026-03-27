#%% #---------------------------------- VALIS连续切片免疫荧光对齐 ----------------------------------#
'''
VALIS: 连续切片免疫荧光对齐

## 进入docker容器内部
docker run -it --memory=200g \
    -v "$HOME:$HOME" \
    -v /public3/Xinyu:/public3/Xinyu \
    cdgatenbee/valis-wsi bash

warp_and_save_slide参数: 
    1. dst_f: 输出的OME-TIFF文件路径
    2. level: 0/1/2/3, 0为最原始分辨率, 设置的分辨率层级
    3. non_grid: bool值, 是否执行非刚性形变
    4. crop: 如何输出裁剪图像, ['overlap', 'reference']
    5. src_f: 指定要warp的源文件, 用于warp某些预处理版本
    6. channel_names: 输出通道名称
    7. colormap: 每个通道的颜色
    8. interp_method: 插值方式, ['nearest', 'bilinear', 'bicubic'], 连续组织选bicubic, mask选nearest
    9. tile_wh: 输出瓦片大小
    10. cimpression: 压缩方式
    11. Q: 压缩质量
    12. pyramid: 是否保存多层级金字塔
    13. reader: 指定slide reader
'''
from valis import registration, valtils, micro_rigid_registrar, preprocessing, slide_io
from skimage import filters
import os
import re
import numpy as np

## 预设路径
slide_src_dir = "/public3/Xinyu/3D_tissue/IF/mIHC_data_all"
results_dst_dir = "/public3/Xinyu/3D_tissue/IF/01.mIHC_rigid_registration"
registered_slide_dst_dir = "/public3/Xinyu/3D_tissue/IF/01.mIHC_rigid_registration/registered_slides"
registered_slide_dst_tif_dir = "/public3/Xinyu/3D_tissue/IF/registration_roi/micro_tif_slides"

os.makedirs(results_dst_dir, exist_ok = True)
os.makedirs(registered_slide_dst_dir, exist_ok = True)

sorted_img_f_list = ['图像_HJZ_1', '图像_HJZ_2', '图像_HJZ_3', '图像_HJZ_5', '图像_HJZ_6', '图像_HJZ_7', '图像_HJZ_8', '图像_HJZ_9', '图像_HJZ_10', '图像_HJZ_11', '图像_HJZ_12', '图像_HJZ_13', 
                     '图像_HJZ_14', '图像_HJZ_15', '图像_HJZ_16', '图像_HJZ_17', '图像_HJZ_18', '图像_HJZ_19', '图像_HJZ_20', '图像_HJZ_21', '图像_HJZ_22', '图像_HJZ_23', '图像_HJZ_24', '图像_HJZ_25', 
                     '图像_HJZ_26', '图像_HJZ_27', '图像_HJZ_28', '图像_HJZ_29', '图像_HJZ_30', '图像_HJZ_31', '图像_HJZ_32', '图像_HJZ_33', '图像_HJZ_34', '图像_HJZ_35', '图像_HJZ_36', '图像_HJZ_37', 
                     '图像_HJZ_39', '图像_HJZ_41', '图像_HJZ_42', '图像_HJZ_43', '图像_HJZ_46', '图像_HJZ_47', '图像_HJZ_48', '图像_HJZ_49', '图像_HJZ_51', '图像_HJZ_52', '图像_HJZ_53', '图像_HJZ_54', 
                     '图像_HJZ_55', '图像_HJZ_56', '图像_HJZ_57', '图像_HJZ_59', '图像_HJZ_60', '图像_HJZ_61', '图像_HJZ_62', '图像_HJZ_63', '图像_HJZ_64', '图像_HJZ_66', '图像_HJZ_67', '图像_HJZ_68', 
                     '图像_HJZ_69', '图像_HJZ_70', '图像_HJZ_71', '图像_HJZ_72', '图像_HJZ_73', '图像_HJZ_74', '图像_HJZ_75', '图像_HJZ_76', '图像_HJZ_77', '图像_HJZ_78', '图像_HJZ_79', '图像_HJZ_80', 
                     '图像_HJZ_81', '图像_HJZ_82', '图像_HJZ_83', '图像_HJZ_84', '图像_HJZ_85', '图像_HJZ_86', '图像_HJZ_88 - 20x_DAPI, BNA, GNA, Cy5_01', '图像_HJZ_90 - 20x_DAPI, BNA, GNA, Cy5_01', 
                     '图像_HJZ_92 - 20x_DAPI, BNA, GNA, Cy5_01', '图像_HJZ_94 - 20x_DAPI, BNA, GNA, Cy5_01', '图像_HJZ_96 - 20x_DAPI, BNA, GNA, Cy5_01', '图像_HJZ_98 - 20x_DAPI, BNA, GNA, Cy5_01', 
                     '图像_HJZ_100 - 20x_DAPI, BNA, GNA, Cy5_01', '图像_HJZ_102 - 20x_DAPI, BNA, GNA, Cy5_01', '图像_HJZ_104', '图像_HJZ_106', '图像_HJZ_108']


# sorted_img_f_list = os.listdir(slide_src_dir)
# valtils.sort_nicely(sorted_img_f_list) # sorting is done in place


## 有顺序的
# registrar = registration.Valis(slide_src_dir, results_dst_dir, img_list= sorted_img_f_list, imgs_ordered=True)
# rigid_registrar, non_rigid_registrar, error_df = registrar.register()

## 刚性和非刚性对齐
# Create a Valis object and use it to register the slides in slide_src_dir
registrar = registration.Valis(slide_src_dir, 
                               results_dst_dir, 
                               micro_rigid_registrar_cls = micro_rigid_registrar.MicroRigidRegistrar,
                               max_processed_image_dim_px = 2000,
                               max_non_rigid_registration_dim_px = 1500,
                               norm_method = 'img_stats')

registrar = registration.Valis(slide_src_dir, results_dst_dir)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()

# Save all registered slides as ome.tiff
registrar.warp_and_save_slides(registered_slide_dst_dir, crop="reference", interp_method = 'bicubic', level = 1)

## 微对齐
# Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that is 25% full resolution.
img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
min_max_size = np.min([np.max(d) for d in img_dims])
img_areas = [np.multiply(*d) for d in img_dims]
max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])

micro_reg_fraction = 0.2
micro_reg_size = np.floor(min_max_size * micro_reg_fraction).astype(int)

# 执行第二次高分辨率非刚性配准
micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px = micro_reg_size)


# colormap = {
#     "DAPI": (0, 0, 255),     # 蓝
#     "MAP2": (0, 255, 0),     # 绿
#     "CD31": (255, 0, 0),     # 红
#     "LYVE1": (128, 0, 128)    # 紫
# }

## 逐张导出
pat = re.compile(r'HJZ_\d+')
for i in sorted_img_f_list:
    print(i)
    slide_f = f'/public3/Xinyu/3D_tissue/IF/mIHC_data_all/{i}.vsi'
    m = pat.search(i)
    base = m.group(0)
    slide_obj = registrar.get_slide(slide_f)
    slide_obj.warp_and_save_slide(f'{registered_slide_dst_dir}/{base}-rigid.ome.tiff', 
                                  level = 1, 
                                  crop = 'reference',
                                  channel_names = ['DAPI', 'MAP2', 'CD31', 'LYVE1'],
                                  interp_method = 'nearest',
                                  compression = 'lzw',
                                  pyramid = True,
                                  non_rigid = False)

# Kill the JVM
registration.kill_jvm()

#%% 使用DAPIGetter类
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

## 刚性和非刚性对齐
# Create a Valis object and use it to register the slides in slide_src_dir
registrar = registration.Valis(slide_src_dir, 
                               results_dst_dir, 
                               micro_rigid_registrar_cls = micro_rigid_registrar.MicroRigidRegistrar,
                               max_processed_image_dim_px = 2000,
                               max_non_rigid_registration_dim_px = 1500,
                               norm_method = 'img_stats')

rigid_registrar, non_rigid_registrar, error_df = registrar.register(if_processing_cls = DAPIGetter, reader_cls = FixedBioFormatsSlideReader)

# Save all registered slides as ome.tiff
# registrar.warp_and_save_slides(registered_slide_dst_dir, crop="reference", interp_method = 'bicubic', level = 3)

## 微对齐
# Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that is 25% full resolution.
img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
min_max_size = np.min([np.max(d) for d in img_dims])
img_areas = [np.multiply(*d) for d in img_dims]
max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])

micro_reg_fraction = 0.2
micro_reg_size = np.floor(min_max_size * micro_reg_fraction).astype(int)

# Perform high resolution non-rigid registration
micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px = micro_reg_size)

## 逐张导出
pat = re.compile(r'HJZ_\d+')
for i in sorted_img_f_list:
    print(i)
    slide_f = f'/public3/Xinyu/3D_tissue/IF/mIHC_data_all/{i}.vsi'
    m = pat.search(i)
    base = m.group(0)
    slide_obj = registrar.get_slide(slide_f)
    slide_obj.warp_and_save_slide(f'{registered_slide_dst_dir}/{base}-DAPI-rigid.ome.tiff', 
                                  level = 1, 
                                  crop = 'reference',
                                  channel_names = ['DAPI', 'MAP2', 'CD31', 'LYVE1'],
                                  interp_method = 'nearest',
                                  compression = 'lzw',
                                  pyramid = True,
                                  non_rigid = False)


#%% #---------------------------------- 保存单通道图像 ----------------------------------#
'''
逐张输出指定通道图像
'''
import numpy as np
import tifffile
import cv2
from roifile import roiread
try:
    from skimage.filters import threshold_otsu
    _has_otsu = True
except Exception:
    _has_otsu = False

# ======================
# 读取 ROI
# ======================
roi_obj = roiread('/public3/Xinyu/3D_tissue/IF/registration_all/04976-11987.roi')  # 单个 ROI 返回 ImagejRoi
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


pat = re.compile(r'HJZ_\d+')
for i in sorted_img_f_list:
    # 可调参数
    m = pat.search(i)
    base = m.group(0)

    print(base)

    CD31_index = 2
    out_base = f"/public3/Xinyu/3D_tissue/IF/registration_all/TIFF-CD31-roi04976-11987/{base}"
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
    slide_f = f'/public3/Xinyu/3D_tissue/IF/mIHC_data_all/{i}.vsi'
    slide_obj = registrar.get_slide(slide_f)
    slide = slide_obj.warp_slide(level=1, non_rigid=False, crop='reference', interp_method = 'bicubic')
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

    roi_img = ch_u16[minr:maxr, minc:maxc]

    # ---- 保存 uint16（不去背景） ----
    tifffile.imwrite(out_chan, roi_img, bigtiff=True)
    print("Saved enhanced uint16 channel:", out_chan)

#%% #---------------------------------- 保存为多通道图像 ----------------------------------#
import numpy as np
import tifffile
import cv2
import random

# -------- 为每个通道做 percentile 归一化并增强 --------
def normalize_channel(ch, p_low=0.5, p_high=99.5, gamma=1.0):
    """把单通道浮点数组映射到 0..1, 使用 percentile 截断并可做 gamma"""
    chf = ch.astype(np.float32)
    lo = np.percentile(chf.reshape(-1), p_low)
    hi = np.percentile(chf.reshape(-1), p_high)
    if hi <= lo:
        hi = chf.max() if chf.max() > lo else lo + 1.0
    chc = np.clip(chf, lo, hi)
    chn = (chc - lo) / (hi - lo)
    if gamma is not None and gamma != 1.0:
        chn = np.power(chn, gamma)
    chn = np.clip(chn, 0.0, 1.0)
    return chn

# -------- 配置区（按需修改） --------
# 要保存的三通道索引（0-based）
ch_dapi = 0   # DAPI -> Blue
ch_map2 = 1   # MAP2 -> Green
ch_cd31 = 2   # CD31 -> Red
ch_lyve1 = 3
SAVE_CHANNELS = [ch_cd31, ch_lyve1, ch_dapi]  # order -> [R, G, B]

# 增强参数
p_low = 0.2     # 下百分位 (用于剪切)
p_high = 99.9   # 上百分位 (用于剪切)
gamma = 0.9     # gamma 校正 (若不需要设置 1.0)

os.makedirs(registered_slide_dst_tif_dir, exist_ok = True)

# -------- 从 pyvips 得到 numpy (C, Y, X) --------
## 随机挑选五张样本，作为训练样本集进行血管标注
random.seed(42)
sample_select = random.sample(sorted_img_f_list, 5)

pat = re.compile(r'HJZ_\d+')
for i in sorted_img_f_list:
    m = pat.search(i)
    base = m.group(0)

    print(base)

    # 输出
    out_rgb = f"{registered_slide_dst_tif_dir}/{base}_CD31_LYVE1_enhance.tif"

    slide_f = f'{slide_src_dir}/{sorted_img_f_list[0]}.vsi'
    slide_obj = registrar.get_slide(slide_f)
    slide = slide_obj.warp_slide(level=1, non_rigid=False, crop='reference')
    v = slide  # 你已经有的 pyvips.Image 对象
    buf = v.write_to_memory()
    arr = np.frombuffer(buf, dtype=np.uint16)
    arr = arr.reshape((v.height, v.width, v.bands))  # H, W, C
    arr_cyx = np.transpose(arr, (2, 0, 1))  # C, Y, X


    # 选择并归一化（注意顺序：我们要保存为 RGB，因此 SAVE_CHANNELS 给的是 [R_index, G_index, B_index]）
    r_ch = normalize_channel(arr_cyx[SAVE_CHANNELS[0]], p_low = p_low, p_high = p_high, gamma = gamma)
    g_ch = normalize_channel(arr_cyx[SAVE_CHANNELS[1]], p_low = p_low, p_high = p_high, gamma = gamma)
    b_ch = normalize_channel(arr_cyx[SAVE_CHANNELS[2]], p_low = p_low, p_high = p_high, gamma = gamma)

    # 合成为 uint8 RGB
    r8 = (r_ch * 255.0).round().astype(np.uint8)
    g8 = (g_ch * 255.0).round().astype(np.uint8)
    b8 = (b_ch * 255.0).round().astype(np.uint8)

    rgb = np.stack([r8, g8, b8], axis=-1)  # H, W, 3

    # 可选：做轻微去噪或形态学操作（例如开运算），如需取消注释
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # for c in range(3):
    #     rgb[..., c] = cv2.morphologyEx(rgb[..., c], cv2.MORPH_OPEN, kernel)

    # -------- 保存为 RGB TIFF（photometric='rgb'） --------
    tifffile.imwrite(out_rgb, rgb, photometric='rgb', bigtiff=True)
    print("Saved RGB image:", out_rgb)


