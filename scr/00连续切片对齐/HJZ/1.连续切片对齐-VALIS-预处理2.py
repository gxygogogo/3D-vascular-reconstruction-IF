#%% 导入库
'''
功能: 对数据进行预处理, 然后使用处理后的DAPI进行对齐

## 进入docker容器内部
docker run -it --memory=200g \
    -v "$HOME:$HOME" \
    -v /public3/Xinyu:/public3/Xinyu \
    cdgatenbee/valis-wsi bash
'''
import os
import cv2
import pyvips
import shutil
import traceback
import numpy as np
import tifffile as tf
from pathlib import Path
from skimage import exposure, filters
from valis import valtils, slide_io, registration, preprocessing
from valis.micro_rigid_registrar import MicroRigidRegistrar

#%% # -------------------- 配置路径 --------------------
# 数据源：一个文件夹，内含多张连续切片的 .vsi 文件（文件名顺序需与切片顺序一致）
VSI_DIR = Path("/public3/Xinyu/3D_tissue/IF/data5")   
OUT_ROOT = Path("/public3/Xinyu/3D_tissue/IF/serial_registration")   # 输出根目录
TMP_DIR = OUT_ROOT / "tmp_clahe"   # 临时 CLAHE padded dapi 文件
PADDED_DIR = OUT_ROOT / "padded"   # pad 后的原始图（保留）
WARPED_DIR = OUT_ROOT / "warped_multi"  # warp 后的多通道输出
LEVEL_FOR_WARP = 1   # 用于 warp 的 pyramid level（0: full-res；1/2: 下采样）
DAPI_BAND = 0        # DAPI 在 vsi 里的通道索引（0-based）
REFERENCE_SLIDE_NAME = '图像_HJZ_1'

# VALIS 配置（可按内存和效果调整）
LEVEL_FOR_PREP = 2
MAX_PROCESSED_IMAGE_DIM_PX = 2000
MAX_NON_RIGID_REGISTRATION_DIM_PX = 1500

# CLAHE 参数
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE = (16,16)

# 是否在结束后删除临时 CLAHE 文件
CLEANUP_TMP = True

# 是否针对 ROI 只处理 ROI（若不需要请设为 None）
ROI_XYWH = None  # e.g. (left, top, width, height) in pixels at same level as used (None 表示处理全图)

# 创建输出目录
OUT_ROOT.mkdir(parents = True, exist_ok = True)
TMP_DIR.mkdir(parents = True, exist_ok = True)
PADDED_DIR.mkdir(parents = True, exist_ok = True)
WARPED_DIR.mkdir(parents = True, exist_ok = True)

#%% 函数定义

def list_vsi_files_sorted(vsi_dir: Path):
    paths = [p for p in vsi_dir.iterdir() if p.is_file() and p.suffix.lower() in (".vsi", ".vms", ".tif", ".tiff")]
    # 尽量按文件名自然排序（若有编号应保证顺序）
    files = [str(p) for p in paths]
    valtils.sort_nicely(files)
    return [Path(f) for f in files]

def pad_pyvips_image(v: pyvips.Image, target_w: int, target_h: int, extend='black'):
    pad_w = max(0, target_w - v.width)
    pad_h = max(0, target_h - v.height)
    if pad_w > 0 or pad_h > 0:
        left = pad_w // 2
        top = pad_h // 2
        return v.embed(left, top, target_w, target_h, extend=extend)
    return v

# pyvips -> numpy (H,W[,C]) for ushort/uchar
def pyvips_to_numpy(vimg):
    # vimg: pyvips.Image, access must permit write_to_memory
    buf = vimg.write_to_memory()
    if vimg.format == 'uchar':
        dtype = np.uint8
    else:
        dtype = np.uint16
    arr = np.frombuffer(buf, dtype=dtype)
    arr = arr.reshape((vimg.height, vimg.width, vimg.bands))
    return arr

def clahe_on_uint16_channel(ch, clip_limit=2.0, tile_grid_size=(16,16)):
    """ 输入 ch: uint16 array, 返回 uint8 clahe 结果 """
    # 先线性缩放到 0-255
    ch = ch.astype(np.float32)
    # 用百分位去掉极端值
    p_low = np.percentile(ch, 0.5)
    p_high = np.percentile(ch, 99.5)
    if p_high <= p_low:
        p_low = ch.min()
        p_high = ch.max() if ch.max()>p_low else p_low+1
    ch = np.clip(ch, p_low, p_high)
    ch8 = ((ch - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    out = clahe.apply(ch8)
    return out

# 根据 slide_obj 找到原始 padded file
def padded_orig_for_slideobj(slide_obj):
    # slide_obj.src_f 是 clahe 路径 (tmp/clahe_<name>.tif)
    src_name = Path(slide_obj.src_f).stem  # clahe_<name>
    if src_name.startswith("clahe_"):
        orig_name = src_name.replace("clahe_", "padded_")
    else:
        orig_name = "padded_" + src_name
    cand = PADDED_DIR / (orig_name + ".tif") if not (PADDED_DIR / orig_name).exists() else PADDED_DIR / orig_name
    # try both with and without .tif extension
    if (PADDED_DIR / (orig_name + ".tif")).exists():
        return str(PADDED_DIR / (orig_name + ".tif"))
    if (PADDED_DIR / orig_name).exists():
        return str(PADDED_DIR / orig_name)
    # fallback: try to find by stem
    for f in PADDED_DIR.iterdir():
        if f.stem == orig_name:
            return str(f)
    return None

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


#%% # -------------------- Step 1：收集并求最大尺寸 --------------------
vsi_files = list_vsi_files_sorted(VSI_DIR)
LEVEL_FOR_PREP = 2  # 可调整
max_w, max_h = 0, 0
failed_files = []

for p in vsi_files:
    try:
        # 尝试用 valis 的 reader 获取尺寸信息（轻量，不会一次性load full-res）
        ReaderCls = slide_io.get_slide_reader(str(p))
        reader = ReaderCls(str(p))
        dims = getattr(reader, "slide_dimensions_wh", None)  # 可能为 None

        if dims and isinstance(dims, (list, tuple)) and len(dims) > LEVEL_FOR_PREP:
            # dims 通常是 [(W0,H0),(W1,H1),...]
            w, h = dims[LEVEL_FOR_PREP]
        elif dims and isinstance(dims, (list, tuple)) and len(dims) > 0:
            # 如果没那层，选最高分辨率（0）
            w, h = dims[0]
        else:
            # 尝试通过 slide2vips 读取指定 level 的 pyvips image（reader 可能实现）
            if hasattr(reader, "slide2vips"):
                try:
                    v = reader.slide2vips(level=LEVEL_FOR_PREP)
                    w, h = v.width, v.height
                except Exception:
                    # 再退回到读取 level=0
                    try:
                        v = reader.slide2vips(level=0)
                        w, h = v.width, v.height
                    except Exception:
                        # 最后兜底用 pyvips 直接打开文件（有风险）
                        v = pyvips.Image.new_from_file(str(p))
                        w, h = v.width, v.height
            else:
                # 如果 reader 没有 slide2vips，兜底用 pyvips
                v = pyvips.Image.new_from_file(str(p))
                w, h = v.width, v.height

        # 如果 reader 有可关闭的方法才调用（有些 reader 没实现 close）
        if hasattr(reader, "close"):
            try:
                reader.close()
            except Exception:
                pass

        max_w = max(max_w, w)
        max_h = max(max_h, h)
        print(f"{p.name} -> level {LEVEL_FOR_PREP} size: {w} x {h}")

    except Exception as e:
        print(f"get dims failed for {p.name} -> {e}")
        traceback.print_exc()
        failed_files.append(str(p))

print("Computed padded target size (w x h):", max_w, max_h)
print("Failed files:", failed_files)

#%% # -------------------- Step 2：为每张切片生成 padded 原图 和 CLAHE padded DAPI --------------------
clahe_paths = []
orig_padded_dapi_paths = []
failed = []

for p in vsi_files:
    name = p.stem
    try:
        ReaderCls = slide_io.get_slide_reader(str(p))
        reader = ReaderCls(str(p))
        # 使用 reader 提取 level 的 image（pyvips 对象），按 VALIS reader 的 api
        # 许多 reader 实现有 slide2vips(level=)
        if hasattr(reader, "slide2vips"):
            v = reader.slide2vips(level=LEVEL_FOR_PREP)   # pyvips.Image
        else:
            # 兜底：使用 slide_io 的通用工具
            v = slide_io.slide2vips(str(p), level=LEVEL_FOR_PREP)
        # 提取 DAPI band（如果多通道）
        if v.bands > 1:
            dapi_v = v.extract_band(DAPI_BAND)
        else:
            dapi_v = v

        # pad: 使用 v.embed 保持 pyvips 对象，不转成 numpy 全图（更高效）
        pad_left = (max_w - dapi_v.width)//2 if max_w > dapi_v.width else 0
        pad_top  = (max_h - dapi_v.height)//2 if max_h > dapi_v.height else 0
        if pad_left>0 or pad_top>0:
            dapi_padded_v = dapi_v.embed(pad_left, pad_top, max_w, max_h, extend="black")
        else:
            dapi_padded_v = dapi_v

        # 转 numpy 做 CLAHE（需要完整像素数据）
        arr = pyvips_to_numpy(dapi_padded_v)  # H,W,1
        ch = arr[:,:,0]
        clahe8 = clahe_on_uint16_channel(ch, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE)

        # 转回 pyvips 并保存临时文件（uint8）
        clahe_v = pyvips.Image.new_from_memory(clahe8.tobytes(), dapi_padded_v.width, dapi_padded_v.height, 1, 'uchar')
        clahe_path = TMP_DIR / f"clahe_{name}.tif"
        clahe_v.write_to_file(str(clahe_path))
        clahe_paths.append(str(clahe_path))

        # 可选：保存 padded 原始多通道（如果你确实需要 later 用 warp 回原始）
        try:
            # 读取原始 multi-channel 的 same level（或用 level=LEVEL_FOR_PREP）
            if hasattr(reader, "slide2vips"):
                orig_v = reader.slide2vips(level=LEVEL_FOR_PREP)  # multi-band
            else:
                orig_v = slide_io.slide2vips(str(p), level=LEVEL_FOR_PREP)
            # pad multi-channel similarly
            if max_w>orig_v.width or max_h>orig_v.height:
                orig_padded_v = orig_v.embed((max_w-orig_v.width)//2, (max_h-orig_v.height)//2, max_w, max_h, extend="black")
            else:
                orig_padded_v = orig_v
            padded_orig_path = PADDED_DIR / f"padded_{name}.tif"
            # 写出（注意：可能很大，若磁盘不足请注释掉这块）
            orig_padded_v.write_to_file(str(padded_orig_path))
            orig_padded_dapi_paths.append(str(padded_orig_path))
        except Exception as e:
            # 如果写 multi-channel 出错，不要中断主流程
            print("warning: saving padded multichannel failed for", name, "->", e)

        reader.close()
        print("Prepared:", name)
    except Exception as e:
        print("Prepare failed for", p.name, "->", e)
        traceback.print_exc()
        failed.append(str(p))

#%% # -------------------- Step 3：用 CLAHE 文件做 VALIS 对齐（连续切片：imgs_ordered=True） --------------------
# 设置 REFERENCE_SLIDE_NAME 为实际文件名（带扩展）；否则用 None 自动选中间图
reference_img_f = None
if REFERENCE_SLIDE_NAME:
    maybe = VSI_DIR / REFERENCE_SLIDE_NAME
    if maybe.exists():
        # 对应 clahe 文件名
        reference_img_f = str(TMP_DIR / f"clahe_{maybe.stem}.tif")
    else:
        print("指定的 REFERENCE_SLIDE_NAME 未找到，使用默认")

print("Starting VALIS registration (using CLAHE padded DAPI files)...")
registrar = registration.Valis(str(TMP_DIR), 
                               str(OUT_ROOT),
                               img_list = None,
                               imgs_ordered = True,
                               reference_img_f = reference_img_f,
                               align_to_reference = True,
                               micro_rigid_registrar_cls = MicroRigidRegistrar,
                               max_processed_image_dim_px = MAX_PROCESSED_IMAGE_DIM_PX,
                               max_non_rigid_registration_dim_px = MAX_NON_RIGID_REGISTRATION_DIM_PX,
                               norm_method = 'img_stats')

rigid_registrar, non_rigid_registrar, error_df = registrar.register()
print("Rigid/non-rigid registration done. Error summary:")
print(error_df.head())

# micro-registration
try:
    micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=2000, align_to_reference=True)
    print("Micro-registration done.")
except Exception as e:
    print("Micro-registration skipped/failed:", e)


#%% # -------------------- Step 4：把变换应用到原始 multi-channel padded 图像，并保存为 OME-TIFF --------------------
# 生成映射：registrar.slide_dict 的 key 通常为 valtils.get_name(src_f)
# slide_obj.src_f -> 名称 -> 对应 padded 原始文件名 padded_<name>.tif
print("Applying transforms to original multi-channel images and saving...")

# get reference slide object for pixel size meta
ref_slide = registrar.get_ref_slide()

# iterate over slides
for slide_name, slide_obj in registrar.slide_dict.items():
    try:
        orig_padded = padded_orig_for_slideobj(slide_obj)
        if orig_padded is None:
            print("Cannot find original padded file for", slide_name)
            continue
        # load original padded (multi-channel) pyvips image
        orig_v = pyvips.Image.new_from_file(orig_padded, access="random")
        # apply warp (non_rigid True/False 视需要)
        warped_v = slide_obj.warp_img(img=orig_v, non_rigid=False, crop='reference')
        # 构造 ome-xml
        ome_xml_obj = slide_io.create_ome_xml(shape_xyzct=slide_io.get_shape_xyzct((warped_v.width, warped_v.height), warped_v.bands),
                                              bf_dtype=slide_io.vips2bf_dtype(warped_v.format),
                                              is_rgb=False,
                                              pixel_physical_size_xyu=ref_slide.reader.metadata.pixel_physical_size_xyu,
                                              channel_names=['DAPI', 'MAP2', 'CD31', 'LYVE1'],
                                              colormap=None
        )
        ome_xml = ome_xml_obj.to_xml()
        out_path = WARPED_DIR / (slide_name + ".ome.tiff")
        # tile_wh 自动
        out_shape_wh = (warped_v.width, warped_v.height)
        tile_wh = slide_io.get_tile_wh(reader=slide_obj.reader, 
                                           level=0, 
                                           out_shape_wh=out_shape_wh)
        # 保存（使用 valis 的保存工具）
        slide_io.save_ome_tiff(warped_v, 
                               dst_f=str(out_path), 
                               ome_xml=ome_xml,
                               tile_wh=tile_wh, 
                               compression='lzw', 
                               Q=100, 
                               pyramid=True)
        print("Saved warped:", out_path)
    except Exception as e:
        print("Warp/save failed for", slide_name, ":", e)


#%% # -------------------- Step 5：按通道单独保存 (提取 CD31 单通道) --------------------
# 使用 pyvips 直接从大 OME-TIFF 文件中提取通道，避免内存问题
SAVE_SINGLE_CHANNEL = True
CH_TO_SAVE = 2  # CD31 index (0:DAPI, 1:MAP2, 2:CD31, 3:LYVE1)
CHANNEL_NAMES = ['DAPI', 'MAP2', 'CD31', 'LYVE1']
CH_NAME = CHANNEL_NAMES[CH_TO_SAVE] if CH_TO_SAVE < len(CHANNEL_NAMES) else f"Ch_{CH_TO_SAVE}"

if SAVE_SINGLE_CHANNEL:
    print(f"\nExtracting and saving single channel: {CH_NAME} (Index {CH_TO_SAVE})")
    
    # 使用一个新的、清晰的目录
    out_chan_dir = OUT_ROOT / "warped_single_channel" 
    out_chan_dir.mkdir(parents=True, exist_ok=True)

    # 设置保存 BigTiff 的瓦片大小
    TILING_WH_SINGLE = (1024, 1024)

    # 循环读取 Step 4 生成的合并 OME-TIFF 文件
    for ome_path in WARPED_DIR.glob("*.ome.tiff"): # 从 MERGED_DIR 加载
        print("-" * 30)
        print(f"Processing merged file: {ome_path.name}")
        try:
            # 1. 使用 pyvips 随机访问模式加载大文件 (高效且低内存占用)
            merged_v = pyvips.Image.new_from_file(str(ome_path), access="random")
            
            # 检查通道数是否正确
            if merged_v.bands <= CH_TO_SAVE:
                print(f"  [FAILURE] File {ome_path.name} only has {merged_v.bands} bands. Cannot extract index {CH_TO_SAVE}.")
                continue

            # 2. 提取所需的单通道
            channel_v = merged_v.extract_band(CH_TO_SAVE)
            
            # 3. 构造输出路径
            slide_name = ome_path.stem.replace("warped_merged_", "")
            out_path = out_chan_dir / f"warped_{slide_name}_{CH_NAME}.tif"
            
            # 4. 使用 pyvips 直接保存为 BigTiff (TIF)，带金字塔和 LZW 压缩
            # pyvips 默认保留原始 uint16 数据类型，无需 exposure.rescale_intensity
            channel_v.write_to_file(
                str(out_path), 
                tile=True, 
                tile_width=TILING_WH_SINGLE[0], 
                tile_height=TILING_WH_SINGLE[1], 
                pyramid=True, 
                compression='lzw', 
                bigtiff=True
            )
            print(f"  [SUCCESS] Saved single channel {CH_NAME}: {out_path.name}")
            
        except Exception as e:
            print(f"  [FAILURE] Extraction/save failed for {ome_path.name} : {e}")

print("\n" + "="*60)
print("SINGLE CHANNEL EXTRACTION COMPLETE!")
print("="*60)

#%% # -------------------- 清理临时文件 --------------------
if CLEANUP_TMP:
    try:
        shutil.rmtree(str(TMP_DIR))
        print("Cleaned up tmp dir:", TMP_DIR)
    except Exception as e:
        print("Cleanup tmp failed:", e)

#%% # -------------------- kill JVM -------------------- 
try:
    registration.kill_jvm()
except Exception:
    pass

print("Pipeline finished.")

