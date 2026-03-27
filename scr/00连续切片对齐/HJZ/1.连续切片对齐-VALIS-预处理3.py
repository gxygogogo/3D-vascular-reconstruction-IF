#%% 导入必要库
'''
## 进入docker容器内部
docker run -it --memory=200g \
    -v "$HOME:$HOME" \
    -v /public3/Xinyu:/public3/Xinyu \
    cdgatenbee/valis-wsi bash
'''
import os
from pathlib import Path
import numpy as np
import pyvips
import cv2
from skimage import exposure
from valis import valtils, slide_io, registration
from valis.micro_rigid_registrar import MicroRigidRegistrar
import shutil
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
    print("tqdm.notebook not found. Using tqdm instead.")

#%% 函数定义
def organize_channels_automatically(base_dir, target_dapi_dir=None, target_all_channels_dir=None):
    """
    功能: 自动按通道分类你的 IF 图像 (更新支持 CH00, CH01, CH02, CH03)
    Args:
        base_dir: 原始图像文件的目录
        target_dapi_dir: 如果给定路径, 则把匹配的dapi通道转移到这里; 若为 None, 函数则会在 base_dir 下创建 dapi 子目录
        target_all_channels_dir: 若给定, 则把匹配其他通道 (ch01/ch02/ch03) 的文件移动到这里；若为 None, 函数会在 base_dir 下创建 all_channels/ 子目录
    """
    base_path = Path(base_dir)

    # Create target directories if not specified
    if target_dapi_dir is None:
        target_dapi_dir = base_path / "dapi"
    else:
        target_dapi_dir = Path(target_dapi_dir)

    if target_all_channels_dir is None:
        target_all_channels_dir = base_path / "all_channels"
    else:
        target_all_channels_dir = Path(target_all_channels_dir)

    # Create directories if they don't exist
    target_dapi_dir.mkdir(parents=True, exist_ok=True)
    target_all_channels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Organizing files from: {base_path}")
    print(f"DAPI (ch00) files will go to: {target_dapi_dir}")
    print(f"Other channels (ch01, ch02, ch03) will go to: {target_all_channels_dir}")
    print("-" * 60)

    # Get all files in the base directory
    all_files = [f for f in base_path.iterdir() if f.is_file()]

    moved_files = {"dapi": [], "other_channels": [], "unrecognized": []}

    for file_path in all_files:
        filename = file_path.name.lower()

        # Check for channel patterns
        if "_ch00" in filename or "_merged_ch00" in filename:
            # This is a DAPI file
            target_path = target_dapi_dir / file_path.name
            shutil.move(str(file_path), str(target_path))
            moved_files["dapi"].append(file_path.name)
            # print(f"📁 DAPI: {file_path.name}") # 减少输出

        # *** 关键修改：添加 ch03 的检查 ***
        elif "_ch01" in filename or "_ch02" in filename or "_ch03" in filename or \
             "_merged_ch01" in filename or "_merged_ch02" in filename or "_merged_ch03" in filename:
            # This is another channel (MAP2, CD31, LYVE1)
            target_path = target_all_channels_dir / file_path.name
            shutil.move(str(file_path), str(target_path))
            moved_files["other_channels"].append(file_path.name)
            # print(f"🔬 Channel: {file_path.name}") # 减少输出

        else:
            # Unrecognized file pattern
            moved_files["unrecognized"].append(file_path.name)
            # print(f"❓ Unrecognized: {file_path.name}") # 减少输出

    # Summary
    print("\n" + "="*60)
    print("ORGANIZATION COMPLETE!")
    print("="*60)
    print(f"DAPI files (ch00) moved: {len(moved_files['dapi'])}")
    print(f"Other channel files (ch01/02/03) moved: {len(moved_files['other_channels'])}")
    print(f"Unrecognized files: {len(moved_files['unrecognized'])}")
    
    # ... (其余总结部分保持不变)
    if moved_files["unrecognized"]:
        print("\nUnrecognized files (not moved):")
        for filename in moved_files["unrecognized"]:
            print(f"  - {filename}")
            
    return moved_files

def organize_multiple_rounds(parent_dir, rounds=None):
    '''
    功能: 对多个轮次自动运行上述通道分类
    Args:
        parent_dir: 包含多个 round 的子文件夹的父目录
        rounds: round 名称列表
    '''

    parent_path = Path(parent_dir)

    if rounds is None:
        # Auto-detect round folders
        rounds = [d.name for d in parent_path.iterdir() if d.is_dir() and d.name.startswith('R')]
        rounds.sort()  # Sort to process in order

    print(f"Found rounds: {rounds}")

    for round_name in rounds:
        round_path = parent_path / round_name
        if round_path.exists():
            print(f"\n{'='*20} PROCESSING {round_name} {'='*20}")
            organize_channels_automatically(round_path)
        else:
            print(f"Round folder not found: {round_path}")


def get_round_name(src_f):
    '''
    功能: 从文件名中提取切片名称 HJZ_XX
    Args:
        src_f: 输入文件路径或者文件名
    '''
    img_name = valtils.get_name(src_f)
    img_name_lower = img_name.lower()
    
    # *** 关键修改：匹配 HJZ_XX 模式 ***
    # 查找 "hjz_" 后跟两位数字的模式
    match = re.search(r'(hjz_\d{2})', img_name_lower)
    
    if match:
        # 返回匹配到的切片ID，例如 'hjz_01'
        return match.group(1)
    
    # 如果没有匹配到 HJZ_XX，则使用整个文件名的一部分作为 ID（作为回退）
    print(f"Warning: Could not find HJZ_XX pattern in {img_name}. Using default naming.")
    
    # 沿用之前的逻辑，提取通道标记前的内容
    match_ch = re.search(r'(_ch\d\d)', img_name_lower)
    if match_ch:
        return img_name_lower[:match_ch.start()]
        
    return img_name_lower # 最终回退

def get_channel_number(src_f):
    '''
    功能: 提取通道信息 (更新支持 CH03)
    Args:
        src_f: 输入文件路径
    '''
    img_name = os.path.basename(src_f).lower()
    if "_ch00" in img_name:
        return 0
    elif "_ch01" in img_name:
        return 1
    elif "_ch02" in img_name:
        return 2
    # *** 关键修改：添加 ch03 ***
    elif "_ch03" in img_name:
        return 3
    else:
        return None

def pad_image(img, target_width, target_height):
    '''
    功能: 把图像补充到统一大小
    Args:
        img: 要 pad 的图像对象
        target_width: 目标画布宽度(像素)
        target_height: 目标画布高度(像素)
    '''
    pad_width = max(0, target_width - img.width)
    pad_height = max(0, target_height - img.height)

    if pad_width > 0 or pad_height > 0:
        pad_left = pad_width // 2
        pad_top = pad_height // 2
        padded = img.embed(pad_left, pad_top, target_width, target_height, extend="black")
        return padded
    return img

def apply_clahe_normalization(img_array, clip_limit=2.0, tile_grid_size=(16,16)):
    '''
    功能: 对图像做CLAHE增强
    Args:
        img_array: 单通道图像数组
        clip_limit: CLAHE的clip limit(对比度限制)
        tile_grid_size: CLAHE 的 tileGridSize
    '''
    if img_array.dtype != np.uint8:
        # Convert to uint8 for CLAHE
        img_normalized = exposure.rescale_intensity(img_array, out_range=(0, 255)).astype(np.uint8)
    else:
        img_normalized = img_array.copy()

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(img_normalized)

    return clahe_img

def create_clahe_padded_image(img_path, target_width, target_height, output_path, clip_limit=2.0, tile_grid_size=(16, 16)):
    '''
    功能: 整体流程封装
    Args:
        img_path: 原始图像文件路径
        target_width: pad后的宽度
        target_height: pad后的高度
        output_path: CLAHE结果输出路径
        clipi_limit: CLAHE的clip limit(对比度限制)
        tile_grid_size: CLAHE 的 tileGridSize
    '''
    try:
        # Load original image
        img = pyvips.Image.new_from_file(img_path)
        # Step 1: Pad the image
        padded_img = pad_image(img, target_width, target_height)

        # Step 2: Convert padded image to numpy for CLAHE processing
        img_array = np.ndarray(buffer=padded_img.write_to_memory(),
                               dtype=np.uint8 if padded_img.format == 'uchar' else np.uint16,
                               shape=[padded_img.height, padded_img.width, padded_img.bands])

        # Step 3: Apply CLAHE to first channel (DAPI)
        if padded_img.bands == 1:
            clahe_array = apply_clahe_normalization(img_array[:, :, 0], clip_limit, tile_grid_size)
            clahe_array = clahe_array[:, :, np.newaxis]
        else:
            # For multi-channel, apply CLAHE to first channel only
            clahe_array = img_array.copy()
            clahe_array[:, :, 0] = apply_clahe_normalization(img_array[:, :, 0], clip_limit, tile_grid_size)

        # Step 4: Convert back to pyvips
        clahe_padded_img = pyvips.Image.new_from_memory(clahe_array.data,
                                                        clahe_array.shape[1],
                                                        clahe_array.shape[0],
                                                        clahe_array.shape[2],
                                                        'uchar' if clahe_array.dtype == np.uint8 else 'ushort')

        # Step 5: Save CLAHE-processed padded image
        clahe_padded_img.write_to_file(output_path)
        return output_path
    
    except Exception as e:
        print(f"Error creating CLAHE-padded image for {img_path}: {str(e)}")
        return None

def get_ome_xml(warped_slide, reference_slide, channel_names = None):
    '''
    功能: 给 warp 后图像构建新的 OME-XML 元数据
    Args:
        warped_slide: 已经warp出来的pyvips图像
        reference_slide: 用于从reference提取像素物理大小
        channel_names: 写入OME-XML的通道名字
    '''
    ref_meta = reference_slide.reader.metadata
    bf_dtype = slide_io.vips2bf_dtype(warped_slide.format)
    out_xyczt = slide_io.get_shape_xyzct((warped_slide.width, warped_slide.height), warped_slide.bands)
    ome_xml_obj = slide_io.create_ome_xml(shape_xyzct = out_xyczt,
                                          bf_dtype = bf_dtype,
                                          is_rgb = False,
                                          pixel_physical_size_xyu = ref_meta.pixel_physical_size_xyu,
                                          channel_names = channel_names,
                                          colormap = None
                                          )
    return ome_xml_obj.to_xml()


#%% 数据预处理流程：先PAD， 后CLAHE， 只针对DAPI
channel_names = ["DAPI (CH00)", "MAP2 (CH01)", "CD31 (CH02)", "LYVE1 (CH03)"]

# get_ome_xml(..., channel_names=channel_names) # 如果您在后续需要重新合并通道
# !!!!! 更改为您实际的图像文件夹路径 !!!!!
base_dir = "/public3/Xinyu/3D_tissue/IF/mIHC_data_all/" 
# ----------------------------------------

# 1. 自动组织文件：将 DAPI (ch00) 和其他通道分开
print("Organizing channel files...")
moved_files = organize_channels_automatically(base_dir)

# 定义主要目录
slide_src_dir = Path(base_dir) / "dapi" # 组织后的 DAPI 文件夹
path_to_all_channels = Path(base_dir) / "all_channels" # 组织后的其他通道文件夹

# 结果目录
results_dst_dir = Path(base_dir) / "registration"
results_dst_dir.mkdir(parents=True, exist_ok=True)
registered_slide_dst_dir = results_dst_dir / "registered_slides"
registered_slide_dst_dir.mkdir(parents=True, exist_ok=True)

# !!! 连续切片配准中，参考切片通常是“第一张”或“中间某张”！！！
# 请根据您的实际情况指定一个参考切片文件名
# 假设我们选择 DAPI 文件夹中名字排序后的第一张作为参考
dapi_img_files = sorted([f for f in slide_src_dir.iterdir() if f.is_file()])
if not dapi_img_files:
    raise FileNotFoundError("DAPI directory is empty. Check your file organization.")
    
reference_slide_path = dapi_img_files[0]
reference_slide_name = reference_slide_path.name
reference_slide = str(reference_slide_path)
print(f"Reference slide chosen: {reference_slide_name}")
# -----------------------------------------------------------

# 输出目录
src_dir = str(slide_src_dir)
dst_dir = str(results_dst_dir)
all_channels_dir = str(path_to_all_channels)

# pad后的DAPI图像的目录(原始强度)
padded_dir = os.path.join(dst_dir, "padded_images")
os.makedirs(padded_dir, exist_ok=True)
padded_dapi_dir = os.path.join(padded_dir, "dapi")
os.makedirs(padded_dapi_dir, exist_ok=True)
padded_all_channels_dir = os.path.join(padded_dir, "all_channels")
os.makedirs(padded_all_channels_dir, exist_ok=True)
padded_ref_dir = os.path.join(padded_dir, "reference")
os.makedirs(padded_ref_dir, exist_ok=True)

# CLAHE后的padded DAPI图像(用于配准)
clahe_padded_dir = os.path.join(dst_dir, "clahe_padded_dapi")
os.makedirs(clahe_padded_dir, exist_ok=True)

# 单独通道的坐标转换
individual_channel_dir = os.path.join(dst_dir, "warped_channels")
os.makedirs(individual_channel_dir, exist_ok=True)

# 获得所有的图像
dapi_imgs = [str(f) for f in slide_src_dir.iterdir() if f.is_file()]
all_channel_imgs = [str(f) for f in path_to_all_channels.iterdir() if f.is_file()]

print(f"Found {len(dapi_imgs)} DAPI images for registration")
print(f"Found {len(all_channel_imgs)} other channel images")


#%% 开始数据预处理
# Step1: 寻找所有图像的最大维度(宽X高)
max_width = 0
max_height = 0

print("Finding maximum dimensions across all images...")
for img_path in tqdm(dapi_imgs + all_channel_imgs, desc="Finding max dims"):
    try:
        img = pyvips.Image.new_from_file(img_path)
        max_width = max(max_width, img.width)
        max_height = max(max_height, img.height)
    except Exception as e:
        print(f"Error reading {img_path}: {str(e)}")

print(f"Maximum dimensions: {max_width} x {max_height}")

# Step2: 创建 pad 原始文件和 CLAHE 后的pad文件
## 处理参考图像 (保持不变)
ref_img = pyvips.Image.new_from_file(reference_slide)
padded_ref_img = pad_image(ref_img, max_width, max_height)
ref_basename = os.path.basename(reference_slide)
padded_reference_slide = os.path.join(padded_ref_dir, f"padded_{ref_basename}")
padded_ref_img.write_to_file(padded_reference_slide)
print(f"Padded reference: {padded_reference_slide}")

## 创建 CLAHE-padded 参考图像，用于配准 (保持不变)
clahe_padded_reference = os.path.join(clahe_padded_dir, f"clahe_padded_{ref_basename}")
create_clahe_padded_image(reference_slide, max_width, max_height, clahe_padded_reference)
print(f"CLAHE-padded reference: {clahe_padded_reference}")

## 处理 DAPI 通道图像
padded_dapi_paths = []
clahe_padded_dapi_paths = []

print("\nProcessing DAPI images (PAD and CLAHE-PAD)...")
for img_path in tqdm(dapi_imgs, desc="Processing DAPI"):
    if os.path.basename(img_path) == ref_basename:
        # Skip re-processing the reference image here
        continue 
        
    try:
        basename = os.path.basename(img_path)
        # Create padded version (original intensities)
        img = pyvips.Image.new_from_file(img_path)
        padded_img = pad_image(img, max_width, max_height)
        padded_path = os.path.join(padded_dapi_dir, f"padded_{basename}")
        padded_img.write_to_file(padded_path)
        padded_dapi_paths.append(padded_path)

        # Create CLAHE-padded version (for registration)
        clahe_padded_path = os.path.join(clahe_padded_dir, f"clahe_padded_{basename}")
        if create_clahe_padded_image(img_path, max_width, max_height, clahe_padded_path):
            clahe_padded_dapi_paths.append(clahe_padded_path)

    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")

# Add the padded reference path to the list for downstream use if needed
padded_dapi_paths.append(padded_reference_slide)
clahe_padded_dapi_paths.append(clahe_padded_reference)

## 处理所有通道图像(只有DAPI，没有CLAHE)
padded_all_channel_paths = []
print("\nProcessing other channels (PAD only)...")
for img_path in tqdm(all_channel_imgs, desc="Padding other channels"):
    try:
        img = pyvips.Image.new_from_file(img_path)
        padded_img = pad_image(img, max_width, max_height)

        basename = os.path.basename(img_path)
        padded_path = os.path.join(padded_all_channels_dir, f"padded_{basename}")
        padded_img.write_to_file(padded_path)
        padded_all_channel_paths.append(padded_path)
    except Exception as e:
        print(f"Error padding {img_path}: {str(e)}")

# Step3: 使用CLAHE-padded DAPI图像进行配准
print("\nPerforming registration with CLAHE-padded DAPI images...")
# 这里 registrar 的输入是 **整个文件夹**，Valis 会根据文件名顺序（默认）或手动指定
# 的参考图像 (clahe_padded_reference) 来进行对齐。
registrar = registration.Valis(clahe_padded_dir, # Directory with ALL CLAHE-padded DAPI images
                               dst_dir,
                               reference_img_f = clahe_padded_reference, # 指定对齐的参考切片
                               align_to_reference = True, # 所有切片都对齐到这个参考
                               micro_rigid_registrar_cls = MicroRigidRegistrar,
                               max_processed_image_dim_px = 2000,
                               max_non_rigid_registration_dim_px = 1500,
                               norm_method = 'img_stats'
                               )

# 进行配准
rigid_registrar, non_rigid_registrar, error_df = registrar.register()

## 执行微对齐
print("Performing micro-registration...")
micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px = 2000,
                                                  align_to_reference = True
                                                  )

# Step4: 应用配齐后变换矩阵到原始的 padded 图像

## 1. 变换原始的DAPI图像
dapi_warped_dir = os.path.join(individual_channel_dir, "dapi_warped")
os.makedirs(dapi_warped_dir, exist_ok=True)

# 创建 CLAHE-padded 和 Original-padded DAPI 图像之间的映射
clahe_to_original_mapping = {}
for slide_name, slide_obj in registrar.slide_dict.items():
    # Find corresponding original padded image
    # Note: slide_obj.src_f is the CLAHE-padded path
    clahe_basename = os.path.basename(slide_obj.src_f)
    original_basename = clahe_basename.replace("clahe_padded_", "padded_")
    
    # 查找原始 padded DAPI 图像的路径 (可能在 padded_dapi_dir 或 padded_ref_dir)
    original_path_dapi = Path(padded_dapi_dir) / original_basename
    original_path_ref = Path(padded_ref_dir) / original_basename
    
    if original_path_dapi.exists():
        original_path = str(original_path_dapi)
    elif original_path_ref.exists():
        original_path = str(original_path_ref)
    else:
        print(f"Could not find original padded DAPI for {slide_name}")
        continue
        
    clahe_to_original_mapping[slide_obj] = original_path

print("\nWarping original DAPI images...")
for slide_obj, original_dapi_path in tqdm(clahe_to_original_mapping.items(), desc="Warping DAPI"):
    try:
        # Load original padded DAPI (original intensities)
        original_dapi_img = pyvips.Image.new_from_file(original_dapi_path)

        # Apply transformation found using CLAHE-padded DAPI
        warped_dapi = slide_obj.warp_img(img=original_dapi_img, non_rigid=True, crop=False)

        basename = os.path.basename(original_dapi_path)
        warped_path = os.path.join(dapi_warped_dir, f"warped_{basename}")
        warped_dapi.write_to_file(warped_path)

    except Exception as e:
        print(f"Error warping original DAPI {slide_obj.name}: {str(e)}")

## 2. 变换所有其他通道到原始强度
# 连续切片配准中，关键在于将**同一切片的 DAPI 变换**应用到**该切片的所有其他通道**上。
# 建立 DAPI 切片对象（slide_obj）与其所有其他通道文件（padded_all_channel_paths）的映射。
slide_channel_mapping = {}
print("\nCreating mapping for DAPI slide object to its other channels...")
for slide_obj in registrar.slide_dict.values():
    # 1. 获取 DAPI 图像的切片名称
    # DAPI 图像的原始文件名是 slide_obj.src_f (CLAHE-padded) 去掉前缀和后缀
    clahe_basename = os.path.basename(slide_obj.src_f)
    original_dapi_basename = clahe_basename.replace("clahe_padded_", "").replace("padded_", "")
    dapi_slice_name = get_round_name(original_dapi_basename) # 使用修改后的 get_round_name

    # 2. 查找所有属于该切片名称的其他通道文件
    matching_channels = []
    for padded_path in padded_all_channel_paths:
        padded_basename = os.path.basename(padded_path).replace("padded_", "")
        channel_slice_name = get_round_name(padded_basename) # 使用修改后的 get_round_name
        
        if channel_slice_name == dapi_slice_name:
            matching_channels.append(padded_path)
            
    if matching_channels:
        valtils.sort_nicely(matching_channels)
        slide_channel_mapping[slide_obj] = matching_channels
    
    print(f"Found {len(matching_channels)} channels for slice: {dapi_slice_name}")


print("\nWarping all other channels with DAPI transformations...")
for slide_obj, channel_paths in slide_channel_mapping.items():
    dapi_slice_name = get_round_name(os.path.basename(slide_obj.src_f).replace("clahe_padded_", "").replace("padded_", ""))
    print(f"\n--- Warping channels for slice: {dapi_slice_name} ---")

    for src_f in tqdm(channel_paths, desc=f"Warping {dapi_slice_name} channels"):
        
        try:
            # Load original intensity padded channel
            channel_img = pyvips.Image.new_from_file(src_f)
            
            # Apply transformation found using CLAHE-padded DAPI
            warped_channel = slide_obj.warp_img(img=channel_img, non_rigid=True, crop=False)
            
            channel_basename = os.path.basename(src_f).replace("padded_", "")
            individual_channel_path = os.path.join(individual_channel_dir, f"warped_{channel_basename}")
            warped_channel.write_to_file(individual_channel_path)
            
        except Exception as e:
            print(f"Error processing {src_f}: {str(e)}")


#%% 结束
# Kill JVM
registration.kill_jvm()
print("\nRegistration and warping complete!")
