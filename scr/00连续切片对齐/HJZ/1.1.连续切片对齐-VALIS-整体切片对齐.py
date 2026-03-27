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
# img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
# min_max_size = np.min([np.max(d) for d in img_dims])
# img_areas = [np.multiply(*d) for d in img_dims]
# max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])

# micro_reg_fraction = 0.2
# micro_reg_size = np.floor(min_max_size * micro_reg_fraction).astype(int)

# Perform high resolution non-rigid registration
# micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px = micro_reg_size)

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
