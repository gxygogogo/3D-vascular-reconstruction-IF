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
slide_src_dir = "/public3/Xinyu/3D_tissue/IF/MIF_HXJ_filter_2"
results_dst_dir = "/public3/Xinyu/3D_tissue/IF/01.mIHC_rigid_registration/MIF_HXJ_3"
registered_slide_dst_dir = "/public3/Xinyu/3D_tissue/IF/01.mIHC_rigid_registration/MIF_HXJ_3/registered_slides"

os.makedirs(results_dst_dir, exist_ok = True)
os.makedirs(registered_slide_dst_dir, exist_ok = True)

sorted_img_f_list = ['图像_HXJ_' + str(x) + '_CD31_LYVE1' for x in range(2, 102)]

#sorted_img_f_list = ['图像_HXJ_' + str(x) + '_CD31_LYVE1' for x in range(2, 102) if x != 57]

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
                               #original_img_list=sorted_img_f_list,
                               micro_rigid_registrar_cls = micro_rigid_registrar.MicroRigidRegistrar,
                               max_processed_image_dim_px = 2000,
                               max_non_rigid_registration_dim_px = 1500,
                               norm_method = 'img_stats',
                               reference_img_f="图像_HXJ_2_CD31_LYVE1.vsi",
                               align_to_reference=True)

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

registrar = registration.load_registrar('/public3/Xinyu/3D_tissue/IF/01.mIHC_rigid_registration/MIF_HXJ_2/MIF_HXJ_filter/data/MIF_HXJ_filter_registrar.pickle')

## 逐张导出
pat = re.compile(r'HXJ_\d+')
for i in sorted_img_f_list:
    print(i)
    slide_f = f'/public3/Xinyu/3D_tissue/IF/MIF_HXJ_filter/{i}.vsi'
    m = pat.search(i)
    base = m.group(0)
    slide_obj = registrar.get_slide(slide_f)
    slide_obj.warp_and_save_slide(f'{registered_slide_dst_dir}/{base}-DAPI-rigid.ome.tiff', 
                                  level = 1, 
                                  crop = 'reference',
                                  channel_names = ['DAPI', 'CD31', 'LYVE1'],
                                  interp_method = 'nearest',
                                  compression = 'lzw',
                                  pyramid = True,
                                  non_rigid = False)
