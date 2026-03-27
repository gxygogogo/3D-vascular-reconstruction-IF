#%% 导入库
from valis import registration, valtils, micro_rigid_registrar, preprocessing, slide_io
from skimage import filters
import os
import re
import numpy as np


#%% 使用DAPIGetter类进行ROI微对齐
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

sample_region = 'HXJ-00003-18426-17191'

## 预设路径
slide_src_dir = f"/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/{sample_region}"
results_dst_dir = f"/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/{sample_region}/Micro-registration"
registered_slide_dst_dir = f"/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/{sample_region}/Micro-registration/micro_registered_slides"

os.makedirs(results_dst_dir, exist_ok = True)
os.makedirs(registered_slide_dst_dir, exist_ok = True)

# sorted_img_f_list = ['HJZ_1-DAPI-cropped', 'HJZ_2-DAPI-cropped', 'HJZ_3-DAPI-cropped', 'HJZ_5-DAPI-cropped', 'HJZ_6-DAPI-cropped']


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
                               #max_processed_image_dim_px = 2000,
                               #max_non_rigid_registration_dim_px = 1500,
                               reference_img_f="HXJ_2-00003-18426-17191.ome.tiff",
                               norm_method = 'img_stats')

rigid_registrar, non_rigid_registrar, error_df = registrar.register()

registrar.warp_and_save_slides(registered_slide_dst_dir, crop="reference", interp_method = 'bicubic', level = 0)
