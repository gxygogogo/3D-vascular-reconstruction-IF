############################################# lmx
'''
Code.3D tissue project.IF血管重建流程.2.1.Cellpose细胞分割 的 Docstring

/public1/yuchen/software/miniconda3/envs/img/bin/ipython
'''

import numpy as np
from cellpose import models, io
import cv2
import tifffile
import gc
import os
import re

# models.CellposeModel(pretrained_model='/full/path/to/model')

data_dir = '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/micro_registered_slides_multichannel'
to_dir = '/public3/Xinyu/3D_tissue/IF/02.mIHC_ROI_RigidMicro_registration/ROI-00004-07662-12501/Micro-registration/micro_registered_slides_cellpose'
# sample_list = ['图像_M89344-CD31', '图像_M89492-CD31', '图像_M74649-6E-CD31']['图像_M93715-6A-CD31']
# sample_list = ['图像_M47525-8D-CD31-HIGH3', '图像_M48618-6D-CD31-HIGH3', '图像_M52897-CD31-HIGH3', '图像_M53489-CD31-HIGH3']
sample_list = os.listdir(data_dir)

remove_list = ['HJZ_68-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_21-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_106-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_26-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_104-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_98-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_48-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_53-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_34-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_42-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_66-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_73-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_43-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_28-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_59-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_67-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_17-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_78-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_18-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_60-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_19-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_72-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_77-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_61-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_83-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_84-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_52-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_51-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_63-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_16-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_2-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_85-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_49-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_31-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_24-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_76-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_92-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_35-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_81-00004-07662-12501_DAPI_CD31_LYVE1_enhanced.tif']

remove_list = ['HJZ_23-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_25-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_77-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_92-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_16-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_26-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_55-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_70-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_2-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_24-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_19-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_74-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif',
               'HJZ_82-00004-02772-17487_DAPI_CD31_LYVE1_enhanced.tif']

sample_list2 =  [x for x in sample_list if x not in remove_list]

pat = re.compile(r'HJZ_\d+-\d+-\d+-\d+')

for sample in sample_list2:
    print(f'Processing {sample}')

    m = pat.search(sample)
    base = m.group(0)

    img = tifffile.imread(f"{data_dir}/{base}_DAPI_CD31_LYVE1_enhanced.tif")

    img = img.transpose(1, 2, 0)

    while img.shape[0] * img.shape[1] >= 1200000000:
        img = cv2.resize(img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

    tifffile.imwrite(f'{data_dir}/{base}_MicroRig.resize.btf', img)

    img_nuc = img[:, :, 0]

    tifffile.imwrite(f'{to_dir}/{base}_channel_dapi_for_cellpose.tif', img_nuc)
    img = io.imread(f'{to_dir}/{base}_channel_dapi_for_cellpose.tif')
    # 初始化模型（nuclei 模式针对 DAPI）
    model = models.Cellpose(model_type='nuclei')
    # 推理：channels=(0, 0) 表示灰度图（DAPI 单通道）
    print('########## cellpose')
    masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0, 0])

    np.save(f'{to_dir}/{base}.dapi.masks.npy', masks)
    
    del img, img_nuc, model, masks, flows, styles, diams
    gc.collect()

