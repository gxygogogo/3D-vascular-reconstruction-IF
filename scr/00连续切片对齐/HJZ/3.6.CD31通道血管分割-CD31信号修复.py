import glob
import tifffile
import numpy as np
import os


def rolling_maximum_enhancement(img_dir, save_dir):
    files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    
    # 读取所有图片到内存 (如果内存够)
    # stack shape: [D, H, W]
    stack = np.array([tifffile.imread(f) for f in files]) 
    
    enhanced_stack = np.zeros_like(stack)
    
    D, H, W = stack.shape
    
    for z in range(D):
        # 边界处理
        start = max(0, z - 1)
        end = min(D, z + 2) # slice切片不包含end，所以是z+2
        
        # 取切片 [z-1, z, z+1] 的最大值
        # 这会把上下层强信号“投影”到当前弱信号层
        enhanced_stack[z] = np.max(stack[start:end], axis=0)
        
    # 保存
    for z in range(D):
        tifffile.imwrite(os.path.join(save_dir, files[z]), enhanced_stack[z])


rolling_maximum_enhancement(img_dir = '/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/signal_extraction/ROI-00004-02772-17487',
                            save_dir = '/public3/Xinyu/3D_tissue/IF/03.mIHC_vascular_segmentation/repair_signal')

