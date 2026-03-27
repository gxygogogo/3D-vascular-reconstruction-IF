import tifffile
import numpy as np
import os

def convert_btf_to_tif(btf_path, tif_path, channels='ALL', compress=True):
    """
    将 btf 转换为 tif，支持通道选择。
    
    参数:
        btf_path (str): 输入 .btf 文件路径
        tif_path (str): 输出 .tif 文件路径
        channels (str, int, list): 
            - 'ALL': 保存所有通道 (合并)
            - 0, 1, 2...: 保存指定索引的单通道
            - [0, 2]: 保存指定的通道子集
        compress (bool): 是否使用 zlib 压缩 (推荐 True 以节省空间)
    """
    print(f"正在读取: {btf_path}")
    
    try:
        # 1. 读取 BTF 文件
        # tifffile 会自动读取最大分辨率层
        img = tifffile.imread(btf_path)
        print(f"  -> 原始数据形状: {img.shape}, 类型: {img.dtype}")
        
        # 2. 处理通道选择逻辑
        data_to_save = img
        
        if channels != 'ALL':
            # 尝试猜测哪个维度是 Channel (通常 Channel 维度较小，且不在最后两维)
            # 常见的 shape: (C, Y, X) 或 (Z, C, Y, X) 或 (C, Z, Y, X)
            # 我们假设 Channel 维度通常 < 100，且肯定不是 Y 或 X
            
            ndim = img.ndim
            if ndim == 2:
                # (Y, X) - 没有通道可分
                print("警告: 图像是单通道 (2D)，忽略通道选择参数，直接保存。")
            
            else:
                # 简单启发式策略：假设第 0 维是通道 (C, ...) 
                # 大多数 tifffile 读取的多通道数据，C 都在第 0 维
                # 如果第 0 维很大(比如 > 100)，可能是 Z 轴，这需要具体情况具体分析
                # 这里默认按 (C, ...) 处理
                
                print(f"  -> 正在提取通道: {channels}")
                
                # 处理单个整数的情况
                if isinstance(channels, int):
                    # 提取单通道，降维
                    data_to_save = img[channels, ...] 
                
                # 处理列表的情况 [0, 2]
                elif isinstance(channels, (list, tuple)):
                    # 提取子集，保持维度
                    data_to_save = img[list(channels), ...]
                    
                print(f"  -> 提取后数据形状: {data_to_save.shape}")

        # 3. 保存
        print(f"正在保存为: {tif_path}")
        
        # 自动创建父文件夹
        os.makedirs(os.path.dirname(os.path.abspath(tif_path)), exist_ok=True)
        
        if compress:
            # compression='zlib' 是最通用的无损压缩
            tifffile.imwrite(tif_path, data_to_save, compression='zlib')
        else:
            tifffile.imwrite(tif_path, data_to_save)
            
        print("转换完成！")
        
    except IndexError:
        print(f"错误: 通道索引 {channels} 超出范围。原图形状为 {img.shape}。")
    except Exception as e:
        print(f"转换失败: {e}")

#%% ================= 使用示例 =================
sample_name = "ETPSKO3_CD31"
input_btf = f"/public3/Xinyu/3D_tissue/IF/Vascular_stat/btf/{sample_name}.btf"
output_tif = f"/public3/Xinyu/3D_tissue/IF/Vascular_stat/CD31_tif/{sample_name}_1.tif"

# 保存合并的所有通道 (默认)
# convert_btf_to_tif(input_btf, output_tif, channels='ALL', compress = False)

# 仅提取第 1 个通道 (通常是 DAPI)
# convert_btf_to_tif(input_btf, "output_dapi.tif", channels=0)

# 仅提取第 3 个通道 (例如 CD31)
convert_btf_to_tif(input_btf, output_tif, channels = 1, compress = False)

# 提取第 1 和第 3 通道组合
# convert_btf_to_tif(input_btf, "output_subset.tif", channels=[0, 2])

