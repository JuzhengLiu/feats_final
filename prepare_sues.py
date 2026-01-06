import os
import shutil
from tqdm import tqdm

# 配置路径
# 原始解压路径
SOURCE_ROOT = '/root/autodl-tmp/SUES-200-512x512'
# 目标标准路径 (脚本将把数据整理到这里)
TARGET_ROOT = '/root/autodl-tmp/SUES-200-Standard'

# 原始数据子文件夹名
DRONE_FOLDER = 'drone_view_512'
SAT_FOLDER = 'satellite-view'

# 训练集索引 (来自 indexs.yaml)
TRAIN_INDEXES = [
    '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0009', '0010', '0011', 
    '0013', '0015', '0016', '0020', '0021', '0024', '0025', '0026', '0027', '0028', 
    '0029', '0030', '0031', '0032', '0033', '0034', '0036', '0037', '0039', '0043', 
    '0044', '0045', '0051', '0055', '0056', '0057', '0058', '0060', '0064', '0066', 
    '0067', '0069', '0070', '0073', '0074', '0076', '0077', '0080', '0081', '0082', 
    '0084', '0085', '0086', '0087', '0088', '0089', '0090', '0091', '0094', '0096', 
    '0097', '0100', '0101', '0102', '0103', '0104', '0105', '0106', '0113', '0116', 
    '0118', '0119', '0120', '0123', '0125', '0126', '0127', '0129', '0132', '0134', 
    '0137', '0139', '0141', '0142', '0145', '0146', '0147', '0148', '0149', '0151', 
    '0152', '0155', '0156', '0157', '0158', '0159', '0160', '0163', '0166', '0167', 
    '0168', '0169', '0171', '0174', '0179', '0180', '0182', '0183', '0184', '0185', 
    '0186', '0187', '0191', '0192', '0193', '0196', '0197', '0198', '0199', '0200'
]

# 高度列表
HEIGHTS = ['150', '200', '250', '300']

# 全部索引 0001-0200
ALL_INDEXES = ["{:0>4d}".format(i+1) for i in range(200)]

# 测试集索引 (全部 - 训练)
TEST_INDEXES = [i for i in ALL_INDEXES if i not in TRAIN_INDEXES]

def safe_copy(src, dst):
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        # 仅在找不到源文件时打印，避免刷屏
        # print(f"Warning: Source not found {src}")
        pass

def main():
    print("Start creating SUES-200 Standard Dataset...")
    print(f"Source: {SOURCE_ROOT}")
    print(f"Target: {TARGET_ROOT}")
    
    # 1. 准备 Training 数据
    # 目标结构: Training/{height}/drone|satellite/{index}
    print("\nProcessing Training Set...")
    
    for idx in tqdm(TRAIN_INDEXES, desc="Training Indexes"):
        # 卫星图源路径 (假设卫星图结构为 satellite-view/{index})
        src_sat = os.path.join(SOURCE_ROOT, SAT_FOLDER, idx)
        
        for h in HEIGHTS:
            # 无人机源路径: drone_view_512/{index}/{height}
            src_dro = os.path.join(SOURCE_ROOT, DRONE_FOLDER, idx, h)
            
            # 目标路径
            dst_root = os.path.join(TARGET_ROOT, "Training", h)
            dst_dro = os.path.join(dst_root, "drone", idx)
            dst_sat = os.path.join(dst_root, "satellite", idx)
            
            # 执行复制
            safe_copy(src_dro, dst_dro)
            safe_copy(src_sat, dst_sat) # 每个高度文件夹下都放一份卫星图，保持对齐

    # 2. 准备 Testing 数据
    # 目标结构: Testing/{height}/query_drone|gallery_satellite|...
    print("\nProcessing Testing Set...")
    
    for h in HEIGHTS:
        dst_test_root = os.path.join(TARGET_ROOT, "Testing", h)
        
        # === D2S 模式 ===
        # Query: Drone (仅 Test Indexes)
        # Gallery: Satellite (All Indexes, SUES 标准通常用全库作为 Gallery)
        
        # 复制 Query Drone
        for idx in tqdm(TEST_INDEXES, desc=f"Test D2S Query ({h})", leave=False):
            src_dro = os.path.join(SOURCE_ROOT, DRONE_FOLDER, idx, h)
            dst_q_dro = os.path.join(dst_test_root, "query_drone", idx)
            safe_copy(src_dro, dst_q_dro)
            
        # 复制 Gallery Satellite (All Indexes)
        for idx in tqdm(ALL_INDEXES, desc=f"Test D2S Gallery ({h})", leave=False):
            src_sat = os.path.join(SOURCE_ROOT, SAT_FOLDER, idx)
            dst_g_sat = os.path.join(dst_test_root, "gallery_satellite", idx)
            safe_copy(src_sat, dst_g_sat)

        # === S2D 模式 ===
        # Query: Satellite (仅 Test Indexes)
        # Gallery: Drone (All Indexes)
        
        # 复制 Query Satellite
        for idx in tqdm(TEST_INDEXES, desc=f"Test S2D Query ({h})", leave=False):
            src_sat = os.path.join(SOURCE_ROOT, SAT_FOLDER, idx)
            dst_q_sat = os.path.join(dst_test_root, "query_satellite", idx)
            safe_copy(src_sat, dst_q_sat)
            
        # 复制 Gallery Drone
        for idx in tqdm(ALL_INDEXES, desc=f"Test S2D Gallery ({h})", leave=False):
            src_dro = os.path.join(SOURCE_ROOT, DRONE_FOLDER, idx, h)
            dst_g_dro = os.path.join(dst_test_root, "gallery_drone", idx)
            safe_copy(src_dro, dst_g_dro)

    print("\nData Preparation Completed!")
    print(f"Dataset stored at: {TARGET_ROOT}")

if __name__ == '__main__':
    main()