import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 替换为你的 ADI 数据路径
adi_path = "./data/training/ADI" 

print(f"正在扫描 {adi_path} 中的异常值...")
files = os.listdir(adi_path)

for f in tqdm(files):
    if not f.endswith(('.png', '.jpg')): continue
    
    path = os.path.join(adi_path, f)
    try:
        # 读取图片并转为 numpy 数组
        img = np.array(Image.open(path))
        
        # 检查是否包含 Inf 或 NaN
        if np.isnan(img).any() or np.isinf(img).any():
            print(f"❌ 发现坏文件: {f} (包含 NaN 或 Inf)")
        
        # 检查是否全黑或全白 (可选)
        if img.max() == 0:
            print(f"⚠️ 发现全黑文件: {f}")
            
    except Exception as e:
        print(f"❌ 无法读取文件: {f}, 错误: {e}")

print("扫描完成。")