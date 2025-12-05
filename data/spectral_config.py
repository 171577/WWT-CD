"""
光谱增强配置文件
用于快速调整光谱预处理的各项参数
"""

# ==================== 光谱增强配置 ====================

# 1. 基础开关
ENABLE_SPECTRAL_AUGMENTATION = True      # 是否启用光谱增强
ENABLE_MULTISCALE_PROCESSING = True      # 是否启用多尺度处理
ENABLE_EDGE_ENHANCEMENT = True           # 是否启用边界锐化
ENABLE_ADAPTIVE_CONTRAST = True          # 是否启用自适应对比度

# 2. CLAHE参数 (对比度受限自适应直方图均衡化)
CLAHE_CLIP_LIMIT = 2.0                   # 裁剪限制 (1.0-4.0, 越大越强)
CLAHE_TILE_SIZE = 8                      # 瓦片大小 (8, 16, 32)

# 3. 多尺度处理参数
MULTISCALE_LEVELS = 3                    # 多尺度层数 (2-4)
MULTISCALE_WEIGHTS = [0.5, 0.3, 0.2]    # 各层权重 (需要与MULTISCALE_LEVELS匹配)

# 4. 边界锐化参数
EDGE_ENHANCE_STRENGTH = 1.5              # 边界增强强度 (0.5-2.5)
LAPLACIAN_KERNEL_SIZE = 3                # Laplacian核大小 (3, 5)

# 5. 自适应对比度参数
ADAPTIVE_CONTRAST_FACTOR = 0.5           # 对比度调整因子 (0.3-0.7)
CONTRAST_STRETCH_RANGE = (0.02, 0.98)   # 对比度拉伸范围 (百分位数)

# 6. 应用概率
SPECTRAL_APPLY_PROBABILITY = 0.8         # 光谱增强应用概率 (0.5-1.0)

# 7. 颜色空间选择
COLOR_SPACE = 'LAB'                      # 'RGB', 'LAB', 'HSV'
USE_HISTOGRAM_EQUALIZATION = True        # 是否使用直方图均衡化

# ==================== 预设配置 ====================

# 保守配置 (轻微增强，适合已经很好的数据)
PRESET_CONSERVATIVE = {
    'enable_spectral': True,
    'enable_multiscale': True,
    'enable_edge_enhance': False,
    'enable_adaptive_contrast': True,
    'clahe_clip_limit': 1.5,
    'clahe_tile_size': 8,
    'edge_enhance_strength': 0.5,
    'spectral_probability': 0.5,
}

# 标准配置 (中等增强，推荐用于大多数情况)
PRESET_STANDARD = {
    'enable_spectral': True,
    'enable_multiscale': True,
    'enable_edge_enhance': True,
    'enable_adaptive_contrast': True,
    'clahe_clip_limit': 2.0,
    'clahe_tile_size': 8,
    'edge_enhance_strength': 1.5,
    'spectral_probability': 0.8,
}

# 激进配置 (强增强，适合数据质量一般的情况)
PRESET_AGGRESSIVE = {
    'enable_spectral': True,
    'enable_multiscale': True,
    'enable_edge_enhance': True,
    'enable_adaptive_contrast': True,
    'clahe_clip_limit': 3.0,
    'clahe_tile_size': 16,
    'edge_enhance_strength': 2.0,
    'spectral_probability': 1.0,
}

# 最小配置 (仅基础增强，用于快速实验)
PRESET_MINIMAL = {
    'enable_spectral': True,
    'enable_multiscale': False,
    'enable_edge_enhance': False,
    'enable_adaptive_contrast': False,
    'clahe_clip_limit': 1.0,
    'clahe_tile_size': 8,
    'edge_enhance_strength': 0.0,
    'spectral_probability': 0.5,
}

# ==================== 使用方法 ====================
"""
在 cd_dataset.py 中使用:

from data.spectral_config import PRESET_STANDARD

# 方法1: 使用预设配置
self.transform = transforms.Compose([
    TransformsStandard(**PRESET_STANDARD)
])

# 方法2: 自定义配置
self.transform = transforms.Compose([
    TransformsStandard(
        enable_spectral=True,
        enable_multiscale=True,
        enable_edge_enhance=True,
        spectral_probability=0.8
    )
])

# 方法3: 快速切换
# 在训练前修改此文件中的参数，然后重新运行训练脚本
"""
