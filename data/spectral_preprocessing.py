"""
光谱预处理模块 - 突出变化区域的多尺度处理
包含:
1. 光谱对比度增强
2. 多尺度差异图生成
3. 边界锐化处理
4. 自适应对比度拉伸
"""

import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import torch
import torch.nn.functional as F


class SpectralPreprocessor:
    """
    光谱预处理器 - 在输入到模型前对图像对进行增强处理
    
    工作流程:
    1. 光谱对比度增强 (CLAHE + 直方图均衡化)
    2. 计算多尺度差异图
    3. 边界锐化处理
    4. 自适应对比度拉伸
    """
    
    def __init__(
        self,
        use_histogram_eq=True,      # 直方图均衡化
        use_edge_enhance=True,       # 边界锐化
        use_adaptive_contrast=True,  # 自适应对比度
        clahe_clip_limit=2.0,        # CLAHE裁剪限制
        clahe_tile_size=8,           # CLAHE瓦片大小
        multiscale_levels=3,         # 多尺度层数
        edge_enhance_strength=1.5,   # 边界增强强度
        verbose=False
    ):
        self.use_histogram_eq = use_histogram_eq
        self.use_edge_enhance = use_edge_enhance
        self.use_adaptive_contrast = use_adaptive_contrast
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.multiscale_levels = multiscale_levels
        self.edge_enhance_strength = edge_enhance_strength
        self.verbose = verbose
        
        # 初始化CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )
    
    def process_pair(self, img1_pil, img2_pil):
        """
        处理图像对 - 返回增强后的PIL图像
        
        Args:
            img1_pil: PIL Image (T1时刻)
            img2_pil: PIL Image (T2时刻)
        
        Returns:
            img1_enhanced: PIL Image (增强后的T1)
            img2_enhanced: PIL Image (增强后的T2)
        """
        # 转换为numpy数组
        img1_np = np.array(img1_pil)
        img2_np = np.array(img2_pil)
        
        # 1. 光谱对比度增强
        if self.verbose:
            print("[SpectralPreprocessor] 执行光谱对比度增强...")
        img1_enhanced = self._enhance_contrast(img1_np)
        img2_enhanced = self._enhance_contrast(img2_np)
        
        # 2. 计算差异图 (用于后续处理)
        if self.verbose:
            print("[SpectralPreprocessor] 计算多尺度差异图...")
        diff_map = self._compute_difference_map(img1_enhanced, img2_enhanced)
        
        # 3. 边界锐化处理
        if self.use_edge_enhance:
            if self.verbose:
                print("[SpectralPreprocessor] 执行边界锐化...")
            img1_enhanced = self._sharpen_edges(img1_enhanced, diff_map)
            img2_enhanced = self._sharpen_edges(img2_enhanced, diff_map)
        
        # 4. 自适应对比度拉伸
        if self.use_adaptive_contrast:
            if self.verbose:
                print("[SpectralPreprocessor] 执行自适应对比度拉伸...")
            img1_enhanced = self._adaptive_contrast_stretch(img1_enhanced, diff_map)
            img2_enhanced = self._adaptive_contrast_stretch(img2_enhanced, diff_map)
        
        # 转换回PIL Image
        img1_result = Image.fromarray(np.uint8(np.clip(img1_enhanced, 0, 255)))
        img2_result = Image.fromarray(np.uint8(np.clip(img2_enhanced, 0, 255)))
        
        return img1_result, img2_result
    
    def _enhance_contrast(self, img_np):
        """
        光谱对比度增强 - 使用CLAHE和直方图均衡化
        
        Args:
            img_np: numpy array [H, W, 3]
        
        Returns:
            enhanced: numpy array [H, W, 3]
        """
        # 转换到LAB颜色空间 (更符合人眼感知)
        if img_np.dtype != np.uint8:
            img_np = np.uint8(np.clip(img_np, 0, 255))
        
        # 如果是RGB，转换到LAB
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            
            # 对L通道应用CLAHE (亮度)
            l_channel = img_lab[:, :, 0]
            l_enhanced = self.clahe.apply(l_channel)
            
            # 对比度增强
            if self.use_histogram_eq:
                l_enhanced = cv2.equalizeHist(l_enhanced)
            
            img_lab[:, :, 0] = l_enhanced
            
            # 转换回RGB
            img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        else:
            # 灰度图
            img_enhanced = self.clahe.apply(img_np)
            if self.use_histogram_eq:
                img_enhanced = cv2.equalizeHist(img_enhanced)
        
        return img_enhanced.astype(np.float32)
    
    def _compute_difference_map(self, img1, img2):
        """
        计算多尺度差异图 - 突出变化区域
        
        Args:
            img1: numpy array [H, W, 3]
            img2: numpy array [H, W, 3]
        
        Returns:
            diff_map: numpy array [H, W] (0-1范围)
        """
        # 计算基础差异
        diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        
        # 转换为灰度差异
        if len(diff.shape) == 3:
            diff_gray = np.mean(diff, axis=2)
        else:
            diff_gray = diff
        
        # 多尺度融合
        diff_multiscale = diff_gray.copy()
        
        for level in range(1, self.multiscale_levels):
            # 下采样
            scale = 2 ** level
            h, w = diff_gray.shape
            h_scaled, w_scaled = h // scale, w // scale
            
            if h_scaled > 0 and w_scaled > 0:
                diff_scaled = cv2.resize(diff_gray, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
                # 上采样回原大小
                diff_scaled = cv2.resize(diff_scaled, (w, h), interpolation=cv2.INTER_LINEAR)
                # 融合 (加权平均)
                weight = 1.0 / (2 ** level)
                diff_multiscale = diff_multiscale * (1 - weight) + diff_scaled * weight
        
        # 归一化到0-1
        diff_map = (diff_multiscale - diff_multiscale.min()) / (diff_multiscale.max() - diff_multiscale.min() + 1e-8)
        
        return diff_map
    
    def _sharpen_edges(self, img, diff_map):
        """
        边界锐化处理 - 在变化区域增强边界
        
        Args:
            img: numpy array [H, W, 3]
            diff_map: numpy array [H, W] (差异图)
        
        Returns:
            sharpened: numpy array [H, W, 3]
        """
        # 计算Laplacian边界
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(np.uint8(np.clip(img, 0, 255)), cv2.COLOR_RGB2GRAY)
        else:
            img_gray = np.uint8(np.clip(img, 0, 255))
        
        # 使用Laplacian算子检测边界
        laplacian = cv2.Laplacian(img_gray, cv2.CV_32F)
        laplacian = np.abs(laplacian)
        
        # 归一化Laplacian
        laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min() + 1e-8)
        
        # 根据差异图调整锐化强度
        # 在变化区域增加锐化，在稳定区域减少锐化
        adaptive_strength = 0.5 + diff_map * self.edge_enhance_strength
        
        # 应用锐化
        if len(img.shape) == 3:
            sharpened = img.copy()
            for c in range(3):
                # 对每个通道应用自适应锐化
                channel = img[:, :, c]
                sharpened_channel = channel + laplacian * adaptive_strength * 10
                sharpened[:, :, c] = sharpened_channel
        else:
            sharpened = img + laplacian * adaptive_strength * 10
        
        return sharpened
    
    def _adaptive_contrast_stretch(self, img, diff_map):
        """
        自适应对比度拉伸 - 根据变化区域自动调整
        
        Args:
            img: numpy array [H, W, 3]
            diff_map: numpy array [H, W] (差异图)
        
        Returns:
            stretched: numpy array [H, W, 3]
        """
        # 计算全局对比度
        img_min = img.min()
        img_max = img.max()
        
        if img_max - img_min < 1e-6:
            return img
        
        # 基础对比度拉伸
        stretched = (img - img_min) / (img_max - img_min) * 255
        
        # 根据差异图调整
        # 在变化区域增加对比度
        contrast_factor = 1.0 + diff_map * 0.5  # 最多增加50%对比度
        
        if len(stretched.shape) == 3:
            for c in range(3):
                center = stretched[:, :, c].mean()
                stretched[:, :, c] = center + (stretched[:, :, c] - center) * contrast_factor
        else:
            center = stretched.mean()
            stretched = center + (stretched - center) * contrast_factor
        
        return stretched
    
    def process_batch(self, img1_list, img2_list):
        """
        批量处理图像对
        
        Args:
            img1_list: List of PIL Images
            img2_list: List of PIL Images
        
        Returns:
            img1_enhanced_list: List of PIL Images
            img2_enhanced_list: List of PIL Images
        """
        img1_enhanced_list = []
        img2_enhanced_list = []
        
        for img1, img2 in zip(img1_list, img2_list):
            img1_enh, img2_enh = self.process_pair(img1, img2)
            img1_enhanced_list.append(img1_enh)
            img2_enhanced_list.append(img2_enh)
        
        return img1_enhanced_list, img2_enhanced_list


class SpectralAugmentationTransform:
    """
    集成到数据加载流程的光谱增强变换
    
    使用方法:
        from data.spectral_preprocessing import SpectralAugmentationTransform
        
        在 cd_dataset.py 中:
        self.spectral_transform = SpectralAugmentationTransform(
            enable_spectral=True,
            enable_multiscale=True,
            enable_edge_enhance=True
        )
        
        在 __getitem__ 中:
        if self.opt.phase == 'train':
            img1, img2 = self.spectral_transform(img1, img2)
    """
    
    def __init__(
        self,
        enable_spectral=True,
        enable_multiscale=True,
        enable_edge_enhance=True,
        enable_adaptive_contrast=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        edge_enhance_strength=1.5,
        apply_probability=0.8  # 80%概率应用
    ):
        self.enable_spectral = enable_spectral
        self.enable_multiscale = enable_multiscale
        self.enable_edge_enhance = enable_edge_enhance
        self.enable_adaptive_contrast = enable_adaptive_contrast
        self.apply_probability = apply_probability
        
        self.preprocessor = SpectralPreprocessor(
            use_histogram_eq=enable_spectral,
            use_edge_enhance=enable_edge_enhance,
            use_adaptive_contrast=enable_adaptive_contrast,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_size=clahe_tile_size,
            multiscale_levels=3 if enable_multiscale else 1,
            edge_enhance_strength=edge_enhance_strength,
            verbose=False
        )
    
    def __call__(self, img1, img2):
        """
        Args:
            img1: PIL Image
            img2: PIL Image
        
        Returns:
            img1_enhanced: PIL Image
            img2_enhanced: PIL Image
        """
        import random
        
        # 根据概率决定是否应用
        if random.random() > self.apply_probability:
            return img1, img2
        
        return self.preprocessor.process_pair(img1, img2)


# ==================== 使用示例 ====================
if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    
    # 创建测试图像
    print("创建测试图像...")
    img1 = Image.fromarray(np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(60, 160, (256, 256, 3), dtype=np.uint8))
    
    # 初始化预处理器
    print("初始化光谱预处理器...")
    preprocessor = SpectralPreprocessor(
        use_histogram_eq=True,
        use_edge_enhance=True,
        use_adaptive_contrast=True,
        verbose=True
    )
    
    # 处理图像对
    print("\n处理图像对...")
    img1_enh, img2_enh = preprocessor.process_pair(img1, img2)
    
    print(f"原始图像1: {img1.size}, {img1.mode}")
    print(f"增强后图像1: {img1_enh.size}, {img1_enh.mode}")
    print(f"原始图像2: {img2.size}, {img2.mode}")
    print(f"增强后图像2: {img2_enh.size}, {img2_enh.mode}")
    
    # 测试Transform
    print("\n测试SpectralAugmentationTransform...")
    transform = SpectralAugmentationTransform(
        enable_spectral=True,
        enable_multiscale=True,
        enable_edge_enhance=True,
        apply_probability=1.0
    )
    
    img1_t, img2_t = transform(img1, img2)
    print(f"Transform后图像1: {img1_t.size}, {img1_t.mode}")
    print(f"Transform后图像2: {img2_t.size}, {img2_t.mode}")
    
    print("\n✓ 所有测试通过!")
