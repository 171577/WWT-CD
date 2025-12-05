import random
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import ImageFilter, Image
from data.spectral_preprocessing import SpectralAugmentationTransform


class TransformsEnhanced(object):
    """
    增强版数据增强 - 针对变化检测任务优化
    
    包含:
    1. 基础几何变换 (翻转、旋转、缩放)
    2. 颜色增强 (亮度、对比度、饱和度)
    3. 噪声注入 (高斯噪声、椒盐噪声)
    4. 模糊增强 (高斯模糊)
    5. 时序增强 (T1/T2交换)
    
    使用方法:
        在cd_dataset.py中:
        from data.transform_enhanced import TransformsEnhanced
        self.transform = transforms.Compose([TransformsEnhanced()])
    """
    
    def __init__(
        self, 
        enable_colorjitter=True,    # 是否启用颜色增强
        enable_noise=False,          # 是否启用噪声（谨慎使用）
        enable_blur=False,           # 是否启用模糊（谨慎使用）
        crop_scale_range=(0.5, 1.0), # 裁剪缩放范围
        colorjitter_strength=0.3,    # 颜色增强强度 (0.1-0.5)
        enable_spectral=True,        # 是否启用光谱增强 (新增)
        enable_multiscale=True,      # 是否启用多尺度处理 (新增)
        enable_edge_enhance=True,    # 是否启用边界锐化 (新增)
        spectral_probability=0.8     # 光谱增强应用概率 (新增)
    ):
        self.enable_colorjitter = enable_colorjitter
        self.enable_noise = enable_noise
        self.enable_blur = enable_blur
        self.crop_scale_range = crop_scale_range
        self.colorjitter_strength = colorjitter_strength
        
        # 光谱增强配置 (新增)
        self.enable_spectral = enable_spectral
        self.enable_multiscale = enable_multiscale
        self.enable_edge_enhance = enable_edge_enhance
        self.spectral_probability = spectral_probability
        
        if enable_spectral:
            self.spectral_transform = SpectralAugmentationTransform(
                enable_spectral=True,
                enable_multiscale=enable_multiscale,
                enable_edge_enhance=enable_edge_enhance,
                enable_adaptive_contrast=True,
                apply_probability=spectral_probability
            )
    
    def __call__(self, _data):
        img1, img2, label, label_weak = _data['img1'], _data['img2'], _data['label'], _data['label_weak']
        
        # ============ 0. 光谱增强 (新增 - 优先级1) ============
        # 在其他增强前应用，以保留原始的变化信息
        if self.enable_spectral:
            img1, img2 = self.spectral_transform(img1, img2)
        
        # ============ 1. 时序交换 (50%) ============
        if random.random() < 0.5:
            img1, img2 = img2, img1
        
        # ============ 2. 水平翻转 (50%) ============
        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            label = TF.hflip(label)
            label_weak = TF.hflip(label_weak)
        
        # ============ 3. 垂直翻转 (50%) ============
        if random.random() < 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            label = TF.vflip(label)
            label_weak = TF.vflip(label_weak)
        
        # ============ 4. 随机旋转 (50%) ============
        if random.random() < 0.5:
            angles = [90, 180, 270]
            angle = random.choice(angles)
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
            label = TF.rotate(label, angle)
            label_weak = TF.rotate(label_weak, angle)
        
        # ============ 5. 随机缩放裁剪 (50%) ============
        if random.random() < 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=(256, 256)).get_params(
                img=img1, 
                scale=self.crop_scale_range,  # 使用可配置的缩放范围
                ratio=[0.75, 1.333]
            )
            img1 = TF.resized_crop(img1, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.BILINEAR)
            img2 = TF.resized_crop(img2, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.BILINEAR)
            label = TF.resized_crop(label, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.NEAREST)
            label_weak = TF.resized_crop(label_weak, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.NEAREST)
        
        # ============ 6. 颜色增强 (40%, 可选) ============
        if self.enable_colorjitter and random.random() < 0.4:
            # 亮度调整
            if random.random() < 0.5:
                brightness_factor = random.uniform(
                    1 - self.colorjitter_strength, 
                    1 + self.colorjitter_strength
                )
                img1 = TF.adjust_brightness(img1, brightness_factor)
                img2 = TF.adjust_brightness(img2, brightness_factor)
            
            # 对比度调整
            if random.random() < 0.5:
                contrast_factor = random.uniform(
                    1 - self.colorjitter_strength, 
                    1 + self.colorjitter_strength
                )
                img1 = TF.adjust_contrast(img1, contrast_factor)
                img2 = TF.adjust_contrast(img2, contrast_factor)
            
            # 饱和度调整
            if random.random() < 0.5:
                saturation_factor = random.uniform(
                    1 - self.colorjitter_strength, 
                    1 + self.colorjitter_strength
                )
                img1 = TF.adjust_saturation(img1, saturation_factor)
                img2 = TF.adjust_saturation(img2, saturation_factor)
        
        # ============ 7. 高斯模糊 (20%, 可选) ============
        if self.enable_blur and random.random() < 0.2:
            radius = random.uniform(0.5, 1.5)
            img1 = img1.filter(ImageFilter.GaussianBlur(radius))
            img2 = img2.filter(ImageFilter.GaussianBlur(radius))
        
        # ============ 8. 噪声注入 (15%, 可选) ============
        if self.enable_noise and random.random() < 0.15:
            # 高斯噪声
            img1 = self.add_gaussian_noise(img1, std=5)
            img2 = self.add_gaussian_noise(img2, std=5)
        
        return {'img1': img1, 'img2': img2, 'label': label, 'label_weak': label_weak}
    
    def add_gaussian_noise(self, img, std=5):
        """添加高斯噪声"""
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, std, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)


class TransformsStandard(object):
    """
    标准数据增强 - 原始版本 + 颜色增强 + 光谱增强
    
    在原始Transforms基础上启用了颜色增强和光谱增强
    推荐用于生产环境
    """
    def __init__(
        self,
        enable_spectral=True,        # 是否启用光谱增强
        enable_multiscale=True,      # 是否启用多尺度处理
        enable_edge_enhance=True,    # 是否启用边界锐化
        spectral_probability=0.8     # 光谱增强应用概率
    ):
        self.enable_spectral = enable_spectral
        self.enable_multiscale = enable_multiscale
        self.enable_edge_enhance = enable_edge_enhance
        self.spectral_probability = spectral_probability
        
        if enable_spectral:
            self.spectral_transform = SpectralAugmentationTransform(
                enable_spectral=True,
                enable_multiscale=enable_multiscale,
                enable_edge_enhance=enable_edge_enhance,
                enable_adaptive_contrast=True,
                apply_probability=spectral_probability
            )
    
    def __call__(self, _data):
        img1, img2, label, label_weak = _data['img1'], _data['img2'], _data['label'], _data['label_weak']
        
        # ============ 0. 光谱增强 (新增 - 优先级1) ============
        if self.enable_spectral:
            img1, img2 = self.spectral_transform(img1, img2)
        
        # 时序交换
        if random.random() < 0.5:
            img1, img2 = img2, img1
        
        # 水平翻转
        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            label = TF.hflip(label)
            label_weak = TF.hflip(label_weak)
        
        # 垂直翻转
        if random.random() < 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            label = TF.vflip(label)
            label_weak = TF.vflip(label_weak)
        
        # 随机旋转
        if random.random() < 0.5:
            angles = [90, 180, 270]
            angle = random.choice(angles)
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
            label = TF.rotate(label, angle)
            label_weak = TF.rotate(label_weak, angle)
        
        # 颜色增强 (新增！)
        if random.random() < 0.4:
            brightness_factor = random.uniform(0.7, 1.3)
            img1 = TF.adjust_brightness(img1, brightness_factor)
            img2 = TF.adjust_brightness(img2, brightness_factor)
            
            contrast_factor = random.uniform(0.7, 1.3)
            img1 = TF.adjust_contrast(img1, contrast_factor)
            img2 = TF.adjust_contrast(img2, contrast_factor)
            
            saturation_factor = random.uniform(0.7, 1.3)
            img1 = TF.adjust_saturation(img1, saturation_factor)
            img2 = TF.adjust_saturation(img2, saturation_factor)
        
        # 随机缩放裁剪
        if random.random() < 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=(256, 256)).get_params(
                img=img1, 
                scale=[0.5, 1.0],  # 增加缩放范围
                ratio=[0.75, 1.333]
            )
            img1 = TF.resized_crop(img1, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.BILINEAR)
            img2 = TF.resized_crop(img2, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.BILINEAR)
            label = TF.resized_crop(label, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.NEAREST)
            label_weak = TF.resized_crop(label_weak, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.NEAREST)
        
        return {'img1': img1, 'img2': img2, 'label': label, 'label_weak': label_weak}


class TransformsMinimal(object):
    """
    最小数据增强 - 仅几何变换
    
    适用于:
    - 快速实验
    - 数据集已经很大
    - 避免过度增强
    """
    def __call__(self, _data):
        img1, img2, label, label_weak = _data['img1'], _data['img2'], _data['label'], _data['label_weak']
        
        # 时序交换
        if random.random() < 0.5:
            img1, img2 = img2, img1
        
        # 水平翻转
        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            label = TF.hflip(label)
            label_weak = TF.hflip(label_weak)
        
        # 垂直翻转
        if random.random() < 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            label = TF.vflip(label)
            label_weak = TF.vflip(label_weak)
        
        return {'img1': img1, 'img2': img2, 'label': label, 'label_weak': label_weak}


# ==================== 使用示例 ====================
if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    
    # 创建测试数据
    img1 = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    label = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8))
    label_weak = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8))
    
    data = {'img1': img1, 'img2': img2, 'label': label, 'label_weak': label_weak}
    
    # 测试不同增强策略
    print("Testing TransformsMinimal...")
    transform_minimal = TransformsMinimal()
    result = transform_minimal(data)
    print(f"  img1 shape: {np.array(result['img1']).shape}")
    
    print("\nTesting TransformsStandard...")
    transform_standard = TransformsStandard()
    result = transform_standard(data)
    print(f"  img1 shape: {np.array(result['img1']).shape}")
    
    print("\nTesting TransformsEnhanced...")
    transform_enhanced = TransformsEnhanced(
        enable_colorjitter=True,
        enable_noise=False,
        enable_blur=False,
        colorjitter_strength=0.3
    )
    result = transform_enhanced(data)
    print(f"  img1 shape: {np.array(result['img1']).shape}")
    
    print("\nAll tests passed!")
