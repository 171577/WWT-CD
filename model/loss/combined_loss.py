"""
组合损失函数 - 焦点损失 + 边界损失 + Dice损失

用于Enhanced v3模型的训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import dice_loss


class FocalLoss(nn.Module):
    """
    焦点损失 - 处理类不平衡和难样本
    
    论文: Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: 类权重 (默认0.25)
        gamma: 焦点参数 (默认2.0)
        weight: 类权重 (默认None)
    """
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] labels (int64)
        
        Returns:
            loss: 标量损失值
        """
        # 计算交叉熵损失 (不求和)
        ce_loss = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        
        # 计算pt (预测正确的概率)
        p = torch.exp(-ce_loss)
        
        # 焦点损失: alpha * (1 - pt)^gamma * ce_loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """
    边界损失 - 增强边界精度
    
    使用Laplacian算子检测边界，对边界像素施加更高的权重
    
    Args:
        weight: 边界权重倍数 (默认2.0)
    """
    def __init__(self, weight=2.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] labels (int64)
        
        Returns:
            loss: 标量损失值
        """
        # 计算边界
        boundary = self._compute_boundary(target)
        
        # 计算加权损失
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = ce_loss * (1 + self.weight * boundary)
        
        return weighted_loss.mean()
    
    def _compute_boundary(self, target):
        """
        使用Laplacian算子检测边界
        
        Args:
            target: [B, H, W] labels
        
        Returns:
            boundary: [B, H, W] 边界掩码 (0或1)
        """
        # Laplacian核
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32, device=target.device)
        
        # 扩展维度用于卷积
        target_float = target.float().unsqueeze(1)  # [B, 1, H, W]
        
        # 应用Laplacian卷积
        boundary = F.conv2d(target_float, kernel.view(1, 1, 3, 3), padding=1)
        
        # 二值化: 梯度大于0表示边界
        boundary = (boundary > 0).float().squeeze(1)  # [B, H, W]
        
        return boundary


class DiceLoss(nn.Module):
    """Dice损失 - 处理类不平衡"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] labels (int64)
        
        Returns:
            loss: 标量损失值
        """
        # 转换为概率
        pred_probs = F.softmax(pred, dim=1)
        
        # 转换target为one-hot
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # 计算Dice系数
        inter = (pred_probs * target_one_hot).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2 * inter + 1e-5) / (union + 1e-5)
        
        return (1 - dice).mean()


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    total_loss = lambda_focal * focal_loss + lambda_boundary * boundary_loss + lambda_dice * dice_loss
    
    Args:
        lambda_focal: 焦点损失权重 (默认1.0)
        lambda_boundary: 边界损失权重 (默认0.5)
        lambda_dice: Dice损失权重 (默认0.5)
        focal_alpha: 焦点损失alpha参数 (默认0.25)
        focal_gamma: 焦点损失gamma参数 (默认2.0)
        boundary_weight: 边界权重倍数 (默认2.0)
        ce_weight: 交叉熵类权重 (默认None)
    """
    def __init__(
        self,
        lambda_focal=1.0,
        lambda_boundary=0.5,
        lambda_dice=0.5,
        focal_alpha=0.25,
        focal_gamma=2.0,
        boundary_weight=2.0,
        ce_weight=None
    ):
        super().__init__()
        self.lambda_focal = lambda_focal
        self.lambda_boundary = lambda_boundary
        self.lambda_dice = lambda_dice
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, weight=ce_weight)
        self.boundary_loss = BoundaryLoss(weight=boundary_weight)
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] labels (int64)
        
        Returns:
            loss_dict: 包含各项损失的字典
            total_loss: 总损失值
        """
        # 计算各项损失
        loss_focal = self.focal_loss(pred, target)
        loss_boundary = self.boundary_loss(pred, target)
        loss_dice = self.dice_loss(pred, target)
        
        # 组合损失
        total_loss = (
            self.lambda_focal * loss_focal +
            self.lambda_boundary * loss_boundary +
            self.lambda_dice * loss_dice
        )
        
        # 返回损失字典和总损失
        loss_dict = {
            'focal': loss_focal.item(),
            'boundary': loss_boundary.item(),
            'dice': loss_dice.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


# 便利函数
def create_combined_loss(device, num_classes=2, class_weights=None):
    """
    创建组合损失函数
    
    Args:
        device: 设备 (cuda或cpu)
        num_classes: 类别数 (默认2)
        class_weights: 类权重 (默认None)
    
    Returns:
        loss_fn: 组合损失函数
    """
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    loss_fn = CombinedLoss(
        lambda_focal=1.0,
        lambda_boundary=0.5,
        lambda_dice=0.5,
        focal_alpha=0.25,
        focal_gamma=2.0,
        boundary_weight=2.0,
        ce_weight=class_weights
    )
    
    return loss_fn
