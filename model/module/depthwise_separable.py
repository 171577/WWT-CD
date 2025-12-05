"""
深度可分离卷积模块
用于模型轻量化，减少参数量和计算量
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积 = 深度卷积 + 逐点卷积
    
    参数减少比例:
    - 3×3卷积: 88.9%
    - 5×5卷积: 96.0%
    - 7×7卷积: 98.0%
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            bias: 是否使用偏置
        """
        super(DepthwiseSeparableConv, self).__init__()
        
        # 深度卷积: 每个通道单独处理
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # 逐点卷积: 1×1卷积用于通道混合
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        # 深度卷积
        x = self.depthwise(x)
        x = self.bn1(x)
        
        # 逐点卷积
        x = self.pointwise(x)
        x = self.bn2(x)
        
        return x


class DepthwiseSeparableConvWithReLU(nn.Module):
    """
    深度可分离卷积 + ReLU激活
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConvWithReLU, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class DepthwiseSeparableConvSequential(nn.Module):
    """
    深度可分离卷积序列，用于替换标准卷积序列
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_relu=True, bias=False):
        super(DepthwiseSeparableConvSequential, self).__init__()
        
        if use_relu:
            self.conv = DepthwiseSeparableConvWithReLU(
                in_channels, out_channels, kernel_size, stride, padding, bias
            )
        else:
            self.conv = DepthwiseSeparableConv(
                in_channels, out_channels, kernel_size, stride, padding, bias
            )
    
    def forward(self, x):
        return self.conv(x)


def replace_conv_with_depthwise(module, in_channels, out_channels, kernel_size=3, 
                                 stride=1, padding=1, use_relu=False):
    """
    将标准卷积替换为深度可分离卷积
    
    Args:
        module: 要替换的模块
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        use_relu: 是否使用ReLU激活
    
    Returns:
        替换后的深度可分离卷积模块
    """
    if use_relu:
        return DepthwiseSeparableConvWithReLU(
            in_channels, out_channels, kernel_size, stride, padding
        )
    else:
        return DepthwiseSeparableConv(
            in_channels, out_channels, kernel_size, stride, padding
        )


# 参数量对比工具
def calculate_conv_params(in_channels, out_channels, kernel_size):
    """计算标准卷积的参数量"""
    return in_channels * out_channels * kernel_size * kernel_size


def calculate_depthwise_params(in_channels, out_channels, kernel_size):
    """计算深度可分离卷积的参数量"""
    return in_channels * kernel_size * kernel_size + in_channels * out_channels


def calculate_reduction_ratio(in_channels, out_channels, kernel_size):
    """计算参数减少比例"""
    standard = calculate_conv_params(in_channels, out_channels, kernel_size)
    depthwise = calculate_depthwise_params(in_channels, out_channels, kernel_size)
    return (standard - depthwise) / standard * 100
