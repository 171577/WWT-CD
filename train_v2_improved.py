import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from option import Options
from data.cd_dataset import DataLoader
from model.module.swcd_dgma2_multiscale import SWCD_DGMA2_MultiScale, SWCD_DGMA2_MultiScale_Attention
from model.loss.dice import dice_loss_v1
from tqdm import tqdm
from util.metric_tool import ConfuseMatrixMeter
import os
import numpy as np
import random
import logging
import datetime
import json
logging.getLogger('PIL').setLevel(logging.WARNING)


def init_logging(filedir: str):
    """Initialize logging to file and console"""
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    logger = logging.getLogger()
    fh = logging.FileHandler(filename=filedir + '/log_' + get_date_str() + '.txt')
    sh = logging.StreamHandler()
    formatter_fh = logging.Formatter('%(asctime)s %(message)s')
    formatter_sh = logging.Formatter('%(message)s')
    fh.setFormatter(formatter_fh)
    sh.setFormatter(formatter_sh)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(10)
    fh.setLevel(10)
    sh.setLevel(10)
    return logging


def setup_seed(seed):
    """Setup random seed for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


class GradientMonitor:
    def __init__(self, model):
        self.model = model
        self.grad_norms = []
    
    def monitor(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
        
        if total_norm > 10.0:
            # 只是警告，后续会有clip_grad_norm处理
            logging.warning(f"⚠️ 梯度较大: {total_norm:.4f} (将被裁剪到1.0)")
        elif total_norm < 1e-6:
            logging.warning(f"⚠️ 梯度消失: {total_norm:.4e}")
        
        return total_norm


class V2ImprovedTrainer(object):
    def __init__(self, opt):
        self.opt = opt
        
        # 数据加载
        train_loader = DataLoader(opt)
        self.train_data = train_loader.load_data()
        train_size = len(train_loader)
        logging.info(f"#training images = {train_size}")
        
        opt.phase = 'val'
        opt.batch_size = 64
        val_loader = DataLoader(opt)
        self.val_data = val_loader.load_data()
        val_size = len(val_loader)
        logging.info(f"#validation images = {val_size}")
        opt.phase = 'train'
        opt.batch_size = getattr(opt, 'batch_size', 64)
        
        # 创建模型
        model_variant = getattr(opt, 'model_variant', 'multiscale_attention')
        
        if model_variant == 'multiscale_attention':
            self.model = SWCD_DGMA2_MultiScale_Attention(
                input_nc=3,
                output_nc=2,
                in_channels_list=[64, 128, 256, 512],
                out_channels_list=[64, 128, 256, 256],
                dam_per_scale=getattr(opt, 'dam_per_scale', 2),
                beta_init=getattr(opt, 'beta_init', 0.3),
                dropout_rate=0.3  # 新增参数: Dropout
            ).cuda()
        else:
            self.model = SWCD_DGMA2_MultiScale(
                input_nc=3,
                output_nc=2,
                in_channels_list=[64, 128, 256, 512],
                out_channels_list=[64, 128, 256, 256],
                dam_per_scale=getattr(opt, 'dam_per_scale', 2),
                beta_init=getattr(opt, 'beta_init', 0.3),
                dropout_rate=0.3  # 新增参数: Dropout
            ).cuda()
        
        # 损失函数
        class_weights = torch.tensor([1.0, 2.0]).cuda()
        # 改进: 使用Label Smoothing
        self.criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.criterion_dice = dice_loss_v1()
        
        # 弱监督损失权重
        self.lambda_sim = getattr(opt, 'lambda_sim', 0.3)  # 弱监督损失权重
        
        # 训练状态
        self.current_stage = 1
        self.stage1_epochs = getattr(opt, 'stage1_epochs', 50)
        self.stage2_epochs = getattr(opt, 'stage2_epochs', 150)  # 减少总轮数
        self.total_epochs = self.stage1_epochs + self.stage2_epochs
        
        self.best_epoch = 0
        self.previous_best = 0.0
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        
        # 早停 - 改进：更严格
        self.early_stopping_patience_stage1 = 30  # 阶段1：
        self.early_stopping_patience_stage2 = 50  # 阶段2：从80改为50，减少过度训练
        self.patience_counter = 0
        
        # 优化器
        self.optimizer = None
        self.scheduler = None
        
        # 梯度监控
        self.grad_monitor = GradientMonitor(self.model)
        
        # 统计信息
        total_params = self.count_parameters()
        logging.info(f"Model parameters: {total_params:.2f}M (with built-in decoder)")
        logging.info(f"Training plan: Stage1={self.stage1_epochs} epochs, Stage2={self.stage2_epochs} epochs")
        logging.info(f"Early stopping patience: Stage1={self.early_stopping_patience_stage1}, Stage2={self.early_stopping_patience_stage2}")
    
    def count_parameters(self):
        """计算总参数量"""
        model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        return model_params
    
    def setup_stage1(self):
        logging.info("\n" + "="*80)
        logging.info("STAGE 1: Training with selective backbone unfreezing (Improved)")
        logging.info("="*80)

        # 改进: 渐进式解冻策略而非完全冻结
        # 冻结早期层（低级特征）
        for param in self.model.res.layer1.parameters():
            param.requires_grad = False
        
        # 低学习率微调中期层
        for param in self.model.res.layer2.parameters():
            param.requires_grad = True
        
        # 正常学习率微调后期层
        for param in self.model.res.layer3.parameters():
            param.requires_grad = True
        
        for param in self.model.res.layer4.parameters():
            param.requires_grad = True
        
        # 解冻新模块
        for param in self.model.dgma2_branch.parameters():
            param.requires_grad = True
        for param in self.model.decoder_sim.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        
        # 改进: 分层学习率策略
        self.optimizer = AdamW([
            {'params': self.model.res.layer2.parameters(), 'lr': 5e-5},   # 低学习率
            {'params': self.model.res.layer3.parameters(), 'lr': 1e-4},   # 中等学习率
            {'params': self.model.res.layer4.parameters(), 'lr': 1e-4},   # 中等学习率
            {'params': self.model.dgma2_branch.parameters(), 'lr': 2e-4},  # 高学习率
            {'params': self.model.decoder_sim.parameters(), 'lr': 2e-4},   # 高学习率
            {'params': self.model.decoder.parameters(), 'lr': 2e-4}        # 高学习率
        ], weight_decay=5e-5)
        
        # 改进2：使用CosineAnnealingWarmRestarts（重启余弦）
        # 每20个epoch重启一次，保持学习率在较高水平
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,      # 初始周期：增加到20
            T_mult=2,    # 周期逐步增长 (必须是整数)
            eta_min=5e-7 # 降低最小学习率
        )
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        logging.info(f"Trainable parameters: {trainable:.2f}M")
        logging.info(f"Learning rate: 2e-4 (new modules), weight_decay: 5e-5")
        logging.info(f"Scheduler: CosineAnnealingWarmRestarts (T_0=20, T_mult=2, eta_min=5e-7)")
        logging.info(f"Early stopping patience: {self.early_stopping_patience_stage1}")
        
        self.current_stage = 1
    
    def setup_stage2(self):
        """
        阶段2: 完全解冻，使用差异化学习率
        改进：分层学习率策略，更精细的微调，避免过度学习
        """
        logging.info("\n" + "="*80)
        logging.info("STAGE 2: Joint fine-tuning with differential learning rates (Improved)")
        logging.info("="*80)
        
        # 完全解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 修复: 使用更小的初始学习率，避免Stage 2过度学习导致陷入局部最优
        # 改进: 分层学习率策略，早期层学习率最低
        self.optimizer = AdamW([
            {'params': self.model.res.layer1.parameters(), 'lr': 1e-6},   # 最低
            {'params': self.model.res.layer2.parameters(), 'lr': 5e-6},   # 很低
            {'params': self.model.res.layer3.parameters(), 'lr': 1e-5},   # 低
            {'params': self.model.res.layer4.parameters(), 'lr': 5e-5},   # 中等
            {'params': self.model.dgma2_branch.parameters(), 'lr': 1e-4},  # 改为1e-4（提升2倍）
            {'params': self.model.decoder_sim.parameters(), 'lr': 1e-4},   # 改为1e-4（提升2倍）
            {'params': self.model.decoder.parameters(), 'lr': 1e-4}        # 改为1e-4（提升2倍）
        ], weight_decay=1e-4)  # 增加权重衰减到1e-4
        
        # 修复5: 优化学习率调度策略
        # T_0=20使学习率衰减更平缓
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,      # 增加到20
            T_mult=2,    # 周期逐步增长 (必须是整数)
            eta_min=5e-7 # 降低最小学习率
        )
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        logging.info(f"Trainable parameters: {trainable:.2f}M (all)")
        logging.info(f"Learning rate: 1e-4 (new modules), 5e-5 (backbone), weight_decay: 1e-4")
        logging.info(f"Scheduler: CosineAnnealingWarmRestarts (T_0=20, T_mult=2, eta_min=5e-7)")
        logging.info(f"Early stopping patience: {self.early_stopping_patience_stage2}")
        
        self.current_stage = 2
    
    def get_dynamic_lambda_dice(self, epoch):
        """
        改进3：动态调整Dice Loss权重
        前期：0.3（更关注CE Loss快速收敛）
        后期：0.7（更关注Dice Loss精细优化）
        使用余弦衰减策略（改进）
        """
        if self.current_stage == 1:
            progress = epoch / self.stage1_epochs
        else:
            progress = (epoch - self.stage1_epochs) / self.stage2_epochs
        
        # 改进: 使用余弦衰减而非线性变化，更符合训练动态
        import math
        lambda_dice = 0.3 + 0.4 * (1 - math.cos(progress * math.pi)) / 2
        return lambda_dice
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0
        total_sim_loss = 0.0
        
        # 获取动态Dice Loss权重
        lambda_dice = self.get_dynamic_lambda_dice(epoch)
        
        pbar = tqdm(self.train_data, desc=f"Epoch {epoch+1}/{self.total_epochs} [TRAIN]")
        
        for batch_idx, batch in enumerate(pbar):
            img1 = batch['img1'].cuda()
            img2 = batch['img2'].cuda()
            label = batch['label'].cuda().long()
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 模型输出
            outputs, sim_mask = self.model(img1, img2)
            
            # 处理多尺度输出 (Deep Supervision)
            if isinstance(outputs, tuple):
                pred, aux_d3, aux_d4 = outputs
            else:
                pred = outputs
                aux_d3 = None
                aux_d4 = None
            
            # 损失计算 - 使用动态权重
            loss_ce = self.criterion_ce(pred, label)
            loss_dice = self.criterion_dice(pred, label.unsqueeze(1).float())
            loss_main = loss_ce + lambda_dice * loss_dice
            
            # 辅助损失 (Deep Supervision)
            loss_aux = 0.0
            if aux_d3 is not None:
                loss_ce_d3 = self.criterion_ce(aux_d3, label)
                loss_dice_d3 = self.criterion_dice(aux_d3, label.unsqueeze(1).float())
                loss_aux += 0.4 * (loss_ce_d3 + lambda_dice * loss_dice_d3)
                
            if aux_d4 is not None:
                loss_ce_d4 = self.criterion_ce(aux_d4, label)
                loss_dice_d4 = self.criterion_dice(aux_d4, label.unsqueeze(1).float())
                loss_aux += 0.3 * (loss_ce_d4 + lambda_dice * loss_dice_d4)
            
            # 修复1: 弱监督损失 - 使用更宽松的软标签避免过度约束
            # 相似度分支应该学习真实的特征相似性，而不是强制匹配二值变化标签
            sim_target = label.float().unsqueeze(1)  # [B, 1, H, W]
            # 改进：变化区域0.65，不变区域0.35（更宽松，允许相似度学习真实特征相似性）
            # 相比原来的[0.3, 0.7]，现在是[0.35, 0.65]，范围更宽
            sim_target_soft = sim_target * 0.3 + 0.35  # 宽松软标签：变化0.65，不变0.35
            loss_sim = F.binary_cross_entropy_with_logits(sim_mask, sim_target_soft)
            
            # 组合损失
            loss = loss_main + loss_aux + self.lambda_sim * loss_sim
            
            # 反向传播
            loss.backward()
            
            # 梯度监控
            grad_norm = self.grad_monitor.monitor()
            
            # 修复6: 调整梯度裁剪阈值为1.0，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # self.scheduler.step() # 移到epoch末尾，避免每个batch都更新导致周期过短
            
            total_loss += loss.item()
            total_ce_loss += loss_ce.item()
            total_dice_loss += loss_dice.item()
            total_sim_loss += loss_sim.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{loss_ce.item():.4f}',
                'dice': f'{loss_dice.item():.4f}',
                'sim': f'{loss_sim.item():.4f}',
                'lambda_dice': f'{lambda_dice:.3f}'
            })
        
        avg_loss = total_loss / len(self.train_data)
        avg_ce_loss = total_ce_loss / len(self.train_data)
        avg_dice_loss = total_dice_loss / len(self.train_data)
        avg_sim_loss = total_sim_loss / len(self.train_data)
        
        # 学习率调度器更新 (每个epoch更新一次)
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        
        logging.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, CE: {avg_ce_loss:.4f}, Dice: {avg_dice_loss:.4f}, Sim: {avg_sim_loss:.4f}, lambda_dice: {lambda_dice:.3f}, LR: {current_lr:.2e}")
        
        return avg_loss
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        self.running_metric.clear()
        
        pbar = tqdm(self.val_data, desc=f"Epoch {epoch+1}/{self.total_epochs} [VAL]")
        
        with torch.no_grad():
            for batch in pbar:
                img1 = batch['img1'].cuda()
                img2 = batch['img2'].cuda()
                label = batch['label']
                
                # 前向传播
                pred, sim_mask = self.model(img1, img2)
                
                # 预测
                pred = torch.argmax(pred, dim=1)
                
                # 更新混淆矩阵
                self.running_metric.update_cm(
                    pr=pred.cpu().numpy(),
                    gt=label.cpu().numpy()
                )
        
        # 获取详细指标
        scores_dict = self.running_metric.get_scores()
        
        logging.info(f"Epoch {epoch+1} Validation Results:")
        for key, val in scores_dict.items():
            logging.info(f"  {key}: {val:.4f}")
        
        # 计算Macro F1
        macro_f1 = (scores_dict.get('F1_0', 0) + scores_dict.get('F1_1', 0)) / 2
        
        # 保存最佳模型
        if macro_f1 > self.previous_best:
            self.previous_best = macro_f1
            self.best_epoch = epoch + 1
            self.patience_counter = 0
            
            # 保存检查点
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_f1': self.previous_best,
                'metrics': scores_dict
            }
            
            checkpoint_path = os.path.join(self.opt.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"✓ Best model saved at epoch {epoch+1} with Macro F1: {macro_f1:.4f}")
        else:
            self.patience_counter += 1
            # 根据阶段选择早停耐心值
            patience = self.early_stopping_patience_stage1 if self.current_stage == 1 else self.early_stopping_patience_stage2
            logging.info(f"No improvement. Patience: {self.patience_counter}/{patience}")
        
        # 早停检查
        patience = self.early_stopping_patience_stage1 if self.current_stage == 1 else self.early_stopping_patience_stage2
        if self.patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            return False
        
        return True
    
    def run(self):
        """运行训练"""
        logging.info("\n" + "="*80)
        logging.info("Improved v2 Training - Starting")
        logging.info("="*80)
        
        # 阶段1
        self.setup_stage1()
        for epoch in range(self.stage1_epochs):
            self.train_epoch(epoch)
            should_continue = self.validate(epoch)
            if not should_continue:
                break
        # 阶段2
        self.setup_stage2()
        for epoch in range(self.stage1_epochs, self.total_epochs):
            self.train_epoch(epoch)
            should_continue = self.validate(epoch)
            if not should_continue:
                break
        # 训练完成
        logging.info("\n" + "="*80)
        logging.info("Training Completed!")
        logging.info(f"Best F1 Score: {self.previous_best:.4f} at epoch {self.best_epoch}")
        logging.info("="*80)
def main():
    opt = Options().parse()
    # 创建检查点目录
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    
    # 初始化日志
    init_logging(opt.checkpoint_dir)
    setup_seed(1)
    
    logging.info("="*80)
    logging.info("Improved v2 Training - Advanced Two-Stage Training")
    logging.info("="*80)
    logging.info(f"Dataset: {opt.dataset}")
    logging.info(f"Checkpoint dir: {opt.checkpoint_dir}")
    logging.info(f"Batch size: {opt.batch_size}")
    # 创建训练器并运行
    trainer = V2ImprovedTrainer(opt)
    trainer.run()
if __name__ == '__main__':
    main()
