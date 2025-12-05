from data.transform import Transforms
from data.transform_enhanced import TransformsStandard  # 标准增强（推荐）
from data.spectral_preprocessing import SpectralAugmentationTransform  # 光谱增强
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from option import Options
import os
import logging

# 抑制PIL/Pillow的调试输出
Image.DEBUG = 0
logging.getLogger('PIL').setLevel(logging.WARNING)


def collate_fn_custom(batch):
    """
    Custom collate function to handle variable-sized tensors.
    Resizes all images to 256x256 if needed to ensure uniform batch shape.
    """
    # Extract individual items
    img1_list = [item['img1'] for item in batch]
    img2_list = [item['img2'] for item in batch]
    label_list = [item['label'] for item in batch]
    label_weak_list = [item['label_weak'] for item in batch]
    label_flag_list = [item['label_flag'] for item in batch]
    name_list = [item['name'] for item in batch]
    
    # Ensure all tensors have the same spatial dimensions (256, 256)
    target_size = (256, 256)
    
    img1_batch = []
    img2_batch = []
    label_batch = []
    label_weak_batch = []
    
    for img1, img2, label, label_weak in zip(img1_list, img2_list, label_list, label_weak_list):
        # Resize if needed
        if img1.shape[-2:] != target_size:
            img1 = torch.nn.functional.interpolate(
                img1.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
            ).squeeze(0)
        if img2.shape[-2:] != target_size:
            img2 = torch.nn.functional.interpolate(
                img2.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
            ).squeeze(0)
        if label.shape[-2:] != target_size:
            label = torch.nn.functional.interpolate(
                label.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode='nearest'
            ).squeeze(0).squeeze(0).long()
        if label_weak.shape[-2:] != target_size:
            label_weak = torch.nn.functional.interpolate(
                label_weak.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode='nearest'
            ).squeeze(0).squeeze(0).long()
        
        img1_batch.append(img1)
        img2_batch.append(img2)
        label_batch.append(label)
        label_weak_batch.append(label_weak)
    
    # Stack into batches
    img1_batch = torch.stack(img1_batch, dim=0)
    img2_batch = torch.stack(img2_batch, dim=0)
    label_batch = torch.stack(label_batch, dim=0)
    label_weak_batch = torch.stack(label_weak_batch, dim=0)
    label_flag_batch = torch.tensor(label_flag_list, dtype=torch.bool)
    
    return {
        'img1': img1_batch,
        'img2': img2_batch,
        'label': label_batch,
        'label_weak': label_weak_batch,
        'label_flag': label_flag_batch,
        'name': name_list
    }
class Load_Dataset(Dataset):
    def __init__(self, opt):
        super(Load_Dataset, self).__init__()
        self.opt = opt
        self.label_rate = opt.label_rate
        self.phase = opt.phase

        file_root = opt.dataroot

        self.img_names = open(os.path.join(file_root, opt.phase, 'list', f'{opt.phase}.txt')).read().splitlines()
        self.t1_paths = [file_root + '/' + opt.phase + '/A/' + x for x in self.img_names]
        self.t2_paths = [file_root + '/' + opt.phase + '/B/' + x for x in self.img_names]
        self.label_paths = [file_root + '/' + opt.phase + '/label/' + x for x in self.img_names]
        self.label_weak_paths = [file_root + '/' + opt.phase + '/label_weak/' + x for x in self.img_names]

        if self.phase == 'train':
            if opt.label_rate is not None:
                self.label_img_names = open(file_root + '/' + opt.phase + '/list/' + opt.phase + '_semi_' + opt.label_rate +'.txt').read().splitlines()
                self.label_t1_paths = [file_root + '/' + opt.phase + '/A/' + x for x in self.label_img_names]
                self.label_t2_paths = [file_root + '/' + opt.phase + '/B/' + x for x in self.label_img_names]
                self.label_label_paths = [file_root + '/' + opt.phase + '/label/' + x for x in self.label_img_names]

        self.normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # 原始增强: self.transform = transforms.Compose([Transforms()])
        # ✅ 修复：降低数据增强强度以加快收敛
        # 推荐使用标准增强（包含颜色增强和光谱增强）
        self.transform = transforms.Compose([
            TransformsStandard(
                enable_spectral=True,        # 启用光谱增强
                enable_multiscale=True,      # 启用多尺度处理
                enable_edge_enhance=False,   # 禁用边界锐化（过度增强）
                spectral_probability=0.5     # 从 0.8 改为 0.5（50%概率）
            )
        ])
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.t1_paths)

    def __getitem__(self, index):

        t1_path = self.t1_paths[index]
        t2_path = self.t2_paths[index]
        label_path = self.label_paths[index]
        label_weak_path = self.label_weak_paths[index]

        name = self.img_names[index]

        if self.phase == 'train':
            if self.label_rate is not None:
                if t1_path in self.label_t1_paths:
                    with_label = True
                else:
                    with_label = False
            else:
                with_label = True
        else:
            with_label = True

        img1 = Image.open(t1_path)
        img2 = Image.open(t2_path)
        label = np.array(Image.open(label_path).convert('L')) // 255
        label = Image.fromarray(label)
        label_weak = np.array(Image.open(label_weak_path).convert('L')) // 255
        label_weak = Image.fromarray(label_weak)

        if self.opt.phase == 'train':
            data = self.transform({'img1': img1, 'img2': img2, 'label': label, 'label_weak': label_weak})
            img1, img2, label, label_weak = data['img1'], data['img2'], data['label'], data['label_weak']

        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        label = torch.from_numpy(np.array(label))
        label_weak = torch.from_numpy(np.array(label_weak))

        input_dict = {'img1': img1, 'img2': img2, 'label': label, 'label_weak': label_weak, 'label_flag': with_label, 'name': name}

        return input_dict


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=opt.phase=='train',
                                                      pin_memory=True,
                                                      drop_last=opt.phase=='train',
                                                      num_workers=int(opt.num_workers),
                                                      collate_fn=collate_fn_custom)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    opt = Options().parse()
    train_loader = DataLoader(opt).load_data()
    for i, data in enumerate(train_loader):
        img1, img2, label, label_weak, label_flag, name = data['img1'], data['img2'], data['label'], data['label_weak'], data['label_flag'], data['fname']
        # print(img1.shape)
        # print(img2.shape)
        # print(label.shape)
        # print(label_weak.shape)
        # print(label_flag)
        if True in label_flag:
            true_indices = torch.nonzero(data['label_flag'], as_tuple=False)
            print(data['label_flag'])
            print(true_indices)
            print(true_indices.shape)
            print(data['img1'][true_indices].squeeze(1).shape)
            print(data['label'][true_indices].squeeze(1).shape)

        print('**************')

    # dataset = DataLoader(opt).dataset
    # for i in range(len(dataset)):
    #     daa = dataset.__getitem__(i)


