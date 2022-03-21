import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import imageio
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F


# ---------------------------------------------------------------
### Dataloader
img_size=512
class Dataset(BaseDataset):
    """CamVid数据集。进行图像读取，图像增强增强和图像预处理.

    Args:
        images_dir (str): 图像文件夹所在路径
        masks_dir (str): 图像分割的标签图像所在路径
        class_values (list): 用于图像分割的所有类别数
        augmentation (albumentations.Compose): 数据传输管道
        preprocessing (albumentations.Compose): 数据预处理
    """

    def __init__(
            self,
            images_dir,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing


    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        shape = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 图像增强应用
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
        return image, shape, os.path.basename(os.path.splitext(self.images_fps[i])[0])

    def __len__(self):
        return len(self.ids)


def get_augmentation():
    test_transform = [
        albu.Resize(img_size, img_size),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """进行图像预处理操作

    Args:
        preprocessing_fn (callbale): 数据规范化的函数
            (针对每种预训练的神经网络)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)


# 图像分割结果的可视化展示
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# ---------------------------------------------------------------
if __name__ == '__main__':

    DATA_DIR = '/data1/public_dataset/tianchi'

    x_test_dir = os.path.join(DATA_DIR, 'test/img')

    ENCODER = 'timm-efficientnet-b3'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = None
    DEVICE = 'cuda'

    # 加载最佳模型
    best_model = torch.load('/data1/xyh/checkpoints/tianchi/MANet_20/best_model.pth')
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # 对检测图像进行图像分割并进行图像可视化展示
    predict_dataset_vis = Dataset(
        x_test_dir,
        augmentation=get_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    test_loader = DataLoader(
        predict_dataset_vis,
        batch_size=1,
        num_workers=0,
        pin_memory=True
    )

    path = '/data1/xyh/checkpoints/tianchi/3_20_2'
    os.makedirs(path, exist_ok=True)
    from tqdm import tqdm
    for image, orishape, name in tqdm(test_loader):
        # 通过图像分割得到的0-1图像pr_mask
        orishape = torch.cat(orishape)
        pr_mask = best_model.predict(image.to(DEVICE))
        pr_mask = F.interpolate(pr_mask, size=orishape.numpy().tolist()[:-1], mode='bilinear', align_corners=False)
        pr_mask = pr_mask.sigmoid().squeeze().cpu().numpy()
        fake_mask = ((pr_mask> 0.5) * 255.).astype(np.uint8)
        cv2.imwrite(os.path.join(path, name[0]+'.png'), fake_mask)
