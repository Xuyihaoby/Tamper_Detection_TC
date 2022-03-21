import os
from typing import Optional

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import os.path as osp
import random
import math

from segmentation_models_pytorch.utils import base

# ---------------------------------------------------------------
### 加载数据
img_size = 480
batch_size = 6


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
            masks_dir,
            augmentation=None,
            preprocessing=None,
            train=True
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, os.path.splitext(mask_id)[0] + '.png') for mask_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.train = train

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if osp.basename(self.masks_fps[i])[:2] == 'Tp':
            mask = cv2.imread(self.masks_fps[i].replace('.', '_gt.'), cv2.IMREAD_GRAYSCALE)
        else:
            mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255

        # 图像增强应用
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.train:
            if np.random.rand() > 0.5:
                image, mask = copy_move(img=image, img2=None, msk=mask)
            if np.random.rand() > 0.5:
                image2 = cv2.imread(np.random.choice(self.images_fps))
                image, mask = copy_move(img=image, img2=image2, msk=mask)
            if np.random.rand() > 0.5:
                image, mask = erase(image, mask)
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


# ---------------------------------------------------------------
# img_augmnet
def rand_bbox(size):
    if len(size) == 4:
        raise NotImplementedError
    elif len(size) == 3:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat_w = random.random() * 0.1 + 0.05
    cut_rat_h = random.random() * 0.1 + 0.05

    cut_w = int(W * cut_rat_w)
    cut_h = int(H * cut_rat_h)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def copy_move(img, msk, img2):
    size = img.shape
    if len(size) == 4:
        raise NotImplementedError
    elif len(size) == 3:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    if img2 is None:
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
        x_move = random.randrange(-bbx1, (W - bbx2))
        y_move = random.randrange(-bby1, (H - bby2))
        img[bbx1 + x_move:bbx2 + x_move, bby1 + y_move:bby2 + y_move, :] = img[bbx1:bbx2, bby1:bby2, :]
        msk[bbx1 + x_move:bbx2 + x_move, bby1 + y_move:bby2 + y_move] = np.ones_like(msk[bbx1:bbx2, bby1:bby2])
    else:
        img2 = cv2.resize(img2, (img_size, img_size))
        assert img.shape == img2.shape
        bbx1, bby1, bbx2, bby2 = rand_bbox(img2.shape)
        x_move = random.randrange(-bbx1, (W - bbx2))
        y_move = random.randrange(-bby1, (H - bby2))
        img[bbx1 + x_move:bbx2 + x_move, bby1 + y_move:bby2 + y_move, :] = img2[bbx1:bbx2, bby1:bby2, :]
        msk[bbx1 + x_move:bbx2 + x_move, bby1 + y_move:bby2 + y_move] = np.ones_like(msk[bbx1:bbx2, bby1:bby2])

    return img, msk

def erase(img, mask):
    mask_ = np.zeros(img.shape[:2], dtype="uint8")

    bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
    x0, y0 = (bbx1, bby1)
    x1, y1 = (bbx2, bby1)
    x2, y2 = (bbx2, bby2)
    x3, y3 = (bbx1, bby2)

    mask[y1:y2, x1:x0] = 1

    x_mid0, y_mid0 = int((x1 + x2) / 2), int((y1 + y2) / 2)
    x_mid1, y_mi1 = int((x0 + x3) / 2), int((y0 + y3) / 2)

    thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    cv2.line(mask_, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
    img = cv2.inpaint(img, mask_, 7, cv2.INPAINT_NS)
    return img, mask

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),

        albu.OneOf([
            albu.Rotate(limit=90, p=1.0),
            albu.Rotate(limit=270, p=1.0),
        ], p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
                albu.ChannelShuffle(always_apply=False, p=1.0)
            ],
            p=0.9,
        ),
        albu.Resize(img_size, img_size, p=1),
        # albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        # ToTensorV2()
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        albu.Resize(img_size, img_size, p=1),
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
        albu.Lambda(image=to_tensor)
    ]
    return albu.Compose(_transform)


class SoftBCEWithLogitsLoss(nn.Module):
    __constants__ = ["weight", "pos_weight", "reduction", "ignore_index", "smooth_factor"]

    def __init__(
            self,
            weight: Optional[torch.Tensor] = None,
            ignore_index: Optional[int] = -100,
            reduction: str = "mean",
            smooth_factor: Optional[float] = None,
            pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true
        y_pred = y_pred.squeeze(1)
        loss = F.binary_cross_entropy_with_logits(
            y_pred, soft_targets, self.weight, pos_weight=self.pos_weight, reduction="none"
        )

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss

    @property
    def __name__(self):
        return 'SoftBCEWithLogitsLoss'


# 重新定义评价指标mIoU
class IoU(base.Metric):
    __name__ = "iou_score"

    def __init__(self, eps=1e-7, threshold=0.5, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        _y_pr = (torch.sigmoid(y_pr) > self.threshold).int().detach()
        intersection = torch.sum(_y_pr * y_gt.unsqueeze(1))
        union = torch.sum(y_gt) + torch.sum(_y_pr) - intersection + self.eps
        return (intersection + self.eps) / union


# 重新定义train epoch


# $# 创建模型并训练
# ---------------------------------------------------------------
if __name__ == '__main__':

    # 数据集所在的目录
    DATA_DIR = '/data1/public_dataset/tianchi/'

    # 训练集
    x_train_dir = os.path.join(DATA_DIR, 'train/img')
    y_train_dir = os.path.join(DATA_DIR, 'train/mask')

    # 验证集
    x_valid_dir = os.path.join(DATA_DIR, 'train/img')
    y_valid_dir = os.path.join(DATA_DIR, 'train/mask')

    # ENCODER = 'se_resnet50'
    ENCODER = 'timm-efficientnet-b3'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = None  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'

    # 使用unet++模型
    model = smp.MAnet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # 加载训练数据集
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        train=True
    )

    # 加载验证数据集
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    loss = SoftBCEWithLogitsLoss(pos_weight=torch.tensor([3.]).cuda())
    metrics = [
        IoU(threshold=0.5),
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

    # 创建一个简单的循环，用于迭代数据样本
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # 进行20轮次迭代的模型训练
    max_score = 0
    root_model_path = '/data1/xyh/checkpoints/tianchi/MANet_20/'
    os.makedirs(root_model_path, exist_ok=True)
    for i in range(0, 30):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # 每次迭代保存下训练最好的模型
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            path_name = root_model_path + 'best_model.pth'
            torch.save(model, path_name)
            print('Model saved!')

        if i == 10:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
        if i == 15:
            optimizer.param_groups[0]['lr'] = 1e-6
            print('Decrease decoder learning rate to 1e-6!')
        if i == 20:
            optimizer.param_groups[0]['lr'] = 1e-7
            print('Decrease decoder learning rate to 1e-5!')
        if i % 10 == 9:
            pth_name = root_model_path + 'epoch_' + str(i) + '_.pth'
            torch.save(model, pth_name)
            print('model_save')
    path_name = root_model_path + 'last_model.pth'
    torch.save(model, path_name)
    print('last_model_save')
