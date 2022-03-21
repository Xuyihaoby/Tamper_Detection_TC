from typing import Callable, List, Tuple
from catalyst import utils
import os
import os.path as osp
import torch
import catalyst
import numpy as np
from torch.utils.data import DataLoader
from catalyst_datasets import compose, resize_transforms, hard_transforms, post_transforms, pre_transforms
from catalyst_loader import get_loaders
from catalyst.dl import SupervisedRunner
import ttach as tta
import cv2 as cv

import segmentation_models_pytorch as smp

from torch import nn
from pathlib import Path
from catalyst_datasets import SegmentationDataset

if __name__ == '__main__':
    img_size = 480
    num_workers: int = 4
    batch_size = 4
    threshold = 0.5
    max_count = 5
    ROOT = Path('/data1/public_dataset/tianchi')
    ENCODER = 'timm-efficientnet-b5'
    dst_path = '/data1/xyh/checkpoints/tianchi/pred_cata'
    os.makedirs(dst_path, exist_ok=True)
    test_image_path = ROOT / "test/img"
    logdir = "./logs/segmentation"
    alid_transforms = compose([pre_transforms(image_size=img_size), post_transforms()])
    TEST_IMAGES = sorted(test_image_path.glob("*.jpg"))
    valid_transforms = compose([pre_transforms(image_size=img_size), post_transforms()])
    # create test dataset
    test_dataset = SegmentationDataset(
        TEST_IMAGES,
        transforms=valid_transforms
    )

    infer_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    device = utils.get_device()
    model = smp.DeepLabV3Plus(encoder_name=ENCODER, classes=1)
    runner = SupervisedRunner(model=model, device=device, input_key="image", input_target_key="mask")
    # this get predictions for the whole loader
    predictions = np.vstack(list(map(
        lambda x: x["logits"].cpu().numpy(),
        runner.predict_loader(loader=infer_loader, resume=f"{logdir}/checkpoints/best.pth")
    )))

    for i, (features, logits) in enumerate(zip(test_dataset, predictions)):
        image = utils.tensor_to_ndimage(features["image"])

        mask_ = torch.from_numpy(logits[0]).sigmoid()
        mask = utils.detach(mask_ > threshold).astype("float")

        orishape = cv.imread(str(test_image_path/ features['filename'])).shape
        _mask = cv.resize(mask, (orishape[1], orishape[0]))

        fake_mask = ((_mask > 0.5) * 255.).astype(np.uint8)
        cv.imwrite(os.path.join(dst_path, osp.splitext(features["filename"])[0] + '.png'), fake_mask)

    # tta
    # model = smp.DeepLabV3Plus(encoder_name=ENCODER, classes=1)
    # tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode="mean")
    #
    # tta_runner = SupervisedRunner(
    #     model=tta_model,
    #     device=utils.get_device(),
    #     input_key="image"
    # )
    # infer_loader = DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=num_workers
    # )
    #
    # batch = next(iter(infer_loader))
    #
    # # predict_batch will automatically move the batch to the Runner's device
    # tta_predictions = tta_runner.predict_batch(batch)
    #
    # image = utils.tensor_to_ndimage(batch["image"][0])
    #
    # mask_ = tta_predictions["logits"][0, 0].sigmoid()
    # mask = utils.detach(mask_ > threshold).astype("float")