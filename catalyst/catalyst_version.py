from typing import Callable, List, Tuple
from catalyst import utils
import os
import torch
import catalyst

print(f"torch: {torch.__version__}, catalyst: {catalyst.__version__}")

from catalyst_datasets import compose, resize_transforms, hard_transforms, post_transforms, pre_transforms
from catalyst_loader import get_loaders

import segmentation_models_pytorch as smp

from torch import nn

from catalyst.contrib.nn import DiceLoss, IoULoss
from torch import optim

from catalyst.contrib.nn import RAdam, Lookahead
from catalyst.dl import SupervisedRunner
from pathlib import Path

from catalyst.dl import DiceCallback, IouCallback, \
  CriterionCallback, MetricAggregationCallback
from catalyst.contrib.callbacks import DrawMasksCallback


if __name__ == '__main__':
    # config
    batch_size = 4
    img_size= 480
    learning_rate = 0.001
    encoder_learning_rate = 0.0005
    num_epochs = 20
    logdir = "./logs/segmentation"
    ENCODER = 'timm-efficientnet-b5'
    ENCODER_WEIGHTS = 'imagenet'

    ROOT = Path('/data1/public_dataset/tianchi')
    train_image_path = ROOT / "train/img"
    train_mask_path = ROOT / "train/mask"
    test_image_path = ROOT / "test/img"

    ALL_IMAGES = sorted(train_image_path.glob("*.jpg"))
    ALL_MASKS = sorted(train_mask_path.glob("*.png"))

    # random initialize
    SEED = 42
    utils.set_global_seed(SEED)
    utils.prepare_cudnn(deterministic=True)

    # transforms
    train_transforms = compose([
        resize_transforms(image_size=img_size),
        hard_transforms(),
        post_transforms()
    ])
    valid_transforms = compose([pre_transforms(image_size=img_size), post_transforms()])

    # show_transforms = compose([resize_transforms(), hard_transforms()])
    # loader
    loaders = get_loaders(
        images=ALL_IMAGES,
        masks=ALL_MASKS,
        random_state=SEED,
        train_transforms_fn=train_transforms,
        valid_transforms_fn=valid_transforms,
        batch_size=batch_size
    )

    # model
    model = smp.DeepLabV3Plus(encoder_name=ENCODER, classes=1)

    # loss
    criterion = {
        "dice": DiceLoss(),
        "iou": IoULoss(),
        "bce": nn.BCEWithLogitsLoss()
    }

    # model settings

    # Since we use a pre-trained encoder, we will reduce the learning rate on it.
    layerwise_params = {"encoder*": dict(lr=encoder_learning_rate, weight_decay=0.00003)}

    # This function removes weight_decay for biases and applies our layerwise_params
    model_params = utils.process_model_params(model, layerwise_params=layerwise_params)

    # Catalyst has new SOTA optimizers out of box
    base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
    optimizer = Lookahead(base_optimizer)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)

    # monitor
    device = utils.get_device()
    runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")

    # train
    callbacks = [
        # Each criterion is calculated separately.
        CriterionCallback(
            input_key="mask",
            prefix="loss_dice",
            criterion_key="dice"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_iou",
            criterion_key="iou"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_bce",
            criterion_key="bce"
        ),

        # And only then we aggregate everything into one loss.
        MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
            # because we want weighted sum, we need to add scale for each loss
            # metrics={"loss_bce": 0.8},
            metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
        ),

        # metrics
        DiceCallback(input_key="mask"),
        IouCallback(input_key="mask"),
        # visualization
        DrawMasksCallback(output_key='logits',
                          input_image_key='image',
                          input_mask_key='mask',
                          summary_step=50
                          )
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        # our dataloaders
        loaders=loaders,
        # We can specify the callbacks list for the experiment;
        callbacks=callbacks,
        # path to save logs
        logdir=logdir,
        num_epochs=num_epochs,
        # save our best checkpoint by IoU metric
        main_metric="iou",
        # IoU needs to be maximized.
        minimize_metric=False,
        # prints train logs
        verbose=True,
    )



