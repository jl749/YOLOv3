from typing import Tuple, Literal
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2

import yolov3.config as config
from yolov3.utils import cxcywh2xyxy, xywh2xyxy


# TODO: specify which loader (PASCAL?, COCO?)
def get_loaders(train_csv_path: Path,
                test_csv_path: Path,
                img_size: int,
                batch_size: int,
                num_workers: int,
                pin_memory=False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    from yolov3.datasets import VOCDataset, get_train_transforms, get_test_transforms

    train_dataset: Dataset = VOCDataset(
        train_csv_path,
        transform=get_train_transforms(img_size),
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        img_size=img_size
    )
    test_dataset: Dataset = VOCDataset(
        test_csv_path,
        transform=get_test_transforms(img_size),
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    train_eval_dataset: Dataset = VOCDataset(
        train_csv_path,
        transform=get_test_transforms(img_size),
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr) -> None:
    print("=> Loading checkpoint")
    _device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_file, map_location=_device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging, TODO: really? we load optimizer too
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_image(image: np.ndarray, boxes: np.ndarray, box_format: Literal['cxcywh', 'xyxy', 'xywh'] = 'cxcywh'):
    """Plots predicted bounding boxes on the image"""
    image = np.ascontiguousarray(image)  # HWC
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W, _ = image.shape
    _cls = boxes[:, 0].astype('int64')
    _conf = boxes[:, 1]
    _coor = boxes[:, 2:] * np.array([W, H, W, H])
    if box_format == 'cxcywh':
        _coor = cxcywh2xyxy(_coor)
    elif box_format == 'xywh':
        _coor = xywh2xyxy(_coor)

    _xyxy = _coor.astype('int64')
    for (cls, conf, xyxy) in zip(_cls, _conf, _xyxy):
        cv2.rectangle(image, xyxy[:2], xyxy[2:], color=(0, 0, 255), thickness=2)
        cv2.putText(image, f'{config.PASCAL_CLASSES[cls]}_{conf:.1f}',
                    xyxy[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('plot_image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO: fix
def plot_couple_examples(model, loader, threshold, nms_threshold, anchors, device="cuda"):
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, _, S, _, _ = out[i].shape  # BATCH_SIZE, 3, 52, 25
            # anchor = anchors[i]
            anchor = torch.tensor([*anchors[i]]).to(device) * S  # scale up anchor 0~1 --> 0~S
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=nms_threshold, obj_threshold=threshold, box_format="midpoint",
        )
        plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes)
