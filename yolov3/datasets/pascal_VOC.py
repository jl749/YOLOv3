from yolov3.config import ANCHORS, test_transforms
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torchvision
from pathlib import Path

import cv2
from yolov3.utils import (
    cells_to_bboxes,
    label_anchor_likelihood,
    plot_image
)

from typing import List


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_file: Path,
                 img_dir: Path,
                 label_dir: Path,
                 anchors,
                 image_size=416, split_size=(13, 26, 52), num_classes=20,
                 transform=None,
                 ):
        self.annotations: pd.DataFrame = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size  # TODO: __get_item__ accordingly
        self.transform = transform
        self.S = split_size

        # anchors[0] contains the largest, anchors[2] contains the smallest object scales
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # (3, 3, 2) --> (9, 2)

        self.num_anchors = self.anchors.shape[0]  # total 9 anchor boxes
        self.num_anchors_per_scale = self.num_anchors // 3  # anchor boxes per ScalePrediction (small, medium. big)
        self.C = num_classes
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """
        read annotation file and assign it to the best fitting anchor box and return it as label
        return - imgs(N, C, H, W),
               - targets([(3, 13, 13, 6), (3, 26, 26, 6), (3, 52, 52, 6)]) (obj_prob, x, y, w, h, class)
        """
        img_path, label_path = self.annotations.iloc[index]
        img_path = str(self.img_dir.joinpath(img_path))
        label_path = str(self.label_dir.joinpath(label_path))

        # ndmin=2 to maintain a 2D format even if there was one line of text
        # np roll(shift=4) (class, x, y, w, h) --> (x, y, w, h, class), xywh 0~1 normalized
        bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)  # (N, 5)
        bboxes = np.roll(bboxes, shift=4, axis=1)

        # By default OpenCV uses BGR color space for color images
        image = cv2.imread(img_path)  # HWC
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # np.flip(image, 2)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image: Tensor = augmentations["image"]  # CHW
            bboxes: List[tuple] = augmentations["bboxes"]

        # S = 13, 26, 52
        # total_anchors // 3 = 3 (number of predictions we make)
        # targets.shape = [(3, 13, 13, 6), (3, 26, 26, 6), (3, 52, 52, 6)]
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]  # (obj_prob, x, y, w, h, class)

        for box in bboxes:  # for every labels (x, y, w, h, class)
            x, y, width, height, class_label = box

            # iou between label box and all the anchor box candidates
            iou_anchors = label_anchor_likelihood(torch.tensor(box[2:4]), self.anchors)  # (9,)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)  # argsort its likelihood indexes
            has_anchor = [False] * self.num_anchors_per_scale  # each scale prediction (small, medium, big) can have only one anchor box

            for anchor_idx in anchor_indices:  # high likelihood anchor box --> low likelihood anchor box
                # scale_idx = anchor_idx // self.num_anchors_per_scale
                scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale,
                                      rounding_mode='trunc')  # which scale you are looking at (small, medium, big)
                S = self.S[scale_idx]  # 13, 26, 52
                i, j = int(S * y), int(S * x)  # which cell

                anchor_on_scale = anchor_idx % self.num_anchors_per_scale  # which anchor within the selected scale?
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]  # object prob
                if not anchor_taken and not has_anchor[scale_idx]:  # obj prob == 0 && has_anchor[scale_idx] == F
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1  # set object prob to 1 (occupied)

                    x_cell, y_cell = S * x - j, S * y - i  # cell-wise x, y coordinate [0~1]
                    w_cell, h_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to the cell
                    x_cell -= w_cell / 2  # IMPORTANT: cxcywh --> xywh
                    y_cell -= h_cell / 2
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, w_cell, h_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates  # set x,y,w,h [0~1] [0~S]
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True  # obj marked in this scale move on to the next scale

                # one scale should have one anchor box, but if the other anchor box in the same scale has high enouch IoU
                # ignore it so that YOLO does not consider it as no object (no punish)
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:  # obj prob == 0 and IoU higher than threshold
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)  # img, ( (3, 13, 13, 6), (3, 26, 26, 6), (3, 52, 52, 6) )


def _test():
    BASE_DIR = Path(__file__).parent.parent.parent

    anchors = ANCHORS  # (3, 3 ,2), contains width height ratio
    # 3 scale predictions, 3 anchor boxes per cell

    transform = test_transforms  # config

    dataset = VOCDataset(
        BASE_DIR.joinpath("data/train.csv"),
        BASE_DIR.joinpath("data/images/"),
        BASE_DIR.joinpath("data/labels/"),
        split_size=(13, 26, 52),
        anchors=anchors,
        transform=transform,
    )

    S = [13, 26, 52]  # [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
    # scale 0~1 normalized anchors --> 0~S
    scaled_anchors = torch.tensor(anchors) * torch.tensor(S)[:, None, None]  # (3, 3, 2)

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for imgs, labels in loader:
        boxes = torch.tensor([])
        num_anchors_per_scale = labels[0].shape[1]  # 3

        # small, medium, big predictions into 0~1 scale again and plot bboxes
        for i in range(num_anchors_per_scale):  # i = 0, 1, 2
            anchor = scaled_anchors[i]  # (3, 2)
            print(anchor.shape)
            print(labels[i].shape)

            _S = labels[i].shape[2]
            _label_bboxes = cells_to_bboxes(labels[i], is_preds=False, anchors=anchor)[0]  # batchsize = 1 for visualization, return (N, S*S*3, 6)
            _label_bboxes[:, 2:] = _label_bboxes[:, 2:] / _S

            boxes = torch.cat([boxes, _label_bboxes], dim=0)
        # from yolov3.utils import non_max_suppression as nms,
        # boxes = nms(boxes, iou_threshold=1, obj_threshold=0.7, box_format="midpoint")
        boxes[:, 4:6] = boxes[:, 2:4] + boxes[:, 4:6]  # xywh --> xyxy
        boxes = boxes[boxes[:, 1] > 0.7]  # filter by conf_threshold
        _nms_indexes = torchvision.ops.nms(boxes=boxes[:, 2:], scores=boxes[:, 1], iou_threshold=1)
        nms_boxes = boxes[_nms_indexes]

        print(nms_boxes)
        # plot_image(imgs[0].permute(1, 2, 0), nms_boxes.numpy(), box_format='xyxy')  # RGB --> BRG

        # or
        nms_boxes[:, 4:6] = nms_boxes[:, 4:6] - nms_boxes[:, 2:4]  # xyxy --> xywh
        plot_image(imgs[0].permute(1, 2, 0), nms_boxes.numpy(), box_format='xywh')  # RGB --> BRG


if __name__ == "__main__":
    _test()
