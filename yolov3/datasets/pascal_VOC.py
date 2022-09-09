from yolov3.config import ANCHORS, test_transforms
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from pathlib import Path

import cv2
from yolov3.utils import (
    cells_to_bboxes,
    label_anchor_likelihood,
    non_max_suppression as nms,
    plot_image
)


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
        self.image_size = image_size
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
        return - imgs(N, C, H, W),
               - targets([(3, 13, 13, 6), (3, 26, 26, 6), (3, 52, 52, 6)]) (obj_prob, x, y, w, h, class)
        """
        img_path, label_path = self.annotations.iloc[index]
        img_path = str(self.img_dir.joinpath(img_path))
        label_path = str(self.label_dir.joinpath(label_path))

        # ndmin=2 to maintain a 2D format even if there was one line of text
        # np roll(shift=4) (class, x, y, w, h) --> (x, y, w, h, class), xywh 0~1 normalized
        bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
        bboxes = np.roll(bboxes, shift=4, axis=1)

        # By default OpenCV uses BGR color space for color images
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # np.flip(image, 2)
        # cv2.imshow('image', image)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image: Tensor = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # S = 13, 26, 52
        # total_anchors // 3 = 3 (number of predictions we make)
        # targets.shape = [(3, 13, 13, 6), (3, 26, 26, 6), (3, 52, 52, 6)]
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]  # (obj_prob, x, y, w, h, class)

        for box in bboxes:  # for every labels (x, y, w, h, class)
            x, y, width, height, class_label = box
            iou_anchors = label_anchor_likelihood(torch.tensor(box[2:4]), self.anchors)  # iou between label box and all the anchor box candidates
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)  # argsort its likelihood indexes
            has_anchor = [False] * 3  # each scale should have one anchor??? why??? nms anyway???

            for anchor_idx in anchor_indices:  # high likelihood anchor box --> low likelihood anchor box
                scale_idx = anchor_idx // self.num_anchors_per_scale  # which scale you are looking at (small, medium, big)
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale  # which anchor in that certain scale?
                S = self.S[scale_idx]  # 13, 26, 52
                i, j = int(S * y), int(S * x)  # which cell

                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]  # object prob
                if not anchor_taken and not has_anchor[scale_idx]:  # obj prob == 0 && has_anchor[scale_idx] == F
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1  # set object prob to 1

                    x_cell, y_cell = S * x - j, S * y - i  # cell-wise x, y coordinate [0~1]
                    w_cell, h_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to the cell
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

    split_size = (13, 26, 52)

    # scale 0~1 normalized anchors
    scaled_anchors = torch.tensor(anchors) * torch.tensor(split_size)[:, None, None]  # (3, 3, 2)

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for imgs, labels in loader:
        boxes = []
        num_anchors_per_scale = labels[0].shape[1]  # 3

        # small, medium, big predictions into 0~1 scale again and plot bboxes
        for i in range(num_anchors_per_scale):  # i = 0, 1, 2
            anchor = scaled_anchors[i]  # (3, 2)
            print(anchor.shape)
            print(labels[i].shape)
            boxes += cells_to_bboxes(  # back to 0~1 scaled bboxes
                labels[i], is_preds=False, stride=labels[i].shape[2], anchors=anchor
            )[0]  # return list of size S*S*3
        boxes = nms(boxes, iou_threshold=1, obj_threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(imgs[0].permute(1, 2, 0).to("cpu"), boxes)  # RGB --> BRG


if __name__ == "__main__":
    _test()
