from typing import Tuple
from yolov3.config import ANCHORS, test_transforms
import numpy as np
import os
import pandas as pd
import torch
from pathlib import Path

from PIL import Image, ImageFile
from yolov3.utils import (
    cells_to_bboxes,
    iou_wh as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file: Path, img_dir: Path, label_dir: Path,
                 anchors,
                 image_size=416,
                 split_size=(13, 26, 52),  # default should be immutable
                 num_classes=20,
                 transform=None,
                 ):
        self.annotations: pd.DataFrame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size: int = image_size
        self.transform = transform
        self.S: Tuple[int] = split_size

        # anchors[0] contains largest, anchors[2] smallest box (Big obj --> Medium obj --> Small obj)
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # 0~1 scale
        # how many width height ratio are there?
        # (3, 3, 2) --> (9, 2)

        self.num_anchors = self.anchors.shape[0]  # total 9 anchor boxes (3*3)
        self.num_anchors_per_scale = self.num_anchors // 3  # anchor boxes per ScalePrediction (small, medium. big)
        self.C = num_classes
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])  # full path to a txt file

        # ndmin=2 to maintain a 2D format even if there was one line of text
        # np roll to put class at the end index (x, y, w, h, class)  -->  it is like torch.permute()
        # (number of expected bbox from an img, 5)
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))  # make sure it is RGB

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # S = 13, 26, 52
        # total_anchors // 3 = 3 (number of predictions we make)
        # targets.shape = [(3, 13, 13, 6), (3, 26, 26, 6), (3, 52, 52, 6)] 3 anchor boxes for each prediction
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]  # (obj_prob, x, y, w, h, class)

        for box in bboxes:  # loop over expected bboxes
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)  # iou between label boc and all the anchor box candidates
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor

            for anchor_idx in anchor_indices:  # highest IoU to lowest
                scale_idx = anchor_idx // self.num_anchors_per_scale  # idx // 3  --> let you know which scale you are looking at (small, medium. big)
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
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, w_cell, h_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates  # set x,y,w,h [0~1] [0~S]
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True  # highest obj marked in this scale move on to the next scale

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:  # obj prob == 1 and IoU higher than threshold
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)  # img, ( (3, 13, 13, 6), (3, 26, 26, 6), (3, 52, 52, 6) )


def test():
    BASE_DIR = Path(__file__).parent.parent.parent

    anchors = ANCHORS  # (3, 3 ,2), contains width height ratio
    # we predict 3 times with 3 anchor boxes per cell each time

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

    scaled_anchors = torch.tensor(anchors) * torch.tensor(split_size).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    # torch.tensor(split_size).unsqueeze(1).unsqueeze(1) --> (3, 1, 1) -- repeat(1, 3, 2) --> (3, 3, 2)
    # [
    # [ [13, 13],
    # [13, 13],
    # [13, 13] ],
    # [ [26, 26],
    # [26, 26],
    # [26, 26] ],
    # [ [52, 52],
    # [52, 52],
    # [52, 52] ]
    # ]
    # (3, 3, 2) --> (3, 3, 2)

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        # small, medium, big predictions all into 0~1 scale again and plot bboxes
        for i in range(y[0].shape[1]):  # i = 0, 1, 2
            anchor = scaled_anchors[i]  # (3, 2)
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(  # back to 0~1 scale bboxes
                y[i], is_preds=False, split_size=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)  # RGB --> BRG


if __name__ == "__main__":
    test()
