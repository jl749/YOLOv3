from typing import List
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
import numpy as np
import cv2

from yolov3.utils import (
    cells_to_bboxes,
    label_anchor_likelihood,
    plot_image
)
from yolov3.config import ANCHORS


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_file: Path,
                 img_dir: Path,
                 label_dir: Path,
                 anchors,
                 img_size=416, num_classes=20,
                 transform=None,
                 ):
        self.annotations: pd.DataFrame = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size  # TODO: __get_item__ accordingly
        self.transform = transform
        self.S = [img_size // 32, img_size // 16, img_size // 8]

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
        return - imgs(N, C, H, W)
               - targets([(3, 13, 13, 6), (3, 26, 26, 6), (3, 52, 52, 6)]) (obj_prob, cx, cy, w, h, class)
               - annotations(num_boxes, 6) (IMG_INDEX, cx, cy, w, h, class)
        """
        img_path, label_path = self.annotations.iloc[index]
        img_path = str(self.img_dir.joinpath(img_path))
        label_path = str(self.label_dir.joinpath(label_path))

        # ndmin=2 to maintain a 2D format even if there was one line of text
        # np roll(shift=4) (class, cx, cy, w, h) --> (cx, cy, w, h, class), xywh 0~1 normalized
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
        # mapping bbox annotations into model output format
        # targets.shape = [(3, 13, 13, 6), (3, 26, 26, 6), (3, 52, 52, 6)]
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]  # (obj_prob, cx, cy, w, h, class)

        for box in bboxes:  # for every labels (cx, cy, w, h, class)
            x, y, width, height, class_label = box

            # iou between a current bbox and all the anchor box candidates
            iou_anchors = label_anchor_likelihood(torch.tensor(box[2:4]), self.anchors)  # (9,)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)  # argsort its likelihood indexes
            has_anchor = [False] * self.num_anchors_per_scale  # each scale prediction (small, medium, big) can have only one anchor box

            for anchor_idx in anchor_indices:  # high likelihood anchor box --> low likelihood anchor box
                # which scale does this anchor belong to (small, medium, big)
                scale_idx: int = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='trunc').item()
                anchor_on_scale: int = (anchor_idx % self.num_anchors_per_scale).item()  # which anchor within the selected scale?

                _S = self.S[scale_idx]  # 13, 26, 52
                i, j = int(_S * y), int(_S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]  # check if an anchor(13x13, 26x26, 32x32) is already reserved for a bbox in previous loop, see issue#4 limitation
                if not anchor_taken and not has_anchor[scale_idx]:  # obj prob == 0 && has_anchor[scale_idx] == F
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1  # set object prob to 1 (occupied)

                    x_cell, y_cell = _S * x - j, _S * y - i  # cell-wise x, y coordinate [0~1]
                    w_cell, h_cell = (
                        width * _S,
                        height * _S,
                    )  # can be greater than 1 since it's relative to the cell
                    _box_coordinates = torch.tensor(
                        [x_cell, y_cell, w_cell, h_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = _box_coordinates  # set x,y,w,h [0~1] [0~S]
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True  # obj marked in this scale move on to the next scale

                # one scale should have one anchor box, but if the other anchor box in the same scale has high enouch IoU
                # ignore it so that YOLO does not consider it as no object (no punish)
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:  # obj prob == 0 and IoU higher than threshold
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        annotations = torch.tensor(bboxes, dtype=torch.float32)
        return image, tuple(targets), annotations  # img, ( (3, 13, 13, 6), (3, 26, 26, 6), (3, 52, 52, 6) )

    @staticmethod
    def collate_fn(batch):
        img_batch = []
        target_batch = [[], [], []]
        annot_batch = []
        for b in batch:
            img_batch.append(b[0])
            target_batch[0].append(b[1][0])
            target_batch[1].append(b[1][1])
            target_batch[2].append(b[1][2])
            annot_batch.append(b[2])

        return torch.stack(img_batch, dim=0), [torch.stack(t) for t in target_batch], annot_batch


def _test():
    BASE_DIR = Path(__file__).parent.parent.parent
    img_size = 416
    anchors = ANCHORS  # (3, 3 ,2), contains width height ratio
    # 3 scale predictions, 3 anchor boxes per cell

    from yolov3.datasets import get_train_transforms, get_test_transforms
    transform = get_test_transforms(img_size=img_size)

    dataset = VOCDataset(
        BASE_DIR.joinpath("data/train.csv"),
        BASE_DIR.joinpath("data/images/"),
        BASE_DIR.joinpath("data/labels/"),
        anchors=anchors,
        transform=transform,
        img_size=img_size
    )

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for imgs, labels, _ in loader:
        boxes = torch.tensor([])
        num_anchors_per_scale: int = labels[0].shape[1]  # 3

        # small, medium, big predictions into 0~1 scale again and plot bboxes
        for i in range(num_anchors_per_scale):  # i = 0, 1, 2
            print(labels[i].shape)

            _label_bboxes = cells_to_bboxes(labels[i], is_preds=False)[0]  # batchsize = 1 for visualization, return (N, S*S*3, 6)

            boxes = torch.cat([boxes, _label_bboxes], dim=0)  # cxcywh

        from yolov3.utils import non_max_suppression as nms
        nms_boxes = nms(boxes.tolist(), iou_threshold=1, obj_threshold=0.7, box_format="midpoint")
        nms_boxes = np.array(nms_boxes)

        # import torchvision
        # boxes = boxes[boxes[:, 1] > 0.7]  # filter by conf_threshold  # TODO: torchvision nms takes xyxy only
        # _nms_indexes = torchvision.ops.nms(boxes=boxes[:, 2:], scores=boxes[:, 1], iou_threshold=1)
        # nms_boxes = boxes[_nms_indexes].numpy()

        print(nms_boxes)
        plot_image(imgs[0].permute(1, 2, 0), nms_boxes, box_format='cxcywh')  # RGB --> BRG


if __name__ == "__main__":
    _test()
