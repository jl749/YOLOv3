import torch
from torch.utils.data import DataLoader
import torchvision

import sys
from typing import List, Tuple
from tqdm import tqdm
from copy import copy

# takes both np array and torch tensor
def cxcywh2xyxy(bbox):
    tmp = copy(bbox)
    tmp[..., 0:2] -= tmp[..., 2:4] / 2
    tmp[..., 2:4] = tmp[..., 0:2] + tmp[..., 2:4]
    return tmp

# takes both np array and torch tensor
def xywh2xyxy(bbox):
    tmp = copy(bbox)
    tmp[..., 2:4] = tmp[..., 0:2] + tmp[..., 2:4]
    return tmp


def label_anchor_likelihood(input_wh: torch.Tensor, anchor_wh: torch.Tensor) -> torch.Tensor:
    """
    Calculates intersection over union using "width" and "height" of the boxes
    :param input_wh: width and height of the label
    :param anchor_wh: widths and heights of the predefined anchors
    :return: IoU between label wh ratio and anchor wh ratio (num_anchors,)
    """
    intersection = torch.min(input_wh[..., 0], anchor_wh[..., 0]) * torch.min(input_wh[..., 1], anchor_wh[..., 1])
    union = input_wh[..., 0] * input_wh[..., 1] + anchor_wh[..., 0] * anchor_wh[..., 1] - intersection
    return intersection / union


@torch.no_grad()
def get_evaluation_bboxes(
        loader: DataLoader,
        model: torch.nn.Module,
        nms_threshold: float,
        conf_threshold: float,
        anchors: List[List[Tuple[float]]],  # 0~S scaled
        # box_format="midpoint",
):
    """
    [POSTPROCESS]
    filter according to the thresholds then
    return final predicted(nms applied) bboxes and expected bboxes
    outputs can be used to calculate mAP of the model
    :param loader: DataLoader
    :param model:
    :param nms_threshold: iou threshold where predicted bboxes is correct (nms)
    :param anchors: pre-defined(K-means) anchors
    :param conf_threshold: threshold to remove predicted bboxes (independent of IoU)
    # :param box_format:
    :return: final predicted(nms applied) bboxes, expected bboxes
            [img_id, class_id. conf, x, y, x, y]
    """
    # make sure model is in eval before get bboxes
    model.eval()
    _device = next(model.parameters()).device

    train_idx = 0  # distinguish image
    all_pred_boxes = []
    all_true_boxes = []
    pbar = tqdm(iter(loader), file=sys.stdout)
    for img_batch, labels, _ in pbar:
        pbar.set_description("EVALUATING ...")
        N, _, H, W = img_batch.shape
        img_batch = img_batch.to(_device)
        labels = [l.to(_device) for l in labels]
        outputs: List[torch.Tensor] = model(img_batch)  # 3 scales (small, medium, large)

        bboxes = [torch.tensor([], device=_device) for _ in range(N)]  # prediction instances for mini-batches, reset every batch
        for i, predictions in enumerate(outputs):
            anchors_i = anchors[i]

            boxes_scale_i = cells_to_bboxes(  # cell-wise --> img-wise bbox information
                predictions, anchors_i, is_preds=True
            )  # (N, num_anchors * S * S, 1+5) with [class index, object score, cx, cy, w, h]
            boxes_scale_i[..., 2:6] = cxcywh2xyxy(boxes_scale_i[..., 2:6])

            for idx, (box) in enumerate(boxes_scale_i):  # loop batches 0 ~ N
                bboxes[idx] = torch.cat([bboxes[idx], box], dim=0)  # append ScalePrediction_i results

        # we just want one bbox for each label, not one for each scale
        # true_bboxes = expected lables
        true_bboxes = cells_to_bboxes(
            labels[-1], anchors[-1], is_preds=False
        )
        true_bboxes[..., 2:6] = cxcywh2xyxy(true_bboxes[..., 2:6])

        for idx in range(N):
            # nms_boxes = non_max_suppression(
            #     bboxes[idx],
            #     iou_threshold=iou_threshold,
            #     obj_threshold=conf_threshold,
            #     box_format="midpoint",  # TODO: no need to call cxcywh2xyxy
            # )  # filter predicted bboxes

            bbox = bboxes[idx]  # idx_batch bbox
            bbox = bbox[bbox[:, 1] > conf_threshold]  # filter by conf_threshold
            _nms_indexes = torchvision.ops.nms(
                boxes=bbox[:, 2:6] * torch.tensor([W, H, W, H], device=_device),  # 0~1 --> 0~img_size
                scores=bbox[:, 1], iou_threshold=nms_threshold)
            nms_boxes = bbox[_nms_indexes]

            # DBUGGING =================================================================================================
            # import cv2;import numpy as np
            # np_img = (img_batch[idx].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            # np_img = np.ascontiguousarray(np_img[..., ::-1])
            # for b in nms_boxes:
            #     xyxy = (b[2:6] * torch.tensor([W, H, W, H], device=_device)).long().tolist()
            #     cv2.rectangle(np_img, xyxy[:2], xyxy[2:], color=(0, 0, 255), thickness=2)
            # cv2.imshow('', np_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # ==========================================================================================================

            img_idx_info = torch.tensor([train_idx], device=_device)
            all_pred_boxes.append(torch.cat([img_idx_info.repeat(nms_boxes.shape[0], 1), nms_boxes], dim=1))
            all_true_boxes.append(
                torch.stack([torch.cat([img_idx_info, box])
                             for box in true_bboxes[idx] if box[1] > conf_threshold], dim=0)
            )
            train_idx += 1  # img index
            assert train_idx == len(all_pred_boxes) == len(all_true_boxes)
    return all_pred_boxes, all_true_boxes


# TODO: currently returning cxcywh in 0~S scale maybe returning xyxy format is better?
def cells_to_bboxes(predictions, anchors=None, is_preds=True):
    """
    Post-process logit output of the model (0 ~ S --> 0 ~ 1 scale)
    :param predictions: tensor of size (N, 3, S, S, 6) __ [obj_prob, cx, cy, w, h, class]
    :param anchors: the anchors used for the predictions (3, 2) must have been scaled up 0~1 --> 0~S
    :param is_preds: whether the input is predictions or the true bounding boxes (label)
    :return: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index, object score, bounding box coordinates
    """
    _device = predictions.device
    N, _, S, _, _ = predictions.shape

    num_anchors: int = 3 if anchors is None else len(anchors)
    box_predictions = predictions[..., 1:5]  # (obj_prob, cx, cy, w, h, class) --> (cx, cy, w, h)
    if is_preds:  # https://github.com/jl749/YOLOv3/issues/1#issuecomment-1016024032
        # x, y
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])

        # w, h
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors  # (N, 3, S, S, 2) * (N, 3, 1, 1, 2)

        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)  # best conditional prob, P(class_i | obj)
    else:  # if `predictions` is label data
        scores = predictions[..., 0:1]  # (BATCH_SIZE, 3, S, S, 1)
        best_class = predictions[..., 5:6]

    yv, xv = torch.meshgrid([torch.arange(S)] * 2, indexing="ij")  # TODO: consider pytorch version?
    yv = yv.to(_device)
    xv = xv.to(_device)
    """
    S = 13
    yv = tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
                [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
                [ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4],
                [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5],
                [ 6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6],
                [ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7],
                [ 8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8],
                [ 9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9],
                [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
                [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]])
    xv = tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]])
    """

    # cell wise location (0~1 within the cell) to grid-wise location (0~S)
    # mul 1 / S to normalize range 0~N --> 0~1
    cx = (xv.repeat(1, 3, 1, 1).unsqueeze(-1) + box_predictions[..., 0:1]) / S
    cy = (yv.repeat(1, 3, 1, 1).unsqueeze(-1) + box_predictions[..., 1:2]) / S
    w_h = box_predictions[..., 2:4] / S
    converted_bboxes = torch.cat([best_class, scores, cx, cy, w_h], dim=-1).reshape(N, num_anchors * S * S, 6)
    return converted_bboxes


# TODO: box_format? do we need it?
def iou(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor, box_format="midpoint") -> torch.Tensor:
    """
    Calculates intersection over union using "coordinates" of the boxes
    :param boxes_preds: Predictions of Bounding Boxes (BATCH_SIZE, 4)
    :param boxes_labels: Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    :param box_format: midpoint/corners, (cx,cy,w,h) or (x1,y1,x2,y2)
    :return: Intersection over union for all examples
    """
    x_hat = boxes_preds[..., 0:1]
    y_hat = boxes_preds[..., 1:2]
    w_hat = boxes_preds[..., 2:3]
    h_hat = boxes_preds[..., 3:4]

    x = boxes_labels[..., 0:1]
    y = boxes_labels[..., 1:2]
    w = boxes_labels[..., 2:3]
    h = boxes_labels[..., 3:4]

    if box_format == "midpoint":
        # lower left corner
        box1_x1 = x_hat - w_hat / 2
        box1_y1 = y_hat - h_hat / 2

        box2_x1 = x - w / 2
        box2_y1 = y - h / 2

        # upper right corner
        box1_x2 = x_hat + w_hat / 2
        box1_y2 = y_hat + h_hat / 2

        box2_x2 = x + w / 2
        box2_y2 = y + h / 2

    if box_format == "corners":
        # centre
        box1_x1 = x
        box1_y1 = y

        box2_x1 = x_hat
        box2_y1 = y_hat

        # upper right corner
        box1_x2 = w
        box1_y2 = h  # (N, 1)

        box2_x2 = w_hat
        box2_y2 = h_hat

    # find intersection corners
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


# TODO: box_format=cxcywh OR xyxy
# TODO: use torchvision if possible
def non_max_suppression(bboxes, iou_threshold: float, obj_threshold: float, box_format="corners"):
    """
    Non Max Suppression on given bboxes (S*S, 6)
    Input: A list of Proposal boxes X, corresponding confidence scores A and overlap threshold B.
    Output: A list of filtered proposals Y.
    :param bboxes: (S*S, 6) containing all bboxes with each bboxes, [class_pred, prob_score, x, y, w, h]
    :param iou_threshold: threshold where predicted bboxes is correct
    :param obj_threshold: threshold to remove predicted bboxes (independent of IoU)
    :param box_format: "midpoint" or "corners" used to specify bboxes
    :return: bboxes after performing NMS given a specific IoU threshold
    """
    assert isinstance(bboxes, list)  # np nor tensor

    bboxes = [box for box in bboxes if box[1] > obj_threshold]  # box[1] = obj confidence
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)  # sort descending
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [  # create new bboxes each loop, filtering low acc bbox
            box
            for box in bboxes
            if box[0] != chosen_box[0] or iou(  # no need to compare iou when classes are different
                torch.tensor(chosen_box[2:]).unsqueeze(0),
                torch.tensor(box[2:]).unsqueeze(0),
                box_format=box_format,
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)  # find the best box for each class detected

    return bboxes_after_nms
