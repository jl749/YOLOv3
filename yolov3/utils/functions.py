from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

import yolov3.config as config
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def iou_Coor(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor,
             box_format="midpoint") -> torch.Tensor:
    """
    Calculates intersection over union using "coordinates" of the boxes
    :param boxes_preds: Predictions of Bounding Boxes (BATCH_SIZE, 4)
    :param boxes_labels: Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    :param box_format: midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    :return: Intersection over union for all examples
    """
    # box_preds = shape=(1, 7, 7, 4)

    # boxes_preds[..., 0:1] --> (1, 7, 7, 1)
    # boxes_preds[..., 0] --> (1, 7, 7)
    x_hat = boxes_preds[..., 0:1]  # (1, 7, 7, 1)
    y_hat = boxes_preds[..., 1:2]
    w_hat = boxes_preds[..., 2:3]
    h_hat = boxes_preds[..., 3:4]

    x = boxes_labels[..., 0:1]  # (1, 7, 7, 1)
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


def iou_wh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculates intersection over union using "width" and "height" of the boxes
    :param boxes1: width and height of the first bounding boxes
    :param boxes2: width and height of the second bounding boxes
    :return: IoU of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
            boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def cells_to_bboxes(predictions, anchors, split_size, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted
    :param predictions: tensor of size (N, 3, S, S, 6) __ [obj_prob, x, y, w, h, class]
    :param anchors: the anchors used for the predictions (3, 2) must have been scaled up 0~1 --> 0~S
    :param split_size: the number of cells the image is divided in (S x S)
    :param is_preds: whether the input is predictions or the true bounding boxes
    :return: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index, object score, bounding box coordinates
    """
    S = split_size
    BATCH_SIZE = predictions.shape[0]

    num_anchors = len(anchors)  # 3
    box_predictions = predictions[..., 1:5]  # x, y, w, h
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])  # x, y
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors  # w, h
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)  # best conditional prob, P(class_i | obj)
    else:
        scores = predictions[..., 0:1]  # (BATCH_SIZE, 3, S, S, 1)
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
            .repeat(predictions.shape[0], 3, S, 1)
            .unsqueeze(-1)
            .to(predictions.device)
    )  # (1, 3, S, S, 1)

    # 0 ~ 1 scale
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


# torchvision.ops.nms()
def non_max_suppression(bboxes, iou_threshold: float, threshold: float, box_format="corners"):
    """
    Non Max Suppression on given bboxes (S*S, 6)
    Input: A list of Proposal boxes X, corresponding confidence scores A and overlap threshold B.
    Output: A list of filtered proposals Y.
    :param bboxes: (S*S, 6) containing all bboxes with each bboxes, [class_pred, prob_score, x, y, w, h]
    :param iou_threshold: threshold where predicted bboxes is correct
    :param threshold: threshold to remove predicted bboxes (independent of IoU)
    :param box_format: "midpoint" or "corners" used to specify bboxes
    :return: bboxes after performing NMS given a specific IoU threshold
    """
    assert type(bboxes) == list  # np nor tensor

    bboxes = [box for box in bboxes if box[1] > threshold]  # box[1] = obj confidence
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)  # sort descending
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [  # create new bboxes each loop, filtering low acc bbox
            box
            for box in bboxes
            if box[0] != chosen_box[0] or iou_Coor(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)  # find the best box for each class detected

    return bboxes_after_nms


##################################################################################################
def mean_average_precision(pred_boxes: List[List[float]],  # get_evaluation_bboxes output
                           true_boxes: List[List[float]],  # get_evaluation_bboxes output
                           iou_threshold=0.5, box_format="midpoint", num_classes=20
                           ) -> float:
    """
    calculate mAP
    :param pred_boxes: predicted bboxes [[train_idx, class_pred, obj_prob, x, y, w, h], ...]
    :param true_boxes: expected bboxes
    :param iou_threshold: threshold where predicted bboxes is correct
    :param box_format: "midpoint" or "corners" used to specify bboxes
    :param num_classes: number of classes
    :return: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = iou_Coor(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def plot_couple_examples(model, loader, threshold, iou_threshold, anchors, device="cuda"):
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
                out[i], anchor, split_size=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_threshold, threshold=threshold, box_format="midpoint",
        )
        plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes)


def check_class_accuracy(model, loader: torch.utils.data.DataLoader, threshold):
    """
    evaluate model
    :param loader: test_loader
    :param threshold: P(obj) threshold
    """
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):  # 4952/BATCH_SIZE loops
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1  # Identity obj_i
            noobj = y[i][..., 0] == 0  # Identity noobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class / (tot_class_preds + 1e-16)) * 100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj / (tot_noobj + 1e-16)) * 100:2f}%")
    print(f"Obj accuracy is: {(correct_obj / (tot_obj + 1e-16)) * 100:2f}%")
    model.train()


def get_loaders(train_csv_path, test_csv_path) -> DataLoader:
    from yolov3.datasets import VOCDataset

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset: Dataset = VOCDataset(
        train_csv_path,
        transform=config.train_transforms,
        split_size=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],  # stride 32, 16, 8
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    test_dataset: Dataset = VOCDataset(
        test_csv_path,
        transform=config.test_transforms,
        split_size=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    train_eval_dataset: Dataset = VOCDataset(
        train_csv_path,
        transform=config.test_transforms,
        split_size=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader


def get_evaluation_bboxes(
        loader: DataLoader,
        model: torch.nn.Module,
        iou_threshold: float, threshold: float,
        anchors: List[List[Tuple[float]]],
        box_format="midpoint",
        device="cuda",
):
    """
    filter according to the thresholds then
    return final predicted(nms applied) bboxes and expected bboxes
    outputs can be used to calculate mAP of the model
    :param loader: DataLoader
    :param model:
    :param iou_threshold: threshold where predicted bboxes is correct
    :param anchors: pre-defined(K-means) anchors
    :param threshold: threshold to remove predicted bboxes (independent of IoU)
    :param box_format:
    :param device:
    :return: final predicted(nms applied) bboxes, expected bboxes
    """
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []

    # labels = [(BATCH_SIZE, 3, 13, 13, 6), (BATCH_SIZE, 3, 26, 26, 6), (BATCH_SIZE, 3, 52, 52, 6)]
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)  # img

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):  # three predictions within each three scales 13, 26, 51
            S = predictions[i].shape[2]  # predictions[i]`s split_size

            # anchor = anchor boxes info in scale_i
            anchor = torch.tensor([*anchors[i]]).to(device) * S  # scale up anchor 0~1 --> 0~S
            boxes_scale_i = cells_to_bboxes(  # cell-wise --> img-wise bbox information ???SHAPE
                predictions[i], anchor, split_size=S, is_preds=True
            )  # (N, num_anchors * S * S, 1+5) with [class index, object score, bounding box coordinates]
            for idx, (box) in enumerate(boxes_scale_i):  # idx 0 ~ N
                bboxes[idx] += box  # append ScalePrediction_i results

        # we just want one bbox for each label, not one for each scale
        # true_bboxes = expected lables
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, split_size=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )  # filter predicted bboxes

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1  # img index

    model.train()  # revert to train mod
    return all_pred_boxes, all_true_boxes


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
##################################################################################################


# def get_bboxes(
#         loader: torch.utils.data.DataLoader,
#         model: torch.nn.Module,
#         iou_threshold: float,
#         threshold: float,
#         pred_format="cells",
#         box_format="midpoint",
#         device="cuda",
# ):
#     """
#     obtain bounding boxes using given dataset and model
#     :param loader: train/test DataLoader
#     :param model: model to be used for predictions
#     :param iou_threshold: iou threshold to be used on non-max suppression
#     :param threshold: object confidence threshold
#     :param pred_format:
#     :param box_format:
#     :param device:
#     :return: predicted box, expected box
#     """
#     all_pred_boxes = []
#     all_true_boxes = []
#
#     # make sure model is in eval before get bboxes
#     model.eval()
#     train_idx = 0
#
#     for batch_idx, (x, labels) in enumerate(loader):
#         x = x.to(device)  # train image, (BATCH_SIZE, 3, 448, 448)
#         labels = labels.to(device)  # train expected labels, (BATCH_SIZE, 7, 7, 30)
#
#         with torch.no_grad():
#             predictions = model(x)
#
#         batch_size = x.shape[0]
#         true_bboxes = cellboxes_to_boxes(
#             labels)  # [predicted_class, best_confidence, ...converted x,y,w,h ...], (BATCH_SIZE, S*S, 6)
#         bboxes = cellboxes_to_boxes(predictions)
#
#         for idx in range(batch_size):
#             nms_boxes = non_max_suppression(
#                 bboxes[idx],  # (BATCH_SIZE, S*S, 6)  -->  (S*S, 6)
#                 iou_threshold=iou_threshold,
#                 threshold=threshold,
#                 box_format=box_format,
#             )
#             if len(nms_boxes) > 0:
#                 print(nms_boxes)
#             # if batch_idx == 0 and idx == 0:
#             #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
#             #    print(nms_boxes)
#
#             for nms_box in nms_boxes:
#                 all_pred_boxes.append([train_idx] + nms_box)
#
#             for box in true_bboxes[idx]:
#                 # many will get converted to 0 pred
#                 if box[1] > threshold:
#                     all_true_boxes.append([train_idx] + box)
#
#             train_idx += 1
#
#     model.train()
#     return all_pred_boxes, all_true_boxes


# def convert_cellboxes(predictions, S=7):  # train/predicted labels, (BATCH_SIZE, 7, 7, 30)
#     """
#     Converts bounding boxes output from Yolo with
#     an image split size of S into entire image ratios
#     rather than relative to cell ratios. (0~1 --scale up--> 0~448)
#     output --> (BATCH_SIZE, 7, 7, 6) [predicted_class, best_confidence, ...converted x,y,w,h ...]
#     """
#
#     predictions = predictions.to("cpu")
#     batch_size = predictions.shape[0]
#     predictions = predictions.reshape(batch_size, 7, 7, 30)
#     bboxes1 = predictions[..., 21:25]  # x1,y1,w1,h1
#     bboxes2 = predictions[..., 26:30]  # x2,y2,w2,h2
#     scores = torch.cat(
#         (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
#     )  # (2, BATCH_SIZE, 7, 7) ---> 1 or 0
#
#     best_box = scores.argmax(0).unsqueeze(-1)  # (BATCH_SIZE, 7, 7, 1)
#     best_boxes = bboxes1 * (
#                 1 - best_box) + best_box * bboxes2  # only best boxes (1 or 0) per cell  --> (BATCH_SIZE, 7, 7, 4)
#     cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)  # (BATCH_SIZE, 7, 7->(arange(7)), 1)
#
#     # find x,y coordinates (grid-wise 0~1 --scale down--> whole-image-wise 0~1)
#     x = 1 / S * (best_boxes[..., :1] + cell_indices)  # (BATCH_SIZE, 7, 7, 1)
#     # https://github.com/jl749/myYOLO/issues/3#issuecomment-1012911442
#     y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))  # (BATCH_SIZE, 7, 7, 1), permute swap 7, 7
#     w_h = 1 / S * best_boxes[..., 2:4]  # (BATCH_SIZE, 7, 7, 2)  rescale w,h as x,y has been resacled too
#
#     converted_bboxes = torch.cat((x, y, w_h), dim=-1)  # (BATCH_SIZE, 7, 7, 4)
#
#     # among 20 classes find the highest (BATCH_SIZE, 7, 7)  --unsqueeze(-1)-->  (BATCH_SIZE, 7, 7, 1)
#     predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
#     best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
#         -1
#     )  # (BATCH_SIZE, 7, 7, 1)
#
#     # wrap the final converted output
#     converted_preds = torch.cat(
#         (predicted_class, best_confidence, converted_bboxes), dim=-1
#     )  # (BATCH_SIZE, 7, 7, 6)
#
#     return converted_preds


# def cellboxes_to_boxes(out, S=7):  # out = train/predicted labels, (BATCH_SIZE, 7, 7, 30)
#     """
#     (BATCH_SIZE, S, S, 6)  -->  (BATCH_SIZE, S*S, 6)
#     :param out: train/predicted labels, (BATCH_SIZE, 7, 7, 30)
#     :param S: split_size
#     :return: reshaped [predicted_class, best_confidence, ...converted x,y,w,h ...] info, (BATCH_SIZE, S*S, 6)
#     """
#     batch_size = out.shape[0]
#
#     converted_pred = convert_cellboxes(out).reshape(batch_size, S * S, -1)  # (BATCH_SIZE, 7, 7, 6)
#     converted_pred[..., 0] = converted_pred[..., 0].long()  # long() == self.to(torch.int64)
#     all_bboxes = []
#
#     for ex_idx in range(batch_size):
#         bboxes = []
#         # for every batch for every cell
#         for bbox_idx in range(S * S):
#             # converted_pred[0, 0, :]  -->  (1, 6)
#             bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
#         all_bboxes.append(bboxes)
#
#     return all_bboxes  # (BATCH_SIZE, S*S, 6)
