import torch
import torch.nn as nn

from yolov3.utils import iou


class YoloLoss(nn.Module):
    def __init__(self, rescaled_anchors):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cee = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.rescaled_anchors = rescaled_anchors
        self.anchor = None

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 2
        self.lambda_noobj = 1
        self.lambda_obj = 5
        self.lambda_box = 10

    def set_anchor(self, index):
        """ which scale anchor to use for the forward call """
        self.anchor = self.rescaled_anchors[index]

    def forward(self, predictions, targets):
        _device = predictions[0].device
        # Check where obj and noobj (we ignore if target == -1)
        obj = targets[..., 0] == 1  # in paper this is Iobj_i
        noobj = targets[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # bce of noobj prob
        no_object_loss = self.bce(  # predicted_no_obj prob VS target_no_obj prob (zeros)
            (predictions[..., 0:1][noobj]), (targets[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        if self.anchor is None:
            raise RuntimeError("please set_anchor first before calling forward")
        rescaled_anchors = self.anchor.reshape(1, 3, 1, 1, 2)  # anchor input = 3X2 (3 anchors each have w,h)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * rescaled_anchors], dim=-1)  # (BATCH_SIZE, 3, S, S, 4) [x, y, w, h] in 0~1 scale
        ious = iou(box_preds[obj].detach(), targets[obj][:, 1:5])

        # predicted bbox is not gonna 100% align with the expected bbox --> ious*target_obj_prob (likelihood actual obj inside predicted bbox)
        # target_obj_prob = 1 or 0
        object_loss = self.bce(self.sigmoid(predictions[obj][:, 0:1]), ious * targets[obj][:, 0:1])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates, map to 0~1 range

        # modify target instead of predictions (exp could overflow)
        # torch.exp(predictions[..., 3:5]) * rescaled_anchors
        targets[..., 3:5] = torch.log(
            (1e-16 + targets[..., 3:5] / rescaled_anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], targets[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.cee(
            (predictions[obj][:, 5:]), (targets[obj][:, 5].long()),
        )

        # print("__________________________________")
        # print(self.lambda_box * box_loss)
        # print(self.lambda_obj * object_loss)
        # print(self.lambda_noobj * no_object_loss)
        # print(self.lambda_class * class_loss)
        # print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
