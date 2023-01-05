import torch
import torch.nn as nn

from yolov3.utils import iou


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cee = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, predictions, targets, scaled_anchors):  # targets = (conf, cx, cy, w, h, cls)
        _device = predictions[0].device
        # ignore if target[..., 0] == -1
        obj = targets[..., 0] == 1  # in paper this is Iobj_i
        noobj = targets[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # bce of noobj prob
        no_object_loss = self.bce(  # predicted_no_obj prob VS target_no_obj prob (zeros)
            predictions[noobj][..., 0:1], targets[noobj][..., 0:1]
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        scaled_anchors = scaled_anchors.reshape(1, 3, 1, 1, 2)  # anchor input = 3X2 (3 anchors each have w,h)
        _box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * scaled_anchors], dim=-1)  # (BATCH_SIZE, 3, S, S, 4) [x, y] in 0~1 scale
        _ious = iou(_box_preds[obj].detach(), targets[obj][:, 1:5])

        # predicted bbox is not gonna 100% align with the expected bbox --> ious*target_obj_prob (likelihood actual obj inside predicted bbox)
        # target_obj_prob = 1 or 0
        object_loss = self.bce(predictions[obj][:, 0:1], _ious * targets[obj][:, 0:1])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates, map to 0~1 range

        # modify target instead of predictions (exp could overflow)
        # torch.exp(predictions[..., 3:5]) * rescaled_anchors
        targets[..., 3:5] = torch.log(
            (1e-16 + targets[..., 3:5] / scaled_anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[obj][..., 1:5], targets[obj][..., 1:5])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.cee(
            predictions[obj][:, 5:], targets[obj][:, 5].long()
        )

        return box_loss, object_loss, no_object_loss, class_loss
