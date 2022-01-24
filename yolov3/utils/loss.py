from torchvision.ops import box_iou
# import pytorch_lightning as pl
import torch
import torch.nn as nn
from yolov3.utils.functions import iou_Coor


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()  # Binary cross entropy error
        self.cee = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    # predictions
    def forward(self, predictions, target, rescaled_anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(  # bce of noobj prob
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        rescaled_anchors = rescaled_anchors.reshape(1, 3, 1, 1, 2)  # anchor input = 3X2 (3 anchors each have w,h)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * rescaled_anchors], dim=-1)  # (BATCH_SIZE, 3, S, S, 4)
        # ious = box_iou(box_preds[obj], target[..., 1:5][obj]).detach()
        ious = iou_Coor(box_preds[obj], target[..., 1:5][obj]).detach()  # detached obj grad will not be tracked

        # predicted bbox is not gonna 100% align with the expected bbox --> ious*target_obj (likelihood actual obj inside predicted bbox)
        object_loss = self.bce(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(  # modify target instead of predictions
            (1e-16 + target[..., 3:5] / rescaled_anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.cee(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
