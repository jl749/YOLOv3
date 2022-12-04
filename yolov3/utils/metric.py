from collections import Counter
from itertools import chain
from typing import List

from tqdm import tqdm
import torch
import torchvision


# TODO: make sure this work
def check_class_accuracy(model: torch.nn.Module,
                         loader: torch.utils.data.DataLoader,
                         conf_threshold: float):
    """
    evaluate model
    :param model: model to be tested
    :param loader: test_loader
    :param conf_threshold: P(obj) threshold
    """
    model.eval()

    _device = next(model.parameters()).device
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):  # 4952/BATCH_SIZE loops
        x = x.to(_device)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(_device)
            obj = y[i][..., 0] == 1  # Identity obj_i
            noobj = y[i][..., 0] == 0  # Identity noobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]  # predicted class == expected class
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > conf_threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class / (tot_class_preds + 1e-16)) * 100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj / (tot_noobj + 1e-16)) * 100:2f}%")
    print(f"Obj accuracy is: {(correct_obj / (tot_obj + 1e-16)) * 100:2f}%")
    model.train()


def mean_average_precision(pred_boxes: List[torch.Tensor],
                           true_boxes: List[torch.Tensor],
                           num_classes: int = 20,
                           iou_threshold: float = 0.5,
                           ):
    """
    calculate mAP
    :param pred_boxes: predicted bboxes [[train_idx, class_pred, obj_prob, x, y, x, y], ...]
    :param true_boxes: expected bboxes  [[train_idx, class_pred, obj_prob, x, y, x, y], ...]
    :param num_classes: pascal_voc=20, coco=80
    :param iou_threshold: iou_threshold to determine TP, FP
    """
    _device = pred_boxes[0].device

    mAPs_per_class = [0] * num_classes
    recalls_per_class = [0] * num_classes
    precisions_per_class = [0] * num_classes

    if all([True if tensor.nelement == 0 else False for tensor in pred_boxes]):  # no predictions
        return mAPs_per_class, recalls_per_class, precisions_per_class

    detections = torch.stack(list(chain.from_iterable(pred_boxes)))
    detections = sorted(detections, key=lambda a: a[2], reverse=True)  # sort by conf (descending)
    detections = torch.stack(detections)

    ground_truths = torch.stack(list(chain.from_iterable(true_boxes)))

    for c in range(num_classes):
        # filter by class
        detections_c = detections[detections[:, 1] == c]
        ground_truths_c = ground_truths[ground_truths[:, 1] == c]

        # {img_idx: number of labels belong to the img}
        _label_counts_per_img = dict(Counter([gt[0].long().item() for gt in ground_truths]))
        for k, v in _label_counts_per_img.items():
            _label_counts_per_img[k] = torch.zeros(v)

        TP = torch.zeros((len(detections_c)), dtype=torch.bool)

        for i, pred in enumerate(detections_c):  # for a single bbox (high conf --> low conf)
            # TODO: boolean indexing every loop... improvement can be made
            labels = ground_truths_c[ground_truths_c[:, 0] == pred[0]]  # compare labels and detections from the same img
            if labels.shape[0] == 0:  # empty label
                continue

            # find best matching GT label from iou_matrix
            iou_matrix = torchvision.ops.box_iou(boxes1=labels[:, 3:], boxes2=pred[3:].unsqueeze(0))
            max_overlap, max_idx = torch.max(iou_matrix, dim=0)  # (label_count, pred_count) --> (pred_count,), pred_count is always 1

            if max_overlap.item() > iou_threshold and _label_counts_per_img[pred[0].item()][max_idx.item()] == 0:
                TP[i] = True
                _label_counts_per_img[pred[0].item()][max_idx.item()] = 1  # this GT label has been used

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(~TP, dim=0)
        _recalls = TP_cumsum / ground_truths_c.shape[0]  # TP_cumsum / total_GT_boxes
        _precisions = TP_cumsum / (TP_cumsum + FP_cumsum)  # TP_cumsum / TP + FP
        _precisions = torch.cat((torch.tensor([1]), _precisions))
        _recalls = torch.cat((torch.tensor([0]), _recalls))

        # DBUGGING =====================================================================================================
        # import matplotlib.pyplot as plt
        # plt.plot(_recalls, _precisions, c="blue")
        # plt.xlabel("recall")
        # plt.ylabel("precision")
        # plt.xlim(0, 1)
        # plt.ylim(0, 1.1)
        # plt.title(f"mAP_{iou_threshold}")
        # plt.grid(color="gray")
        # plt.savefig('foo.png')
        # plt.close()
        # ==============================================================================================================

        mAP = torch.trapz(y=_precisions, x=_recalls)
        mAPs_per_class[c] = mAP.item()

        # TODO: make sure
        precisions_per_class[c] = TP.sum().item() / TP.shape[0]  # TP / TP + FP
        recalls_per_class[c] = TP.sum().item() / ground_truths_c.shape[0]  # TP / TP + FN

    return mAPs_per_class, recalls_per_class, precisions_per_class
