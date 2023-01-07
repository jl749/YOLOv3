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
    :param pred_boxes: predicted bboxes [[class, conf, x, y, x, y], ...]
    :param true_boxes: expected bboxes  [[x, y, x, y, class], ...]
    :param num_classes: pascal_voc=20, coco=80
    :param iou_threshold: iou_threshold to determine TP, FP
    """
    _device = pred_boxes[0].device
    APs_per_class = [0] * num_classes
    recalls_per_class = [0] * num_classes
    precisions_per_class = [0] * num_classes

    def _unique(x, dim=-1):  # return firstly appeared unique elements' indices
        unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([dim]), perm.flip([dim])
        return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)

    def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        """One-dimensional linear interpolation for monotonically increasing sample
        points.

        Returns the one-dimensional piecewise linear interpolant to a function with
        given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

        Args:
            x: the :math:`x`-coordinates at which to evaluate the interpolated
                values.
            xp: the :math:`x`-coordinates of the data points, must be increasing.
            fp: the :math:`y`-coordinates of the data points, same length as `xp`.

        Returns:
            the interpolated values, same size as `x`.
        """
        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])

        indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indicies = torch.clamp(indicies, 0, len(m) - 1)

        return m[indicies] * x + b[indicies]

    stats = []
    for p, t in zip(pred_boxes, true_boxes):
        t = t.to(_device)
        iou_matrix = torchvision.ops.box_iou(boxes1=t[:, 0:4], boxes2=p[:, 2:6])  # (num_labels, num_preds)
        _filtered_indexes = torch.where(
            torch.bitwise_and(iou_matrix > iou_threshold, t[:, -1:] == p[:, 0])
        )
        _TP = torch.zeros(p.shape[0], dtype=torch.bool, device=_device)
        if _filtered_indexes[0].shape[0]:  # if match exists
            matches = torch.cat([torch.stack(_filtered_indexes, 1),
                                 iou_matrix[_filtered_indexes].unsqueeze(-1)], dim=1)
            """ matches: (i, j, iou_corresponding_to_index_ij) """

            # match label -- detection (one2one)
            matches = matches[torch.argsort(matches[:, 2], descending=True)]  # sort by ious
            matches = matches[_unique(matches[:, 1])[1]]  # sort by detection index (if dup keep first observed, drop rest)
            matches = matches[torch.argsort(matches[:, 2], descending=True)]  # sort by ious
            matches = matches[_unique(matches[:, 0])[1]]  # sort by label index  (if dup keep first observed, drop rest)
            _TP[matches[:, 1].long()] = True  # TODO: matches[:, 2] > iou_thresh for multiple iou_thresh
        stats.append((_TP, p[:, 1], p[:, 0], t[:, -1]))
        """
        stats = [(TP, conf, pcls, tcls), ...]
        - TP = (num_detection,)  # detection_ious > iou_threshold
        - conf = (num_detection,)  # detection confidence
        - pcls = (num_detection,)  # predicted classes
        - tcls = (num_targets,)  # target classes
        """
    stats = [torch.cat(x, 0) for x in zip(*stats)]
    TP, _pred_conf, pred_cls, target_cls = stats
    _sorted_index = torch.argsort(_pred_conf, descending=True)
    TP = TP[_sorted_index]
    pred_cls = pred_cls[_sorted_index]

    unique_cls, target_cls_counts = torch.unique(target_cls, return_counts=True)
    for i, c in enumerate(unique_cls):
        _sorted_index_c = pred_cls == c  # boolean index
        total_labels: int = target_cls_counts[i]  # TP + FN
        total_preds: int = torch.sum(_sorted_index_c)  # TP + FP
        _TP_c = TP[_sorted_index_c]

        if total_labels == 0 or total_preds == 0:
            continue
        TP_cumsum = torch.cumsum(TP[_sorted_index_c], dim=0)
        FP_cumsum = torch.cumsum(~TP[_sorted_index_c], dim=0)

        _recalls = TP_cumsum / (total_labels + 1e-16)  # TP_cumsum / total_GT_boxes (TP + FN)
        _precisions = TP_cumsum / (TP_cumsum + FP_cumsum)  # TP_cumsum / TP + FP

        # https://github.com/ultralytics/yolov5/blob/e808f2267d0164edb7bc45588c4fcda68c3dd8cb/utils/metrics.py#L107-L112
        _precisions = torch.cat([torch.tensor([1], device=_device), _precisions, torch.tensor([0], device=_device)])
        _recalls = torch.cat([torch.tensor([0], device=_device), _recalls, torch.tensor([1], device=_device)])
        _precisions = torch.flip(torch.cummax(torch.flip(_precisions, dims=(0,)), dim=0)[0], dims=(0,))

        # DBUGGING =====================================================================================================
        # import matplotlib.pyplot as plt
        # plt.plot(_recalls.cpu().numpy(), _precisions.cpu().numpy(), c="blue")
        # plt.xlabel("recall")
        # plt.ylabel("precision")
        # plt.title(f"mAP_{iou_threshold}")
        # plt.grid(color="gray")
        # plt.show()
        # plt.close()
        # ==============================================================================================================

        # AP = torch.trapz(y=_precisions, x=_recalls)  # without interp or cummax (VOC2010)
        # or
        # x_ = torch.linspace(0, 1, 101).to(_device)  # 101-point interp (COCO), 11-point interp (VOC2007)
        # AP = torch.trapz(y=_interp(x_, _recalls, _precisions), x=x_)  # or torch.mean(_interp(x_, _recalls, _precisions)), interpolation may add noise
        # or
        _i = torch.where(_recalls[1:] != _recalls[:-1])[0]  # points where x-axis (recall) changes
        AP = torch.sum((_recalls[_i + 1] - _recalls[_i]) * _precisions[_i + 1])  # area under curve

        APs_per_class[c.long()] = AP.item()
        recalls_per_class[c.long()] = _TP_c.sum().item() / total_labels  # TP / TP + FN
        precisions_per_class[c.long()] = _TP_c.sum().item() / total_preds  # TP / TP + FP

    return APs_per_class, recalls_per_class, precisions_per_class
