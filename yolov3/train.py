"""
Main file for training Yolo model on Pascal VOC and COCO data
"""
import argparse
from pathlib import Path
# import warnings
# warnings.filterwarnings("ignore")

from tqdm import tqdm
import torch
import torch.optim as optim

from model import YOLOv3
from loss import YoloLoss
from yolov3.utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    save_loss_plot,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
)
from yolov3.config import ANCHORS, LAMBDA_CLASS, LAMBDA_NOOBJ, LAMBDA_OBJ, LAMBDA_BOX

parser = argparse.ArgumentParser(description='YOLOv3 training')
parser.add_argument('--img_size', type=int, default=416, help='train image size')
parser.add_argument('--data_dir', type=str, default='data', help='dataset directory')
parser.add_argument('--epochs', type=int, default=300, help='num epochs for training')
parser.add_argument('--chkpt', type=str, help='fine-tuning chkpt')
parser.add_argument('--num_classes', type=int, default=20, help='number of classes, VOC=20, COCO=80')

# dataloader
parser.add_argument('--batch_size', type=int, default=8, help='dataloader batch size')
parser.add_argument('--num_workers', type=int, default=0, help='dataloader num workers')
parser.add_argument('--pin_memeory', action='store_true')  # train on GPU, dataloader --> GPU

# optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
parser.add_argument('--step_size', type=int, default=100, help='gamma update step size for LRScheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma update for LRScheduler')

# prediction hyperparameters
parser.add_argument('--nms_threshold', type=float, default=0.45, help='nms threshold for prediction')
parser.add_argument('--conf_threshold', type=float, default=0.2, help='confidence threshold for prediction')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='iou threshold for evaluation (TP, FP boundary)')

parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
torch.backends.cudnn.benchmark = True
DATA_DIR = Path(args.data_dir)

def train_fn(train_loader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             criterion: torch.nn.Module,
             scaler, epoch):
    _device = next(model.parameters()).device

    model.train()
    loss_tracker = torch.tensor([0, 0, 0, 0], dtype=torch.float32, device=_device, requires_grad=False)
    pbar = tqdm(train_loader, leave=True)

    # y = [(3, 13, 13, 6), (3, 26, 26, 6), (3, 52, 52, 6)], given input img_size=416
    for batch_idx, (img_batch, labels, _) in enumerate(pbar):
        img_batch = img_batch.to(device)
        labels = (
            labels[0].to(device),  # (3, 13, 13, 6)
            labels[1].to(device),  # (3, 26, 26, 6)
            labels[2].to(device),  # (3, 52, 52, 6)
        )

        loss = 0.
        with torch.cuda.amp.autocast():  # float16 forward pass
            predictions = model(img_batch)  # [(N, 3, 13, 13, 5+num_cls), (N, 3, 26, 26, 5+num_cls), (N, 3, 52, 52, 5+num_cls)]
            for i, pred in enumerate(predictions):  # for each scale (big, mid, small)
                criterion.set_anchor(i)
                _box_loss, _object_loss, _no_object_loss, _class_loss = criterion(pred, labels[i])

                _box_loss *= LAMBDA_BOX
                _object_loss *= LAMBDA_OBJ
                _no_object_loss *= LAMBDA_NOOBJ
                _class_loss *= LAMBDA_CLASS
                loss = _box_loss + _object_loss + _no_object_loss + _class_loss

                with torch.no_grad():
                    loss_tracker[0] += _box_loss
                    loss_tracker[1] += _object_loss
                    loss_tracker[2] += _no_object_loss
                    loss_tracker[3] += _class_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # loss.backward()
        scaler.step(optimizer)  # optimizer.step()
        scaler.update()

        # update tqdm progress bar
        with torch.no_grad():
            mean_losses = loss_tracker / (batch_idx + 1)
        pbar.set_description(f'epoch: {epoch} || box_loss={mean_losses[0]:.3f} || obj_loss={mean_losses[1]:.3f} || no_obj_loss={mean_losses[2]:.3f} || class_loss={mean_losses[3]:.3f}')

    return mean_losses.detach().cpu().tolist()


def eval_func(test_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              scaled_anchors):
    # check_class_accuracy(model, test_loader, threshold=args.conf_threshold)  # TODO: double check
    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        nms_threshold=args.nms_threshold,
        anchors=scaled_anchors,
        conf_threshold=args.conf_threshold,
    )

    mAPs_per_class, recalls_per_class, precisions_per_class = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=args.iou_threshold,
        num_classes=args.num_classes,
    )
    print(f"mAP: {sum(mAPs_per_class) / len(mAPs_per_class)}")
    print(f"recall: {sum(recalls_per_class) / len(recalls_per_class)}")
    print(f"precision: {sum(precisions_per_class) / len(precisions_per_class)}")


def main():
    model = YOLOv3(num_classes=args.num_classes).to(device)

    # CNN based vision models often use SGD over Adam
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    _S = [args.img_size // 32, args.img_size // 16, args.img_size // 8]
    scaled_anchors = (  # scale anchor boxes 0~1 --> 0~S
            torch.tensor(ANCHORS)
            * torch.tensor(_S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)  # (3, 3, 2)
    criterion = YoloLoss(scaled_anchors)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=DATA_DIR.joinpath("train.csv"),
        test_csv_path=DATA_DIR.joinpath("test.csv"),
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memeory
    )

    if args.chkpt:  # if provided
        load_checkpoint(
            args.chkpt, model, optimizer, args.lr
        )

    losses = []
    for epoch in range(args.epochs):
        # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)

        mean_losses = train_fn(train_loader, model, optimizer, criterion, scaler, epoch=epoch)
        losses.append(mean_losses)
        save_loss_plot(losses)
        save_checkpoint(model, optimizer, filename="checkpoint.pth.tar")

        # check_class_accuracy(model, train_loader, threshold=args.conf_threshold)

        scheduler.step()
        # EVALUATION every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            eval_func(test_loader, model, scaled_anchors=scaled_anchors)

    # plot_couple_examples(model, train_eval_loader,
    #                      nms_threshold=args.nms_threshold,
    #                      threshold=args.conf_threshold,
    #                      anchors=ANCHORS


if __name__ == "__main__":
    main()
