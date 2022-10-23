"""
Main file for training Yolo model on Pascal VOC and COCO data
"""
import torch
import torch.optim as optim

import argparse
from pathlib import Path
from tqdm import tqdm
from model import YOLOv3
from yolov3.utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    YoloLoss
)
import config
# import warnings
# warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='YOLOv3 training')
parser.add_argument('--img_size', type=int, nargs='+', default=(416, 416), help='train image spatial dim (H, W)')
parser.add_argument('--data_dir', type=str, default='data', help='dataset directory')
parser.add_argument('--epochs', type=int, default=300, help='num epochs for training')
parser.add_argument('--chkpt', type=str, help='fine-tuning chkpt')
parser.add_argument('--num_classes', type=int, default=20, help='number of classes, VOC=20, COCO=80')

# optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
parser.add_argument('--step_size', type=int, default=100, help='gamma update step size for LRScheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma update for LRScheduler')

# prediction hyperparameters
parser.add_argument('--nms_threshold', type=float, default=0.45, help='nms threshold for prediction')
parser.add_argument('--conf_threshold', type=float, default=0.6, help='confidence threshold for prediction')
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
    model.train()
    losses = []
    pbar = tqdm(train_loader, leave=True)

    # y = [(3, 13, 13, 6), (3, 26, 26, 6), (3, 52, 52, 6)]
    for batch_idx, (img_batch, labels) in enumerate(pbar):
        img_batch = img_batch.to(device)
        y0, y1, y2 = (
            labels[0].to(device),  # (3, 13, 13, 6)
            labels[1].to(device),  # (3, 26, 26, 6)
            labels[2].to(device),  # (3, 52, 52, 6)
        )

        with torch.cuda.amp.autocast():  # float16 forward pass
            pred = model(img_batch)

            criterion.set_anchor(0)
            big_scale_loss = criterion(pred[0], y0)

            criterion.set_anchor(1)
            mid_scale_loss = criterion(pred[1], y1)

            criterion.set_anchor(2)
            small_scale_loss = criterion(pred[2], y2)

            loss = big_scale_loss + mid_scale_loss + small_scale_loss

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()  # loss.backward()
        scaler.step(optimizer)  # optimizer.step()
        scaler.update()

        # update tqdm progress bar
        mean_loss = sum(losses) / len(losses)
        pbar.set_description(f'epoch: {epoch} || loss={mean_loss:.3f}')


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
    scaled_anchors = (  # scale anchor boxes 0~1 --> 0~S
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)  # TODO: image_size from argparse
    ).to(device)  # (3, 3, 2)
    criterion = YoloLoss(scaled_anchors)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=DATA_DIR.joinpath("train.csv"), test_csv_path=DATA_DIR.joinpath("test.csv")
    )

    if args.chkpt:  # if provided
        load_checkpoint(
            args.chkpt, model, optimizer, args.lr
        )

    for epoch in range(args.epochs):
        # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, criterion, scaler, epoch=epoch)
        save_checkpoint(model, optimizer, filename="checkpoint.pth.tar")

        # print(f"Current epoch {epoch}")
        # print("On Train Eval loader:")
        # print("On Train loader:")
        # check_class_accuracy(model, train_loader, threshold=args.conf_threshold)
        scheduler.step()

        # EVALUATION every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            eval_func(test_loader, model, scaled_anchors=scaled_anchors)

    # plot_couple_examples(model, train_eval_loader,
    #                      nms_threshold=args.nms_threshold,
    #                      threshold=args.conf_threshold,
    #                      anchors=config.ANCHORS)


if __name__ == "__main__":
    main()
