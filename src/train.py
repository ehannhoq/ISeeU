import iseeu

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import argparse


class WIDERFaceWrapped(datasets.WIDERFace):
    def __init__(self, root, split='train', transform=None, download=False, *kwargs):
        super().__init__(root=root, split=split, transform=None, download=download, *kwargs)

        self.user_transform = transform

        
    def __getitem__(self, index):
        img, target = super().__getitem__(index=index)
        orig_w, orig_h = img.size

        if target is not None and "bbox" in target and len(target["bbox"]) > 0:
            scaled_box = []
            for bbox in target["bbox"]:
                x, y, w, h = map(float, bbox)
                x /= orig_w
                y /= orig_h
                w /= orig_w
                h /= orig_h

                if w < 0.1 or h < 0.1:
                    continue
                    
                scaled_box.append([x, y, w, h])

            target["bbox"] = scaled_box
        else:
            target = {"bbox": []}
        
        if self.user_transform is not None:
            img = self.user_transform(img)

        return img, target


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)

    bboxes = []
    for t in targets:
        if t is None or "bbox" not in t or len(t["bbox"]) == 0:
            bbox = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        else:
            bbox = torch.tensor(t["bbox"][0], dtype=torch.float32)
            
        bboxes.append(bbox)
        
    bboxes = torch.stack(bboxes, 0)
    return images, bboxes


def iou(box1, box2):
    
    x1 = box1[:, 0]
    y1 = box1[:, 1]
    w1 = box1[:, 2]
    h1 = box1[:, 3]

    x2 = box2[:, 0]
    y2 = box2[:, 1]
    w2 = box2[:, 2]
    h2 = box2[:, 3]

    int_top_left_x = torch.max(x1, x2)
    int_top_left_y = torch.max(y1, y2)
    int_bottom_right_x = torch.min(x1 + w1, x2 + w2)
    int_bottom_right_y = torch.min(y1 + h1, y2 + h2)

    int_w = torch.clamp(int_bottom_right_x - int_top_left_x, min=0)
    int_h = torch.clamp(int_bottom_right_y - int_top_left_y, min=0)
    int_area = int_w * int_h

    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - int_area
    return int_area / (union_area + 1e-6)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--save_to_local", type=bool, default=True)
args = parser.parse_args()
if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = args.batch_size

    
    trainset = WIDERFaceWrapped(root="./data", split="train", transform=transform, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    

    valset = WIDERFaceWrapped(root="./data", split="val", transform=transform, download=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                             shuffle=False, num_workers=4,
                                             collate_fn=collate_fn)
    
    print(f"Loading valset and valdata with batch size {args.batch_size}")
    
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    
    
    net = iseeu.ISeeU(iseeu.ISeeUModule, [3, 4, 6, 3]).to(device)

    print("Initialized ISeeU model")

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.95)

    criterion_bbox = nn.SmoothL1Loss()
    criterion_conf = nn.BCELoss()

    wandb.init(project="iseeu-training")

    best_val_iou = 0.0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        net.train()
        running_loss_bbox = 0.0
        running_loss_conf = 0.0
        num_batches = 0

        for i, (images, bboxes) in enumerate(trainloader, 0):
            images, bboxes = images.to(device), bboxes.to(device)
            optimizer.zero_grad()
            pred_bbox, pred_conf = net(images)
            gt_conf = (bboxes.sum(dim=1) > 0).float().unsqueeze(1)
            loss_bbox = criterion_bbox(pred_bbox, bboxes)
            loss_conf = criterion_conf(pred_conf, gt_conf)
            loss = loss_bbox * 15.0 + loss_conf * 1.0
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss_bbox += loss_bbox.item()
            running_loss_conf += loss_conf.item()
            num_batches += 1

        avg_loss_bbox = running_loss_bbox / num_batches if num_batches > 0 else 0.0
        avg_loss_conf = running_loss_conf / num_batches if num_batches > 0 else 0.0
        avg_total_loss = avg_loss_bbox * 15.0 + avg_loss_conf * 1.0
        print(f'[Epoch: {epoch + 1}] Avg Bounding Box Loss: {avg_loss_bbox:.3f} Avg Confidence Loss: {avg_loss_conf:.3f} Avg Total Loss: {avg_total_loss:.3f}')
        wandb.log({
            "epoch": epoch + 1,
            "avg_bbox_loss": avg_loss_bbox,
            "avg_conf_loss": avg_loss_conf,
            "avg_total_loss": avg_total_loss
        })

        net.eval()
        val_iou_scores = []
        with torch.no_grad():
            for images, bboxes in valloader:
                images, bboxes = images.to(device), bboxes.to(device)
                pred_bboxes, pred_conf = net(images)
                batch_iou = iou(pred_bboxes, bboxes)
                val_iou_scores.extend(batch_iou.cpu().numpy())
        mean_val_iou = sum(val_iou_scores) / len(val_iou_scores)
        wandb.log({"mean_val_iou": mean_val_iou})
        print(f"Epoch {epoch + 1} validation mean IoU: {mean_val_iou:.4f}")

        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

            if epochs_no_improve > patience:
                print(f"Early stopping at Epoch {epoch + 1}")

                if args.save_to_local:
                    if not os.path.exists("weights/"):
                        os.mkdir("weights/")
                    torch.save(net.state_dict(), "weights/ISeeU.pth")

                break
    
        net.train()
    
    print('Finished Training')