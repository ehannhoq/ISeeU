import iseeu

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import argparse


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)

    bboxes = []
    for t in targets:
        if t is None or "bbox" not in t or len(t["bbox"]) == 0:
            bbox = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        else:
            bbox = torch.as_tensor(t["bbox"][0], dtype=torch.float32)
            
        bboxes.append(bbox)
        
    bboxes = torch.stack(bboxes, 0)
    return images, bboxes


def iou(box1, box2):
    b1_x1 = box1[:, 0] - box1[:, 2] / 2
    b1_y1 = box1[:, 1] - box1[:, 3] / 2
    b1_x2 = box1[:, 0] + box1[:, 2] / 2
    b1_y2 = box1[:, 1] + box1[:, 3] / 2

    b2_x1 = box2[:, 0] - box2[:, 2] / 2
    b2_y1 = box2[:, 1] - box2[:, 3] / 2
    b2_x2 = box2[:, 0] + box2[:, 2] / 2
    b2_y2 = box2[:, 1] + box2[:, 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union = b1_area + b2_area - inter_area
    return inter_area / (union + 1e-6)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--save_to_local", type=bool, default=True)
parser.add_argument("--save_to_wandb", type=bool, default=True)
args = parser.parse_args()
if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = args.batch_size

    trainset = datasets.WIDERFace(root='./data', transform=transform, 
                                  split="train", download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4,
                                              collate_fn=collate_fn)
    
    print(f"Loading trainset and traindata with batch size {args.batch_size}")
    
    testset = datasets.WIDERFace(root='./data', transform=transform, 
                                 download=True, split="test")
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4,
                                             collate_fn=collate_fn)
    
    print(f"Loading testset and testdata with batch size {args.batch_size}")

    
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    
    
    net = iseeu.ISeeU(iseeu.ISeeUModule, [3, 4, 6, 3]).to(device)

    print("Initialized ISeeU model")

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion_bbox = nn.SmoothL1Loss()
    criterion_conf = nn.BCELoss()

    wandb.init(project="iseeu-training")

    for epoch in range(10):
        net.train()
        running_loss = 0.0

        for i, (images, bboxes) in enumerate(trainloader, 0):

            images, bboxes = images.to(device), bboxes.to(device)

            optimizer.zero_grad()
            pred_bbox, pred_conf = net(images)

            gt_conf = (bboxes.sum(dim=1) > 0).float().unsqueeze(1)

            loss_bbox = criterion_bbox(pred_bbox, bboxes)
            loss_conf = criterion_conf(pred_conf, gt_conf)
            loss = loss_bbox + loss_conf

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 49:
                print('[Epoch: %d, Batch: %5d] Loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                wandb.log({"epoch_avg_loss": running_loss / 50})
                running_loss = 0.0

        print(f"Epoch {epoch + 1}, Loss {running_loss / len(trainloader):.4f}")
    
    print('Finished Training')

    if args.save_to_local:
        if not os.path.exists("weights/"):
            os.mkdir("weights/")
        torch.save(net.state_dict(), "weights/ISeeU.pth")

    if args.save_to_wandb:
        wandb.save("ISeeU.pth")


    net.eval()
    iou_scores = []

    with torch.no_grad():
        for images, bboxes in testloader:
            if images is None:
                continue

            images, bboxes = images.to(device), bboxes.to(device)
            pred_bbox, pred_conf = net(images)

            batch_iou = iou(pred_bbox, bboxes)
            iou_scores.extend(batch_iou.cpu().numpy())

    mean_iou = sum(iou_scores) / len(iou_scores)
    wandb.log({"test_mean_iou": mean_iou})