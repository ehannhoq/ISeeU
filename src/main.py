import iseeu

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)

    bboxes = []
    for t in targets:
        if "bbox" in t and len(t["bbox"] > 0):
            bbox = torch.as_tensor(t["bbox"][0], dtype=torch.float32)
        else:
            bbox = torch.zeros(4, dtype=torch.float32)
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


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = 64

    trainset = datasets.WIDERFace(root='./data', transform=transform, 
                                  split="train", download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4,
                                              collate_fn=collate_fn)
    
    testset = datasets.WIDERFace(root='./data', transform=transform, 
                                 download=True, split="test")
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4,
                                             collate_fn=collate_fn)
    
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    
    
    net = iseeu.ISeeU(iseeu.ISeeUModule, [3, 4, 6, 3]).to(device)

    if os.path.exists("weights/ISeeU.pth"):
        net.load_state_dict(torch.load("weights/ISeeU.pth"))

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

            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                wandb.log({"epoch_avg_loss": running_loss / 2000})
                running_loss = 0.0
        
            wandb.log({"loss": running_loss})

        net.eval()
        iou_scores = []
        with torch.no_grad():
            for i, (images, bboxes) in enumerate(testloader, 0):
                images, bboxes = images.to(device), bboxes.to(device)
                pred_bbox, pred_conf = net(images)
                batch_iou = iou(pred_bbox, bboxes)
                iou_scores.extend(batch_iou.cpu().numpy())
        mean_iou = sum(iou_scores) / len(iou_scores)
        wandb.log({"test_iou": mean_iou})

        print(f"Epoch {epoch + 1}, Loss {running_loss / len(trainloader):.4f}, IoU: {mean_iou:.4f}")
    
    print('Finished Training')
    torch.save(net.state_dict(), "weights/ISeeU.pth")
    wandb.save("ISeeU.pth")


    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    wandb.log({"accuracy": correct/total})