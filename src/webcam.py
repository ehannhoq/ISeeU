import iseeu

from torchvision import transforms
import torch
import os
import cv2
import time

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cam = cv2.VideoCapture(0)
    device = torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available()
                        else "cpu")
    
    net = iseeu.ISeeU(iseeu.ISeeUModule, [3, 4, 6, 3]).to(device)
    
    path = "weights/ISeeU.pth"
    if not os.path.exists(path):
        raise Exception(f"Weights not found at {path}")

    net.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device('cpu')))
    net.eval()
    
    if not cam.isOpened():
        exit()

    while True:
        ret, frame = cam.read()

        if not ret:
            print("Error: Could not read frame.")
            break
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        orig_h, orig_w = frame.shape[:2]
        
        img = transform(frame).unsqueeze(0).to(device)

        bbox, conf = net(img)
        bbox = bbox[0].detach().cpu().numpy()

        x, y, w, h = bbox
        w *= orig_w
        h *= orig_h

        x1 = x * orig_w
        y1 = y * orig_h
        x2 = x1 + w
        y2 = y1 + h

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cv2.imshow('Webcam', frame)
        print(f"Bounding Box: {[x1, y1, x2, y2]}, Confidence: {conf[0][0]}", end="\r")
        
        time.sleep(0.1)

    print()



        
