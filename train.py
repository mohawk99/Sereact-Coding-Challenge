import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision
import torch.optim as optim
from torchvision.transforms import Normalize
from PIL import Image
import cv2
import random
import logging
import argparse

#Logging
logging.basicConfig(
    filename='training_logs.txt',  
    filemode='a',                 
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO             
)

logging.info("Script started.")

class Custom3DDataset(Dataset):
    def __init__(self, subfolders, transforms=None):
        self.subfolders = subfolders
        self.transforms = transforms

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, idx):
        subfolder_path = self.subfolders[idx]

        # Load RGB image
        img_path = os.path.join(subfolder_path, "rgb.jpg")
        image = cv2.imread(img_path)
        if image is None:
            logging.error(f"Failed to load image at {img_path}.")
            raise FileNotFoundError(f"Image not found at {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load binary mask
        mask_path = os.path.join(subfolder_path, "mask.npy")
        if not os.path.exists(mask_path):
            logging.error(f"Mask file missing at {mask_path}.")
            raise FileNotFoundError(f"Mask file not found at {mask_path}")

        mask = np.load(mask_path)

        # Load 3D bounding boxes
        bbox_path = os.path.join(subfolder_path, "bbox3d.npy")
        bbox3d = np.load(bbox_path)

        # Load point cloud
        pc_path = os.path.join(subfolder_path, "pc.npy")
        point_cloud = np.load(pc_path)

        H, W = point_cloud.shape[1], point_cloud.shape[2]
        assert image.shape[:2] == (H, W), "Image and point cloud dimensions must match"

        point_cloud_flat = point_cloud.reshape(3, -1).T
        bboxes_2d = []

        for bbox in bbox3d:
            x_min, y_min, z_min = bbox.min(axis=0)
            x_max, y_max, z_max = bbox.max(axis=0)

            inside_box = (
                (point_cloud_flat[:, 0] >= x_min) & (point_cloud_flat[:, 0] <= x_max) &
                (point_cloud_flat[:, 1] >= y_min) & (point_cloud_flat[:, 1] <= y_max) &
                (point_cloud_flat[:, 2] >= z_min) & (point_cloud_flat[:, 2] <= z_max)
            )

            indices_inside = np.where(inside_box)[0]
            y_coords, x_coords = np.divmod(indices_inside, W)

            if len(x_coords) > 0 and len(y_coords) > 0:
                x_min_2d, y_min_2d = x_coords.min(), y_coords.min()
                x_max_2d, y_max_2d = x_coords.max(), y_coords.max()
                bboxes_2d.append([x_min_2d, y_min_2d, x_max_2d, y_max_2d])

        bboxes_2d = torch.as_tensor(bboxes_2d, dtype=torch.float32)
        target = {
            "boxes": bboxes_2d,
            "masks": torch.as_tensor(mask, dtype=torch.uint8),
            "labels": torch.ones((len(bboxes_2d),), dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }
        for box in target["boxes"]:
            if not (box[2] > box[0] and box[3] > box[1]):
                logging.warning(f"Invalid bounding box: {box} in subfolder {subfolder_path}")

        if self.transforms:
            image = Image.fromarray(image)
            image, target = self.transforms(image, target)

        return torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1), target



def collate_fn(batch):
    return tuple(zip(*batch))


# Training Function
def train_model(model, dataloader, optimizer, device, num_epochs=10):
    model.train()  
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for epoch in range(num_epochs):
        epoch_loss = 0
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}...")
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [normalize(img / 255.0).to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if all(len(target["boxes"]) == 0 for target in targets):
                logging.warning(f"Batch {batch_idx}: Skipping due to empty targets.")
                continue

            try:
                model.train()
                loss_dict = model(images, targets)

                if isinstance(loss_dict, dict) and len(loss_dict) > 0:
                    losses = sum(loss for loss in loss_dict.values())
                else:
                    logging.error(f"Batch {batch_idx}: Received unexpected loss_dict: {loss_dict}")
                    continue

                if torch.isnan(losses):
                    logging.error(f"Detected NaN loss in epoch {epoch}, batch {batch_idx}")
                    logging.debug(f"Loss breakdown: {loss_dict}")
                    continue

                optimizer.zero_grad()
                losses.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()

                epoch_loss += losses.item()

            except RuntimeError as e:
                logging.error(f"RuntimeError in Batch {batch_idx}: {e}")
                continue

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")


def handle_dataset_splits(root_dir):
    if not (os.path.exists('train.txt') and os.path.exists('test.txt')):
        all_folders = [
            os.path.join(root_dir, folder)
            for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder))
        ]

        random.shuffle(all_folders)
        train_size = int(0.7 * len(all_folders))
        train_folders = all_folders[:train_size]
        test_folders = all_folders[train_size:]

        with open('train.txt', 'w') as f:
            for folder in train_folders:
                f.write(f"{folder}\n")


        with open('test.txt', 'w') as f:
            for folder in test_folders:
                f.write(f"{folder}\n")
        print("Dataset splits created and saved to files.")
        logging.info("Dataset splits created and saved to files.")
    else:
        print("Dataset split files found. Loading splits...")
        logging.info("Dataset split files found. Loading splits...")


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Mask R-CNN Training Script")
    parser.add_argument(
        "--root_dir", 
        type=str, 
        required=True, 
        help="Root directory for dataset handling and splits."
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="weights.pth", 
        help="Path to save/load the model checkpoint."
    )
    args = parser.parse_args()

    root_dir = args.root_dir
    checkpoint_path = args.checkpoint_path

    torch.cuda.empty_cache()

    handle_dataset_splits(root_dir)

    with open('train.txt', 'r') as f:
        train_folders = [line.strip() for line in f.readlines()]
    with open('test.txt', 'r') as f:
        test_folders = [line.strip() for line in f.readlines()]

    train_dataset = Custom3DDataset(train_folders)
    test_dataset = Custom3DDataset(test_folders)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Model Setup
    num_classes = 2  
    model = maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)

    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}...")
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        logging.info("Checkpoint loaded successfully.")
        print("Checkpoint loaded successfully.")

    logging.info("Starting training...")
    print("Starting training...")
    train_model(model, train_loader, optimizer, device, num_epochs=10)

    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Model saved to {checkpoint_path}")
    print(f"Model saved to {checkpoint_path}")
