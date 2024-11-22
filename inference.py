import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import argparse

logging.basicConfig(
    filename='Inference_logs.txt',  
    filemode='a',                  
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO           
)

logging.info("Script started.")


def load_model(model_path, num_classes, device):

    model = maskrcnn_resnet50_fpn(pretrained=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Load binary mask
        mask_path = os.path.join(subfolder_path, "mask.npy")
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
        masks_tensor = torch.as_tensor(mask, dtype=torch.uint8)

        target = {
            "boxes": bboxes_2d,
            "masks": masks_tensor,
            "labels": torch.ones((len(bboxes_2d),), dtype=torch.int64)
        }

        if self.transforms:
            image = Image.fromarray(image)
            image, target = self.transforms(image, target)

        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)

        return image_tensor, target



# Function to calculate Average Precision (AP)
def calculate_ap(outputs, targets, iou_threshold=0.5):
    pred_boxes = outputs[0]['boxes'].cpu()
    pred_scores = outputs[0]['scores'].cpu()
    true_boxes = targets['boxes'].cpu()
    ious = torchvision.ops.box_iou(pred_boxes, true_boxes)

    true_positives = torch.zeros(len(pred_boxes))
    assigned = torch.zeros(len(true_boxes))

    for i, pred in enumerate(pred_boxes):

        max_iou, max_idx = ious[i].max(0)
        if max_iou >= iou_threshold and not assigned[max_idx]:
            true_positives[i] = 1  
            assigned[max_idx] = 1  

    sorted_indices = torch.argsort(pred_scores, descending=True)
    sorted_true_positives = true_positives[sorted_indices]

    cumulative_tp = torch.cumsum(sorted_true_positives, dim=0)
    cumulative_fp = torch.cumsum(1 - sorted_true_positives, dim=0)

    precision = cumulative_tp / (cumulative_tp + cumulative_fp)
    recall = cumulative_tp / len(true_boxes)

    ap = torch.trapz(precision, recall).item()
    return ap

def visualize_predictions(original_image, outputs, batch_idx, results_dir="results"):

    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir, f"visualization_batch_{batch_idx}.jpg")

    vis_image = original_image.copy()

    for i, box in enumerate(outputs['boxes']):
        x_min, y_min, x_max, y_max = map(int, box.tolist())

        cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        if 'masks' in outputs:
            mask = outputs['masks'][i, 0].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)  # Binary mask
            color = np.array([255, 0, 0], dtype=np.uint8)  # Red color for mask
            mask_indices = np.where(mask == 1)
            vis_image[mask_indices[0], mask_indices[1]] = (
                0.5 * vis_image[mask_indices[0], mask_indices[1]] + 0.5 * color
            ).astype(np.uint8)

        if 'labels' in outputs:
            label = int(outputs['labels'][i].item())
            cv2.putText(vis_image, f"Label: {label}", (x_min, max(y_min - 10, 0)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)

    cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print(f"Visualization saved to {save_path}")
    logging.info(f"Visualization saved to {save_path}")






def main():
    # Paths
    parser = argparse.ArgumentParser(description="Instance Segmentation Inference Script")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the trained model weights file (e.g., weights.pth)"
    )
    args = parser.parse_args()

    model_path = args.model_path

    num_classes = 2

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = load_model(model_path, num_classes, device)
    print("Model loaded successfully.")
    logging.info("Model loaded successfully.")


    with open('test.txt', 'r') as f:
        all_folders = [line.strip() for line in f.readlines()]

    dataset = Custom3DDataset(subfolders=all_folders)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_batches = len(dataloader)
    random_batches = random.sample(range(total_batches), 5)

    total_ap = {0.25: 0, 0.5: 0, 0.75: 0}
    num_images = len(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader):
        try:
            images = [images[0].to(device)]

            original_image = cv2.imread(os.path.join(all_folders[batch_idx], "rgb.jpg"))

            targets = [{k: v.squeeze(0).to(device) if v.dim() > 1 else v.to(device) for k, v in targets.items()}]

            with torch.no_grad():
                outputs = model(images)

            outputs = [{k: v.cpu() for k, v in output.items()} for output in outputs]
            targets = [{k: v.cpu() for k, v in target.items()} for target in targets]

            if outputs[0]['boxes'].shape[0] == 0:
                print(f"Batch {batch_idx}: No predictions made by the model.")
                logging.info(f"Batch {batch_idx}: No predictions made by the model.")
                continue

            for iou_thresh in [0.25, 0.5, 0.75]:
                try:
                    ap = calculate_ap(outputs, targets[0], iou_threshold=iou_thresh)
                    total_ap[iou_thresh] += ap
                    #print(f"Batch {batch_idx} - AP at IoU={iou_thresh}: {ap:.4f}")
                    logging.info(f"Batch {batch_idx} - AP at IoU={iou_thresh}: {ap:.4f}")
                except Exception as e:
                    print(f"Error calculating AP for IoU={iou_thresh} in batch {batch_idx}: {e}")
                    logging.info(f"Error calculating AP for IoU={iou_thresh} in batch {batch_idx}: {e}")

            if batch_idx in random_batches:
                try:
                    visualize_predictions(original_image=original_image, outputs=outputs[0], batch_idx=batch_idx)
                except Exception as e:
                    print(f"Error visualizing predictions for batch {batch_idx}: {e}")
                    logging.info(f"Error visualizing predictions for batch {batch_idx}: {e}")

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            logging.info(f"Error processing batch {batch_idx}: {e}")

    # Calculate mean AP (mAP)
    if num_images > 0:
        for iou_thresh in total_ap.keys():
            mAP = total_ap[iou_thresh] / num_images
            print(f"Mean Average Precision (mAP) at IoU={iou_thresh}: {mAP:.4f}")
            logging.info(f"Mean Average Precision (mAP) at IoU={iou_thresh}: {mAP:.4f}")
    else:
        print("No valid images were processed.")
        logging.info("No valid images were processed.")



if __name__ == "__main__":
    main()