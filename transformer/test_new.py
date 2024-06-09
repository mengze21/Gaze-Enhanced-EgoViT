import torch
import os
import datetime
import time
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
from torchvision.models.video import swin3d_b
from EgoviT_swinb import EgoviT_swinb
from myDataset import PreprocessedImageHODataset, PreprocessedImageGazeDataset, PreprocessedOnlyImageDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# from sklearn.metrics import accuracy_score


def test_one_epoch(model, test_loader, criterion, device):
    model.eval()
    total_acc_test = 0
    total_loss_test = 0
    total_samples = 0
    top5_acc = 0
    per_class_correct = np.zeros(106)
    per_class_total = np.zeros(106)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            images = batch['images'].to(device).float()
            features = batch['features'].to(device).float()
            labels = batch['label'].to(device).long()

            with autocast():
                outputs = model(images, features)
                loss = criterion(outputs, labels)

            acc = (outputs.argmax(1) == labels).sum().item()
            total_acc_test += acc
            total_loss_test += loss.item()
            total_samples += labels.size(0)

            # Calculate top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_acc += (top5_pred == labels.unsqueeze(1)).sum().item()

            # Calculate per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = outputs[i].argmax().item()
                per_class_total[label] += 1
                if label == pred:
                    per_class_correct[label] += 1

    avg_loss = total_loss_test / total_samples
    avg_acc = total_acc_test / total_samples
    top5_acc = top5_acc / total_samples
    per_class_acc = per_class_correct / per_class_total
    mean_class_acc = np.mean(per_class_acc)

    return avg_loss, avg_acc, top5_acc, per_class_acc, mean_class_acc


def test(data_folder, model_path, batch_size, transform=None, acc_path="transformer/test_result/accuracy_history.txt"):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load pre-trained model Video Swin Transformer b
    swin3d = swin3d_b(weights='KINETICS400_V1')
    model = EgoviT_swinb(swin3d, G=4).to(device)
    model.load_state_dict(torch.load(model_path)['model'])

    criterion = nn.CrossEntropyLoss().to(device)

    # Load batch data
    test_dataset = PreprocessedImageHODataset(data_folder, transform=transform)
    # test_dataset = PreprocessedImageGazeDataset(data_folder, transform=transform)
    # test_dataset = PreprocessedOnlyImageDataset(data_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(model_path), 'tensorboard_logs'))

    start_time = time.time()

    avg_loss, avg_acc, top5_acc, per_class_acc, mean_class_acc = test_one_epoch(model, test_loader, criterion, device)

    epoch_time = time.time() - start_time
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, Top-5 Accuracy: {top5_acc:.4f}, Mean Class Accuracy: {mean_class_acc:.4f}, Time: {epoch_time:.2f}s")

    # Save accuracy and loss to file
    with open(acc_path, "a") as f:
        f.write(f"Test Loss: {avg_loss:.4f} Accuracy: {avg_acc:.4f} Top-5 Accuracy: {top5_acc:.4f}\n")
        f.write(f"Class-wise accuracy: {per_class_acc}\n")
        f.write(f"Mean Class Accuracy: {mean_class_acc:.4f}\n")

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/test', avg_loss, 1)
    writer.add_scalar('Accuracy/test', avg_acc, 1)
    writer.add_scalar('Top5Accuracy/test', top5_acc, 1)
    writer.add_scalar('MeanClassAccuracy/test', mean_class_acc, 1)

    writer.close()

    return avg_loss, avg_acc, top5_acc, mean_class_acc


today = datetime.datetime.now().strftime("%m%d")
acc_path = f"/scratch/users/lu/msc2024_mengze/transformer/test_result/test_result_EgoviT_orignal_{today}.txt"
model_path = f"/scratch/users/lu/msc2024_mengze/transformer/train_result/eEgoviT_swinb_EP15_0528_HO.pth"

# Hyperparameters
BATCH_SIZE = 1

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Check if the file exists, if not, create a new one
if not os.path.exists(acc_path):
    with open(acc_path, "w") as f:
        f.write(f"Batch Size:{BATCH_SIZE} Dataset:test_split1 with gaze Model:EgoviT_HO\n")
        

# 
test(data_folder='/scratch/users/lu/msc2024_mengze/Extracted_HOFeatures/test_split1/all_data', 
     model_path=model_path, batch_size=BATCH_SIZE, acc_path=acc_path, transform=transform)