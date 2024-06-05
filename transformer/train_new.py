# -------------------------------------------
# New training script for EgoviT_swinb model
# frame is 224x224
# -------------------------------------------

import torch
import os
import datetime
import time
import torch.nn as nn
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.video import swin3d_b
from EgoviT_swinb import EgoviT_swinb
from myDataset import PreprocessedImageHODataset, PreprocessedImageGazeDataset, PreprocessedOnlyGazeDataset
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(state, checkpoint_dir, filename):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_acc_train = 0
    total_loss_train = 0
    total_batches = 0
    total_samples = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        images = batch['images'].to(device).float()
        features = batch['features'].to(device).float()
        labels = batch['label'].to(device).long()

        optimizer.zero_grad()

        with autocast():
            outputs = model(images, features)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = (outputs.argmax(1) == labels).sum().item()
        total_acc_train += acc
        total_loss_train += loss.item()
        total_samples += labels.size(0)
        total_batches += 1

    avg_loss = total_loss_train / total_batches
    avg_acc = total_acc_train / total_samples

    return avg_loss, avg_acc


def train(data_folder, epochs, learning_rate, batch_size, transform=None, acc_path="transformer/train_result/accuracy_history.txt", checkpoint_dir="checkpoints"):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load pre-trained model Video Swin Transformer b
    swin3d = swin3d_b(weights='KINETICS400_V1')
    model = EgoviT_swinb(swin3d, G=8).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Load batch data
    # train_dataset = PreprocessedImageHODataset(data_folder, transform=transform)
    # train_dataset = PreprocessedImageGazeDataset(data_folder, transform=transform)
    train_dataset =  PreprocessedOnlyGazeDataset(data_folder, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'tensorboard_logs'))

    for epoch in range(epochs):
        start_time = time.time()

        avg_loss, avg_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, Time: {epoch_time:.2f}s")

        # Save accuracy and loss to file
        with open(acc_path, "a") as f:
            f.write(f"{epoch + 1} {avg_acc:.4f} {avg_loss:.4f}\n")

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch + 1)
        writer.add_scalar('Accuracy/train', avg_acc, epoch + 1)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc_path': acc_path,
        }
        checkpoint_filename = f'checkpoint_epoch_{epoch + 1}.pth'
        save_checkpoint(checkpoint, checkpoint_dir, checkpoint_filename)

    # # Final model saving
    # final_model_path = os.path.join(checkpoint_dir, 'EgoviT_swinb_final.pth')
    # save_checkpoint(checkpoint, checkpoint_dir, 'EgoviT_swinb_final.pth')

    writer.close()

    return None


today = datetime.datetime.now().strftime("%m%d")
acc_path = f"/scratch/users/lu/msc2024_mengze/transformer/train_result/accuracy_history_onlygaze_{today}.txt"
checkpoint_dir = f"/scratch/users/lu/msc2024_mengze/transformer/train_result/checkpoints_onlygaze_{today}_HO"

# Hyperparameters
EPOCHS = 15
LEARNING_RATE = 0.00001
BATCH_SIZE = 1

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Check if the file exists, if not, create a new one
if not os.path.exists(acc_path):
    with open(acc_path, "w") as f:
        f.write(f"Epochs:{EPOCHS} Learning Rate:{LEARNING_RATE} Batch Size:{BATCH_SIZE} Dataset:train_split1 only gaze features Model:EgoviT_swinb\n")
        f.write(f"Epoch avg_acc avg_loss\n")

# Training
train(data_folder='/scratch/users/lu/msc2024_mengze/Extracted_Features_224/train_split1/only_gaze_features', 
      epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, acc_path=acc_path, transform=transform, checkpoint_dir=checkpoint_dir)