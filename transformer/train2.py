# Use new dataset and Mixed Precision Training

import torch
import os
import datetime
import time
import torch.nn as nn
from newT4 import GEgoviT
# from myDataset3 import ImageGazeDataset
from myDataset import ImageHODataset, PreprocessedImageGazeDataset, PreprocessedImageHODataset
from torchvision.transforms import v2
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter


def model_train(data_folder, epochs, learning_rate, batch_size, transform=None, acc_path="transformer/train_result/accuracy_history.txt"):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device('cpu')
    # Load model, loss function, and optimizer
    model = GEgoviT().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Load batch data
    # train_dataset = ImageGazeDataset(image_folder, gaze_folder, label_folder, transform=transform)
    train_dataset = PreprocessedImageGazeDataset(data_folder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    today = datetime.datetime.now().strftime("%m%d")
    acc_path = f"/scratch/users/lu/msc2024_mengze/transformer/train_result/accuracy_history_{today}.txt"

    # Initialize TensorBoard
    log_dir = f"/scratch/users/lu/msc2024_mengze/transformer/train_result/log/experiment_gaze_{datetime.datetime.now().strftime('%m%d-%H')}"
    writer = SummaryWriter(log_dir)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # Fine tune the loop
    for epoch in range(epochs):
        # epoch_start = time.time()
        total_acc_train = 0
        total_loss_train = 0
        total_samples = 0

        # scaler = GradScaler()
        for batch in tqdm(train_loader):
            images = batch['images'].to(device)
            if transform:
                images = transform(images).float()
            gazes = batch['features'].to(device).float()
            labels = batch['label'].to(device).long()

            # outputs = model(images, gazes)
            # loss = criterion(outputs, labels)

            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # Mixed Precision Training
            with autocast():
                outputs = model(images, gazes)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()

            acc = (outputs.argmax(1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += loss.item()
            total_samples += labels.size(0)


        avg_loss = total_loss_train / total_samples
        avg_acc = total_acc_train / total_samples

        # Save accuracy to a text file
        # Get today's date in a formatted string (YYYYMMDD)
        # today = datetime.datetime.now().strftime("%m%d")
        # Construct the new file path with the date
        # acc_path = f"/scratch/users/lu/msc2024_mengze/transformer/train_result/accuracy_history_{today}.txt"
        with open(acc_path, "a") as f:
            f.write(f"{epoch + 1} {avg_acc:.4f} {avg_loss:.4f}\n")

        # Log the training loss and accuracy to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_acc, epoch)

        # epoch_end = time.time()
        # epoch_time = (epoch_end - epoch_start) / 60
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}, Accuracy: {avg_acc}")
    
    # save checkpoint
    model_save_dir = (f'/scratch/users/lu/msc2024_mengze/transformer/train_result/')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = (model_save_dir + f'eEgoviT_EP{epochs}_{today}_gaze.pth')
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'loss': avg_loss,
        'accuracy': avg_acc,
        'acc_history': acc_path,
    }
    print(f"checkpoint keys: {checkpoint.keys()}")
    torch.save(checkpoint, model_save_path)

    # Close the TensorBoard writer
    writer.close()

    return model, optimizer


# Hyperparameters
EPOCHS = 15
LEARNING_RATE = 0.00001
BATCH_SIZE = 2

# save the hyperparameters
today = datetime.datetime.now().strftime("%m%d")

# Check if the file exists, if not, create a new one
if not os.path.exists(f"/scratch/users/lu/msc2024_mengze/transformer/train_result/accuracy_history_{today}.txt"):
        with open(f"/scratch/users/lu/msc2024_mengze/transformer/train_result/accuracy_history_{today}.txt", "w") as f:
            f.write(f"Epochs:{EPOCHS} Learning Rate:{LEARNING_RATE} Batch Size:{BATCH_SIZE} Dataset:train_split1 with gaze\n")
            f.write(f"Epoch avg_acc avg_loss\n")

# Train the model
# image_folder = '/scratch/users/lu/msc2024_mengze/Frames3/train_split1'
dataset_folder = '/scratch/users/lu/msc2024_mengze/Extracted_HOFeatures/train_split1/all_data'
# label_folder = '/scratch/users/lu/msc2024_mengze/dataset/train_split12.txt'

# 图像预处理
transforms = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trained_model, optimizer = model_train(dataset_folder, EPOCHS, LEARNING_RATE, BATCH_SIZE, transform=transforms)

