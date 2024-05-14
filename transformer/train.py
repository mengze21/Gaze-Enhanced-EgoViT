import torch
import os
import datetime
import time
import torch.nn as nn
from newT4 import GEgoviT
from myDataset3 import ImageGazeDataset
from torchvision.transforms import v2
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm



def model_train(image_folder, gaze_folder, label_folder, epochs, learning_rate, batch_size, transform=None, acc_path="transformer/train_result/accuracy_history.txt"):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load model, loss function, and optimizer
    model = GEgoviT().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Load batch data
    train_dataset = ImageGazeDataset(image_folder, gaze_folder, label_folder, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    today = datetime.datetime.now().strftime("%m%d")
    acc_path = f"/scratch/users/lu/msc2024_mengze/transformer/train_result/accuracy_history_{today}.txt"

    # Fine tune the loop
    for epoch in range(epochs):
        epoch_start = time.time()
        total_acc_train = 0
        total_loss_train = 0.0

        for batch in tqdm(train_loader):
            images = batch['images'].to(device)
            if transform:
                images = transform(images)
            gazes = batch['features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, gazes)
            loss = criterion(outputs, labels)

            acc = (outputs.argmax(1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss_train / len(train_loader)
        avg_acc = total_acc_train / len(train_loader)

        # Save accuracy to a text file
        # Get today's date in a formatted string (YYYYMMDD)
        # today = datetime.datetime.now().strftime("%m%d")
        # Construct the new file path with the date
        # acc_path = f"/scratch/users/lu/msc2024_mengze/transformer/train_result/accuracy_history_{today}.txt"
        with open(acc_path, "a") as f:
            f.write(f"{epoch + 1} {avg_acc:.4f} {avg_loss:.4f}\n")

        epoch_end = time.time()
        epoch_time = (epoch_end - epoch_start) / 60
        print(f"Epoch {epoch + 1}, Loss: {total_loss_train / len(train_loader)}, total_acc_train: {total_acc_train / len(train_loader)}, time: {epoch_time:.2f} min")
    
    # save checkpoint
    model_save_dir = (f'/scratch/users/lu/msc2024_mengze/transformer/train_result/')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = (model_save_dir + f'eEgoviT_EP{epochs}_{today}.pth')
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

    return model, optimizer


# Hyperparameters
EPOCHS = 10
LEARNING_RATE = 0.00001
BATCH_SIZE = 2

# save the hyperparameters
today = datetime.datetime.now().strftime("%m%d")

# Check if the file exists, if not, create a new one
if not os.path.exists(f"/scratch/users/lu/msc2024_mengze/transformer/train_result/accuracy_history_{today}.txt"):
        with open(f"/scratch/users/lu/msc2024_mengze/transformer/train_result/accuracy_history_{today}.txt", "w") as f:
            f.write(f"Epochs:{EPOCHS} Learning Rate:{LEARNING_RATE} Batch Size:{BATCH_SIZE}, Dataset:train_split1\n")
            f.write(f"Epoch avg_acc avg_loss\n")

# Train the model
image_folder = '/scratch/users/lu/msc2024_mengze/Frames3/train_split1'
gaze_folder = '/scratch/users/lu/msc2024_mengze/Extracted_HOFeatures/train_split1/combined_features'
label_folder = '/scratch/users/lu/msc2024_mengze/dataset/train_split12.txt'

# 图像预处理
transforms = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trained_model, optimizer = model_train(image_folder, gaze_folder, label_folder, EPOCHS, LEARNING_RATE, BATCH_SIZE, transform=transforms)

