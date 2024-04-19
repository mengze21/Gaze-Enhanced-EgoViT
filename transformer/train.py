import numpy as np
import torch
import cv2
import time
import torch.nn as nn
from newT import GEgoviT
# from CustomDataset import ImageGazeDataset
from myDataset2 import ImageGazeDataset
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


def model_train(image_folder, gaze_folder, label_folder, epochs, learning_rate, batch_size,transform):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load model, loss function, and optimizer
    model = GEgoviT().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Load batch data
    loading_start = time.time()
    print("Loading data...")
    train_dataset = ImageGazeDataset(image_folder, gaze_folder, label_folder, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loading_end = time.time()
    print(f"Data loaded in {(loading_end - loading_start) / 60 } minutes")

    # Fine tune the loop
    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0.0

        for batch in tqdm(train_loader):

            images = batch['images'].to(device)
            gazes = batch['features'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images, gazes)
            # print("$$$$$$$$")
            # print(f'outputs is {outputs}')
            # print(f'labels is {labels}')
            # print("$$$$$$$$")
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        print(f"Epoch {epoch + 1}, Loss: {total_loss_train / len(train_loader)}, total_acc_train: {total_acc_train / len(train_loader)}")

    return model


# Hyperparameters
EPOCHS = 5
LEARNING_RATE = 0.0001
BATCH_SIZE = 1


# Train the model
image_folder = '/scratch/users/lu/msc2024_mengze/Frames3/test_split1'
gaze_folder = '/scratch/users/lu/msc2024_mengze/Extracted_HOFeatures/test_split1/combined_features'
label_folder = '/scratch/users/lu/msc2024_mengze/dataset/test_split12.txt'

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor()
])

train_model = model_train(image_folder, gaze_folder, label_folder, EPOCHS, LEARNING_RATE, BATCH_SIZE, transform=transform)