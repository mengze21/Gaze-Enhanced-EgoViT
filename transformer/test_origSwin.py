import torch
import torchvision.transforms as transforms
import timm
import cv2
import time
import os
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from myDataset_224 import MyDataset
from torchvision.models.video import swin3d_b


def train_model(clips_dir, labels_file, epochs, learning_rate, batch_size, acc_path, pretrained=True, transform=None):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the Swin Transformer model from timm
    model = swin3d_b(weights='KINETICS400_V1')

    # change the head of the model, to match the number of classes in the dataset
    model.head = nn.Linear(model.head.in_features, 106)

    # model_state = model.state_dict()
    # print("weight:", model_state['head.weight'].shape)
    # print("bias:", model_state['head.bias'].shape)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Load the dataset
    test_dataset = MyDataset(clips_dir, labels_file, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        total_samples = 0

        for batch in tqdm(test_loader):
            images = batch['images'].to(device)
            # reshape the images to (batch_size, 3, 32, 224, 224)
            images = images.view(images.size(0), 3, 32, 224, 224)
            labels = batch['label'].to(device).long()

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = (outputs.argmax(1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += loss.item()
            total_samples += labels.size(0)

        avg_loss = total_loss_train / total_samples
        avg_acc = total_acc_train / total_samples

        with open(acc_path, "a") as f:
            f.write(f"{epoch + 1} {avg_acc:.4f} {avg_loss:.4f}\n")

        print(f"Epoch {epoch + 1}, Loss: {avg_loss}, Accuracy: {avg_acc}")

    # save checkpoint
    model_save_dir = (f'/scratch/users/lu/msc2024_mengze/transformer/train_result/')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = (model_save_dir + f'eEgoviT_EP{epochs}__swinb.pth')
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

def evaluate(clips_dir, labels_file, checkpoint_fpath, batch_size, transform=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the Swin Transformer model from timm
    model = swin3d_b(weights='KINETICS400_V1')
    # model = swin3d_b()
    # change the head of the model, to match the number of classes in the dataset
    model.head = nn.Linear(model.head.in_features, 106)
    # print(model.head)
    # checkpoint = torch.load(checkpoint_fpath, map_location=device)
    # model.load_state_dict(checkpoint['model'])

    model_state = model.state_dict()
    print("weight:", model_state['head.weight'].shape)
    print(model_state['head.weight'])
    print("bias:", model_state['head.bias'].shape)
    print("patch embedding:", model_state['patch_embed.proj.weight'])
    model.to(device)
    model.eval()

    # with open('orig_swin.txt', 'w') as f:
    #     f.write(str(model))

    criterion = nn.CrossEntropyLoss().to(device)

    # Load the dataset
    test_dataset = MyDataset(clips_dir, labels_file, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    correct_1 = 0
    correct_5 = 0
    total_loss_test = 0
    total_samples = 0

    # Initialize counters for class-wise accuracy
    class_correct = [0] * 106
    # print(f"class_correct: {class_correct}")
    class_total = [0] * 106

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['images'].to(device)
            # reshape the images to (batch_size, 3, 32, 224, 224)
   
            images = images.view(images.size(0), 3, 32, 224, 224)
            # print(images.shape)
            labels = batch['label'].to(device).long()

            outputs = model(images)
            # print(f"outputs shape is {outputs.shape}")
            loss = criterion(outputs, labels)

            # print(f"Loss: {loss.item()}")
            # print(f"labels.size: {labels.size(0)}")

            total_loss_test += loss.item()
            _, predicted_1 = torch.max(outputs, 1)
            # Get the top 5 predictions
            _, predicted_5 = torch.topk(outputs, 5, dim=1)

            total_samples += labels.size(0)

            correct_1 += (predicted_1 == labels).sum().item()
            # For top-5 accuracy, check if the true label is among the top-5 predictions
            correct_5 += sum([1 if label in predicted_5[i] else 0 for i, label in enumerate(labels)])

            # Update class-wise counters
            for label, prediction in zip(labels, predicted_1):
                class_total[label] += 1
                if label == prediction:
                    class_correct[label] += 1

        avg_loss_test = total_loss_test / total_samples
        top1_acc_test = correct_1 / total_samples
        top5_acc_test = correct_5 / total_samples

        # Calculate class mean accuracy
        class_accuracies = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
        mean_class_accuracy = sum(class_accuracies) / len(class_accuracies)
        print(f"Test Loss: {avg_loss_test}, Test Accuracy: {top1_acc_test}, Top-5 Accuracy: {top5_acc_test}, Mean Class Accuracy: {mean_class_accuracy}")

         # save the result
        result_folder = "/scratch/users/lu/msc2024_mengze/transformer/test_result"
        checkpoint_fname = os.path.basename(checkpoint_fpath)
        checkpoint_fname = os.path.splitext(checkpoint_fname)[0]
        result_fpath = os.path.join(result_folder, f"{checkpoint_fname}2.txt")

        with open(result_fpath, "a") as f:
            f.write(f"Tested model: {checkpoint_fname}\n")
            f.write(f"Test Loss: {avg_loss_test:.4f}\n")
            f.write(f"Top-1 Accuracy: {top1_acc_test:.4f}\n")
            f.write(f"Top-5 Accuracy: {top5_acc_test:.4f}\n")
            f.write(f"Mean Class Accuracy: {mean_class_accuracy:.4f}\n")
            f.write(f"Class-wise accuracy: {class_accuracies}\n")
            f.write(f"class_correct: {class_correct}\n")
            f.write(f"class_total: {class_total}\n")

if __name__ == "__main__":

    EPOCHS = 20
    LEARNING_RATE = 0.00001
    BATCH_SIZE = 1
    pretrained = True
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    TRAINING = False

    if TRAINING:
        print("Training the model")
        # train the model
        clips_dir = "/scratch/users/lu/msc2024_mengze/Frames3/train_split1"
        labels_file = "/scratch/users/lu/msc2024_mengze/dataset/train_split12.txt"
        acc_fpath = "/scratch/users/lu/msc2024_mengze/accuracy_history_swinb.txt"
        # evaluate(clips_dir, labels_file, batch_size, transform=transform)

        # Check if the file exists, if not, create a new one
        if not os.path.exists(acc_fpath):
                with open(acc_fpath, "w") as f:
                    f.write(f"Epochs:{EPOCHS} Learning Rate:{LEARNING_RATE} Batch Size:{BATCH_SIZE} Dataset:train_split1 with gaze\n")
                    f.write(f"Epoch avg_acc avg_loss\n")
        train_model(clips_dir, labels_file, EPOCHS, LEARNING_RATE, BATCH_SIZE, acc_path=acc_fpath, transform=transform)

    else:
        print("Evaluating the model")
        # evaluate the model
        checkpoint_fpath = "/scratch/users/lu/msc2024_mengze/transformer/train_result/eEgoviT_EP20__swinb.pth"
        clips_dir_test = "/scratch/users/lu/msc2024_mengze/Frames3/test_split1"
        labels_file_test = "/scratch/users/lu/msc2024_mengze/dataset/test_split12.txt"
        evaluate(clips_dir_test, labels_file_test, checkpoint_fpath, BATCH_SIZE, transform=transform)
