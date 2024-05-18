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


BATCH_SIZE = 16 
transforms = v2.Compose(
    [v2.ToDtype(torch.float32, scale=True),
     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def test(batch_size, transform=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device('cpu')
    test_folder = "/scratch/users/lu/msc2024_mengze/Extracted_HOFeatures/test_split1/all_data"
    checkpoint_fpath = "/scratch/users/lu/msc2024_mengze/transformer/train_result/eEgoviT_EP30_0516_nogaze.pth"

    testset = PreprocessedImageHODataset(test_folder)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = GEgoviT().to(device)
    checkpoint = torch.load(checkpoint_fpath, map_location=device)
    model.load_state_dict(checkpoint['model'])

    criterion = nn.CrossEntropyLoss().to(device)

    correct_1 = 0
    correct_5 = 0
    total = 0
    total_loss_test = 0.0

    # Initialize counters for class-wise accuracy
    class_correct = [0] * 106
    # print(f"class_correct: {class_correct}")
    class_total = [0] * 106
    model.eval()
    print("Start testing")
    with torch.no_grad():
        # for data in testloader:
        for batch_idx, data in enumerate(tqdm(testloader, desc="Testing")):
            images = data['images'].to(device)
            if transform:
                images = transform(images).float()
            features = data['features'].to(device).float()
            labels = data['label'].to(device).long()
            # print(f"labels: {labels}")

            outputs = model(images, features)
            loss = criterion(outputs, labels)

            _, predicted_1 = torch.max(outputs.data, 1)
            # Get the top 5 predictions
            _, predicted_5 = torch.topk(outputs, 5, dim=1)

            total += labels.size(0)
            # print(f"total: {total}")
            correct_1 += (predicted_1 == labels).sum().item()
            # For top-5 accuracy, check if the true label is among the top-5 predictions
            correct_5 += sum([1 if label in predicted_5[i] else 0 for i, label in enumerate(labels)])
            # print(f"correct-1: {correct_1}")
            # print(f"correct-5: {correct_5}")

            # Update class-wise counters
            for label, prediction in zip(labels, predicted_1):
                class_total[label] += 1
                if label == prediction:
                    class_correct[label] += 1
            
            total_loss_test += loss.item()

        avg_loss = total_loss_test / total
        top1_acc = correct_1 / total
        top5_acc = correct_5 / total
        print(f"Top-1 accuracy: {top1_acc}")
        print(f"Top-5 accuracy: {top5_acc}")
        # Calculate class mean accuracy
        class_accuracies = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
        mean_class_accuracy = sum(class_accuracies) / len(class_accuracies)
        print(f"Mean class accuracy: {mean_class_accuracy}")
        print(f"Class-wise accuracy: {class_accuracies}")
        print("End testing")

        # save the result
        result_folder = "/scratch/users/lu/msc2024_mengze/transformer/test_result"
        checkpoint_fname = os.path.basename(checkpoint_fpath)
        checkpoint_fname = os.path.splitext(checkpoint_fname)[0]
        result_fpath = os.path.join(result_folder, f"{checkpoint_fname}_2.txt")

        with open(result_fpath, "a") as f:
            f.write(f"Test Loss: {avg_loss:.4f}\n")
            f.write(f"Top-1 Accuracy: {top1_acc:.4f}\n")
            f.write(f"Top-5 Accuracy: {top5_acc:.4f}\n")
            f.write(f"Mean Class Accuracy: {mean_class_accuracy:.4f}\n")
            f.write(f"Class-wise accuracy: {class_accuracies}\n")
            f.write(f"class_correct: {class_correct}\n")
            f.write(f"class_total: {class_total}\n")

test(BATCH_SIZE, transform=transforms)