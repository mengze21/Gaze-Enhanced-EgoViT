import torch
import os
import datetime
import time
import torch.nn as nn
from torchvision import transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.video import swin3d_b
from EgoviT_swinb import EgoviT_swinb, EgoviT_swinb_v3, EgoviT_swinb_v4
# from myDataset import PreprocessedImageHODataset, PreprocessedImageGazeDataset, PreprocessedOnlyGazeDataset
from myDataset_v2 import PreprocessedImageGazeDataset, PreprocessedOnlyGazeDataset
from torch.utils.tensorboard import SummaryWriter

# 保存检查点函数
def save_checkpoint(state, checkpoint_dir, filename):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

# 加载检查点函数
def load_checkpoint(model, optimizer, scaler, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded successfully from {checkpoint_path} at epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0

# 单个epoch的训练函数
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps):
    model.train()
    total_acc_train = 0
    total_loss_train = 0
    total_batches = 0
    total_samples = 0

    for batch_index, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = batch['images'].to(device).float()
        features = batch['features'].to(device).float()
        labels = batch['label'].to(device).long()

        with autocast():
            outputs = model(images, features)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        if (batch_index + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        acc = (outputs.argmax(1) == labels).sum().item()
        total_acc_train += acc
        total_loss_train += loss.item()
        total_samples += labels.size(0)
        total_batches += 1

    avg_loss = total_loss_train / total_batches
    avg_acc = total_acc_train / total_samples

    return avg_loss, avg_acc

# 训练函数
def train(data_folder, epochs, batch_size, learning_rate, accumulatino_steps=1, transform=None, acc_path=None, checkpoint_dir="checkpoints", resume_from_checkpoint=None, feature_type='G'):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")

    # 加载预训练模型
    swin3d = swin3d_b(weights='KINETICS400_V1')
    model = EgoviT_swinb_v4(swin3d, G=4).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    print(f"learning rate: {learning_rate}")
    # 设置AdamW优化器，所有参数组使用相同的学习率
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 数据加载
    if feature_type == 'HO':
        train_dataset = PreprocessedImageHODataset(data_folder, transform=transform)
        print("Dataset contains head-object")
    if feature_type == 'GHO':
        print("Dataset contains gaze-hand-object")
        train_dataset = PreprocessedImageGazeDataset(data_folder, gaze_folder='/scratch/lu/msc2024_mengze/Extracted_Features_224/train_split1/gaze_v2_32',transform=transform)
    if feature_type == 'G':
        train_dataset = PreprocessedOnlyGazeDataset(data_folder, gaze_folder='/scratch/lu/msc2024_mengze/Extracted_Features_224/train_split1/gaze_v2_32', transform=transform)
        print("Dataset contains only gaze")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 初始化GradScaler用于混合精度训练
    scaler = GradScaler()

    # 初始化TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'tensorboard_logs'))

    start_epoch = 0
    if resume_from_checkpoint:
        start_epoch = load_checkpoint(model, optimizer, scaler, resume_from_checkpoint)

    for epoch in range(start_epoch, epochs):
        avg_loss, avg_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps=accumulatino_steps)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

        # 保存准确率和损失到文件
        with open(acc_path, "a") as f:
            f.write(f"{epoch + 1} {avg_acc:.4f} {avg_loss:.4f}\n")

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch + 1)
        writer.add_scalar('Accuracy/train', avg_acc, epoch + 1)

        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'acc_path': acc_path,
        }
        checkpoint_filename = f'checkpoint_epoch_{epoch + 1}.pth'
        save_checkpoint(checkpoint, checkpoint_dir, checkpoint_filename)

    writer.close()

    return None

# 获取当前日期
today = datetime.datetime.now().strftime("%m%d")
# data features type
feature_type = 'GHO'
acc_path = f"/scratch/lu/msc2024_mengze/transformer/train_result/EgoviT_swinb_v4n_lr1e5_{feature_type}_preweights_newg_{today}.txt"
checkpoint_dir = f"/scratch/lu/msc2024_mengze/transformer/train_result/EgoviT_swinb_v4n_lr1e5_preweights_{feature_type}_newg_{today}"

# 超参数
EPOCHS =25
BATCH_SIZE = 4
LEARNING_RATE = 0.00001
RESUME_FROM_CHECKPOINT = "/scratch/lu/msc2024_mengze/transformer/train_result/EgoviT_swinb_lr1e5_preweights_GHO_newg_0629/checkpoint_epoch_20.pth"  # Update with actual checkpoint path

transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 检查文件是否存在，如果不存在则创建新文件
if not os.path.exists(acc_path):
    with open(acc_path, "w") as f:
        f.write(f"Epochs:{EPOCHS} Learning Rate: {LEARNING_RATE} Batch Size:{BATCH_SIZE} {feature_type} Model:EgoviT_swinb new gaze weights='KINETICS400_V1' G=4\n")
        f.write(f"Epoch avg_acc avg_loss\n")

# 训练 
train(data_folder='/scratch/lu/msc2024_mengze/Extracted_Features_224/train_split1/all', 
      epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, acc_path=acc_path, transform=transform, checkpoint_dir=checkpoint_dir, accumulatino_steps=1, resume_from_checkpoint=None, feature_type=feature_type)
