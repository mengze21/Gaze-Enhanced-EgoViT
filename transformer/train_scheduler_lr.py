# update from train_new.py

import torch
import os
import datetime
import time
import torch.nn as nn
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.video import swin3d_b
from EgoviT_swinb import EgoviT_swinb
from myDataset import PreprocessedImageHODataset, PreprocessedImageGazeDataset, PreprocessedOnlyGazeDataset
from torch.utils.tensorboard import SummaryWriter

# 自定义线性预热调度器
class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]


# 调度器包装器，用于在预热之后应用余弦退火
class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_scheduler, cosine_scheduler):
        self.warmup_scheduler = warmup_scheduler
        self.cosine_scheduler = cosine_scheduler
        super(WarmupCosineScheduler, self).__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_scheduler.warmup_epochs:
            return self.warmup_scheduler.get_lr()
        else:
            self.cosine_scheduler.last_epoch = self.last_epoch - self.warmup_scheduler.warmup_epochs
            return self.cosine_scheduler.get_lr()
        

# 保存检查点函数
def save_checkpoint(state, checkpoint_dir, filename):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

# 单个epoch的训练函数
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

# 训练函数
def train(data_folder, epochs, learning_rate, batch_size, transform=None, acc_path="transformer/train_result/accuracy_history.txt", checkpoint_dir="checkpoints"):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:5" if use_cuda else "cpu")

    # 加载预训练模型
    swin3d = swin3d_b(weights='KINETICS400_V1')
    model = EgoviT_swinb(swin3d, G=4).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # 为主干和头部分别设置参数组
    STStage_params = list(model.STStage.parameters())
    PADM_params = list(model.PADM.parameters())
    LTStage_params = list(model.LTStage.parameters())
    norm_params = list(model.norm.parameters())
    head_params = list(model.head.parameters())

    optimizer = AdamW([
        {'params': STStage_params, 'lr': 3e-5 * 0.1},
        {'params': PADM_params, 'lr': 3e-5},
        {'params': LTStage_params, 'lr': 3e-5 * 0.1},
        {'params': norm_params, 'lr': 3e-4},
        {'params': head_params, 'lr': 3e-4}
    ], weight_decay=1e-2)

    # 数据加载
    train_dataset = PreprocessedImageHODataset(data_folder, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # 初始化GradScaler用于混合精度训练
    scaler = GradScaler()

    # 初始化TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'tensorboard_logs'))

    # 创建调度器
    warmup_epochs = 1.5
    total_epochs = epochs
    warmup_scheduler = LinearWarmupScheduler(optimizer, warmup_epochs, total_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    scheduler = WarmupCosineScheduler(optimizer, warmup_scheduler, cosine_scheduler)

    for epoch in range(epochs):
        start_time = time.time()

        avg_loss, avg_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, Time: {epoch_time:.2f}s")

        # 保存准确率和损失到文件
        with open(acc_path, "a") as f:
            f.write(f"{epoch + 1} {avg_acc:.4f} {avg_loss:.4f}\n")

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch + 1)
        writer.add_scalar('Accuracy/train', avg_acc, epoch + 1)

        # 更新学习率
        scheduler.step()

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
acc_path = f"/scratch/lu/msc2024_mengze/transformer/train_result/scheduler_lr_onlyHO_{today}.txt"
checkpoint_dir = f"/scratch/lu/msc2024_mengze/transformer/train_result/EgoviT_swinb_scheduler_lr_orig_onlyHO_{today}"

# 超参数
EPOCHS = 15
LEARNING_RATE = 0.00001
BATCH_SIZE = 4

transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 检查文件是否存在，如果不存在则创建新文件
if not os.path.exists(acc_path):
    with open(acc_path, "w") as f:
        f.write(f"Epochs:{EPOCHS} Learning Rate:{LEARNING_RATE} Batch Size:{BATCH_SIZE} only HO Model:EgoviT_swinb\n")
        f.write(f"Epoch avg_acc avg_loss\n")

# 训练
train(data_folder='/scratch/lu/msc2024_mengze/Extracted_Features_224/train_split1/all', 
      epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, acc_path=acc_path, transform=transform, checkpoint_dir=checkpoint_dir)