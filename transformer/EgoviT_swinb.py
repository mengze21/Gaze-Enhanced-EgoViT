import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_b
from einops import rearrange

# Load pre-trained model Video Swin Transformer b
swin3d_b = swin3d_b(weights='KINETICS400_V1')
# print(swin3d_b.features[5:])

class DCTG(nn.Module):
    """
    DCTG (Dynamic Glass Token Generator) module.

    Args:
        hidden_size (int): The size of the hidden state in the LSTM layer.
        embed_dim: The size of the output features.

    Output:
        Class token of gaze-hand-objeckt features.
    """
    def __init__(self, hidden_size=256, embed_dim=128):
        super(DCTG, self).__init__()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, embed_dim)

    def forward(self, x):
        # calculate the mean features of the input
        x_mean = x.mean(dim=2)
        # forward pass the input features through the LSTM layer
        lstm_out, _ = self.lstm(x_mean)
        # forward pass the output of the LSTM layer through the linear layer
        x_cls = lstm_out[:, -1, :]
        x_cls = self.fc(x_cls)

        return x_cls

class ShortTermStage(nn.Module):
    def __init__(self, original_model, num_classes=106):
        super(ShortTermStage, self).__init__()
        self.dctg = DCTG()
        self.patch_embed = original_model.patch_embed
        self.Stage1_3 = original_model.features[0:5]

    def create_class_token_map(self, x, xcls):
        """
        Args:
        x: A tensor of shape (B, D, H, W, C)
        x_cls: A tensor of shape (B, C)

        Returns:
            x: A tensor of shape (B, D+1, H, W, C)
        """

        B, D, H, W, C = x.shape

        xcls = xcls.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        xcls = xcls.expand(-1, -1, H, W, -1)
        x = torch.cat((xcls, x), dim=1)

        return x

    def forward(self, x, features):

        # merge the x and features to create the class token map
        x_cls = self.dctg(features)
        x = self.patch_embed(x) # (B, D, H, W, C)
        x = self.create_class_token_map(x, x_cls) # (B, D+1, H, W, C)

        # feed the class token map to the stage1 to stage3 of the Swin
        x = self.Stage1_3(x)

        return x
    

class PAMD(nn.Module):
    def __init__(self, G=4, embed_dim=128, window_size=(1,7,7)):
        super(PAMD, self).__init__()
        self.G = G
        self.window_size = window_size
        self.num_features = int(embed_dim * 2 ** 3)
        # pool size (4, 1, 1) is only for G=4 window size (1,7,7)
        # self.avg_poll1 = nn.AvgPool3d(kernel_size=(4, 1, 1))
        self.avg_poll2 = nn.AvgPool3d(kernel_size=window_size)
        # self.downsample = PatchMerging(dim=512, norm_layer=nn.LayerNorm)

    def avgpool3d(self, x_cls_group):
            """
            Applies average pooling operation to the input tensor to get the x_cls.

            Args:
                x_cls_group (torch.Tensor): Input tensor of shape (B, G, C, D, H, W)

            Returns:
                torch.Tensor: Output tensor after applying average pooling operation.
                            The shape of the output tensor is (B, G, C, D', H', W'), where
                            D' = D // window_size, H' = H // window_size, and W' = W // window_size.

            """
            window_size = self.window_size
            B, G, C, D, H, W = x_cls_group.shape
            x_cls_group = x_cls_group.view(B * G, C, D, H, W)
            x_cls_group = self.avg_poll2(x_cls_group)
            # reshape to (B, G, C, D', H', W')
            x_cls_group = x_cls_group.view(B, G, C, D // window_size[0], H // window_size[1], W // window_size[2])
            # print(f"x_cls_group shape after avg_pool3d: {x_cls_group.shape}")

            return x_cls_group
    
    def merge_cls(self, x_cls):
        B, G, C, D, H, W = x_cls.shape
        # Flatten spatial dimensions
        x_cls_flat = x_cls.view(B, G, C, -1)  # [batch, group, channels, spatial]

        # Compute L2 norms
        # l2norms = torch.linalg.norm(x_cls_flat, ord=2, dim=2, keepdim=True)
 
        # Compute norms
        norms = torch.norm(x_cls_flat, dim=2, keepdim=True)  # Norm across channels

        norm = x_cls_flat / norms

        dot_products = torch.einsum('bgki,bgkj->bgij', norm, norm)

        # Mask diagonal to exclude self-comparison
        diag_mask = torch.eye(G, dtype=torch.bool).unsqueeze(0).repeat(B, 1, 1)
        dot_products[diag_mask] = 0

        # score per group
        alpha = dot_products.sum(dim=-1)

        # total score of all groups
        alpha_gs = alpha.sum(dim=1)

        # Apply softmax to normalize scores
        alpha_normalized = F.softmax(alpha_gs, dim=1)

        # Weighted sum of class tokens along group axis
        xcls_weighted = torch.einsum('bgcs,bg->bcs', x_cls_flat, alpha_normalized)

        # Reshape to original shape (B,C, D, H, W)
        x_cls_weighted = xcls_weighted.view(B, C, D, H, W)

        return x_cls_weighted
    
    def upsampling_xcls(self, x_cls):
        B, C, Di, Hi, Wi = x_cls.shape

        upsampled_size1 = (1, Hi * 7, Wi * 7)
        x_cls = F.interpolate(x_cls, size=upsampled_size1, mode='nearest')

        # # slice the x_cls to (B, G, C, D, H, W)
        # x_cls = x_cls.narrow(3, 0, 8)
        # x_cls = x_cls.narrow(4, 0, 10)

        return x_cls
    
    def forward(self, x):
        """Input x is x_cls_s1, shape is (B, G, D, H, W, C)"""
        # separate the x and x_cls, the first D is the x_cls
        
        x_clses = x[:, :, 0, :, :, :] # (B, G, H, W, C)
        x_normal = x[:, :, 1:, :, :, :] # (B, G, D, H, W, C)

        # Merging the normal class tokens
        # average pooling x_normal along temporal dimension D for each group
        # (B, G, D, H, W, C) -> (B, G, H, W, C)
        x_normal = x_normal.mean(dim=2, keepdim=False) 

        # reshape x_clses to (B, G, D, H, W, C)
        x_clses = x_clses.unsqueeze(2)

        # reshape x_clses to (B, G, C, D, H, W)
        x_clses = rearrange(x_clses, 'B G D H W C -> B G C D H W')

        # get the x_cls from every windows
        x_clses = self.avgpool3d(x_clses) # B G C D' H' W'

        # merge the x_clses
        x_cls_weighted = self.merge_cls(x_clses)
        
        # upsampling the x_cls
        x_cls_weighted = self.upsampling_xcls(x_cls_weighted)
        # reshape x_cls_weighted to (B, D, H, W, C)
        x_cls_weighted = rearrange(x_cls_weighted, 'B C D H W -> B D H W C')

        # merge the x_cls with x_group
        x_xcl_s2 = torch.cat((x_cls_weighted, x_normal), dim=1)

        return x_xcl_s2


class LongTermStage(nn.Module):
    def __init__(self, original_model):
        super(LongTermStage, self).__init__()
        self.Stage4 = original_model.features[5:]

    def forward(self, x):
        x = self.Stage4(x)

        return x
    
class Head(nn.Module):
    def __init__(self, original_model, num_classes=106, dropout=0.5):
        super(Head, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.avgpool = original_model.avgpool
        self.fc = nn.Linear(original_model.head.in_features, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x
    
class EgoviT_swinb(nn.Module):
    def __init__(self, original_model, G=4, num_classes=106):
        super(EgoviT_swinb, self).__init__()
        self.G = G
        # self.features = nn.Sequential(*list(original_model.children())[2:-1])
        self.STStage = ShortTermStage(original_model)
        self.PADM = PAMD(G=G)
        self.LTStage = LongTermStage(original_model)
        self.norm = original_model.norm
        # self.avgpool = original_model.avgpool
        self.head = Head(original_model, num_classes=num_classes)

    def forward(self, x, features):
        # split the inputs into G groups at the temporal dimension
        x_split = torch.split(x, x.size(2) // self.G, dim=2)
        features_split = torch.split(features, features.size(1) // self.G, dim=1)

        x_xcl_list = []

        for x, features in zip(x_split, features_split):
            x_xcl = self.STStage(x, features)
            x_xcl_list.append(x_xcl)
        x_xcl_s1 = torch.stack(x_xcl_list, dim=1) # (B, G, D, H, W, C)

        del x_split, features_split, x_xcl_list, x_xcl, features
        gc.collect()
        # torch.cuda.empty_cache()
        x = self.PADM(x_xcl_s1)
        # torch.cuda.empty_cache()
        x = self.LTStage(x)

        # torch.cuda.empty_cache()
        x = x[:, 0, :, :, :]
        x = x.unsqueeze(1)
        x = self.norm(x)
        x = rearrange(x, 'B D H W C -> B C D H W')

        x = self.head(x)

        return x
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EgoviT_swinb = EgoviT_swinb(swin3d_b).to(device)
# print(EgoviT_swinb)

x = torch.randn(1, 3, 32, 224, 224).to(device)
features = torch.randn(1, 32, 3, 2048).to(device)
outputs = EgoviT_swinb(x, features)
# print(outputs.shape)