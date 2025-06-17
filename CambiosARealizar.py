import torch
import os
from PIL import Image
import torchvision.transforms as transforms

class EgoExoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_size = image_size
        self.samples = []
        
        # Populate samples list with paths
        for sample_id in os.listdir(root_dir):
            sample_path = os.path.join(root_dir, sample_id)
            if os.path.isdir(sample_path):
                self.samples.append(sample_path)
        
        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # For masks
        ])
        self.rgb_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For RGB
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_path = self.samples[idx]
        
        # Load all required images
        inputs = {
            'exo_frame': self.rgb_transform(Image.open(os.path.join(base_path, 'exo_frame.png')).convert('RGB')),
            'exo_mask': self.transform(Image.open(os.path.join(base_path, 'exo_mask.png')).convert('L')),
            'exo_object_crop': self.rgb_transform(Image.open(os.path.join(base_path, 'exo_object_crop.png')).convert('RGB')),
            'cropped_exo_mask': self.transform(Image.open(os.path.join(base_path, 'cropped_exo_mask.png')).convert('L')),
            'ego_mask': self.transform(Image.open(os.path.join(base_path, 'ego_mask.png')).convert('L')),
            'cropped_ego_mask': self.transform(Image.open(os.path.join(base_path, 'cropped_ego_mask.png')).convert('L')),
        }
        target = self.rgb_transform(Image.open(os.path.join(base_path, 'ego_object_crop.png')).convert('RGB'))
        
        # Concatenate inputs along channel dimension
        conditioning = torch.cat([
            inputs['exo_frame'],
            inputs['exo_mask'],
            inputs['exo_object_crop'],
            inputs['cropped_exo_mask'],
            inputs['ego_mask'],
            inputs['cropped_ego_mask']
        ], dim=0)
        
        return conditioning, target
    


########## Modificar DiT Model Architecture ##########

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ConditioningEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # First ResNet for exo images (exo_frame + exo_mask)
        self.resnet1 = resnet50(pretrained=False)
        self.resnet1.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet1.fc = nn.Identity()
        
        # Second ResNet for exo crop + masks
        self.resnet2 = resnet50(pretrained=False)
        self.resnet2.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet2.fc = nn.Identity()
        
        # Projection to hidden size
        self.proj = nn.Linear(4096, hidden_size)  # 2048 * 2

    def forward(self, x):
        # Split into two groups
        group1 = x[:, :4]   # exo_frame (3) + exo_mask (1)
        group2 = x[:, 4:]   # exo_object_crop (3) + masks (2)
        
        feat1 = self.resnet1(group1)
        feat2 = self.resnet2(group2)
        features = torch.cat([feat1, feat2], dim=1)
        return self.proj(features)

class DiT_pix(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=14,  # 4 (latent) + 10 (conditioning)
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_encoder = ConditioningEncoder(hidden_size)
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def forward(self, x, t, conditioning):
        # x: noisy latent (4 channels)
        # conditioning: 10 channels of input images
        
        # Combine with conditioning
        x = torch.cat([x, conditioning], dim=1)
        
        x = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        c_emb = self.cond_encoder(conditioning)
        c = t_emb + c_emb
        
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

########### Add new model  ##########
def DiT_S_2(**kwargs):
    return DiT_pix(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT_pix(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT_pix(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

# Add to model registry
DiT_models = {
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


########## Modify Training Loop ##########

# In main() function:
transform = None  # Now handled in dataset
dataset = EgoExoDataset(args.data_path, image_size=args.image_size)  # Custom dataset

# In training loop:
for conditioning, target in loader:  # New data format
    conditioning = conditioning.to(device)
    target = target.to(device)
    
    with torch.no_grad():
        # Encode target to latent space
        latent = vae.encode(target).latent_dist.sample().mul_(0.18215)
    
    t = torch.randint(0, diffusion.num_timesteps, (latent.shape[0],), device=device)
    model_kwargs = {"conditioning": conditioning}  # Pass conditioning
    
    loss_dict = diffusion.training_losses(model, latent, t, model_kwargs)
    loss = loss_dict["loss"].mean()
    # ... rest of training loop unchanged ...