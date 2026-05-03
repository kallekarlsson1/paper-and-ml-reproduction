import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import  DataLoader
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
import numpy as np

"""
We need patch embedding, positional encoding, attention, multi head attention, transformer encoder, and final vit model,

"""

"""
patch embedding. Only the projection part actually do anything with the numbers. Transpose is noly to keep the transformers input convention. Flatten unroll the 
2D grid to a sequence of patches.
ex. (32, 3, 224, 224)
we have that the stride and the kernel must be the same size to ensure no overlap. if we have convolution (768,3,16,16), we get 224/16=14 grid of patches. (32,3,224,224) -> (32,768,14,14) 
This means that we have a 14*14 grid of patches were each of them holds a 768 dimensional vector which is the embedding if that patch. When we flatten it we get a 14*14=196 
1D sequence of patches (32,768,196), (whole rows, so (1,0), (1,1) . (2,0)).  

The core idea is to reduce the number of entries attention has to handle. Since self-attentino is O(n^2), it would be impossible to make it efficient. If we have 3*224*224= 150528 values.
After the patch embedding we would have 196*768=150528 values. Even tough we make this reorganization we lose no information. The thign we lose is the granualrity, Attention can not
handle what happens in each patch since it only sees the patch as "one entity not a entity builty by entities"
"""

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()

        self.d_model = d_model  #dimension of the model
        self.img_size = img_size  #size of the image 
        self.patch_size = patch_size #size of each patch 
        self.n_channels = n_channels #number of channels, rgb 3, ir 1
        self.linear_projection = nn.Conv2d(self.n_channels, self.d_model, kernel=patch_size, stride=self.img_size)

    def forward(self, x):
        x = self.linear_projection(x) # (B,C,H,W) -> (B, d_model, p_col, p_row)
        x = x.flatten(2) # (B, d_model, p_col, p_row) -> (B, d_model, P)
        x = x.transpose(1,2) #(B, d_model, P) -> (B, P, d_model)
    
        return x

