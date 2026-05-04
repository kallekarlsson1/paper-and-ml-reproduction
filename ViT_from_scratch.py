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


"""
Positional encoding and Class token

When we attend to each token we will encode information about all other tokens and itself in the token. That token is now biased by its own information. we need a class level token
that can represent the whole image. This is a learnable token and it is the one that is used for example classification. 

Since the transformer does not handle the data sequentially (handles it in parallel), we need to encode some type of information that says what position the image patch has in the sequence.
The transformer operates on the values of dimensions embedding in each patch. So we cant really have it "on the side". We then add the positional encoding to each embedding in each dimension for each 
patch. So now there is positional information and "pixel" information. We work in relations, so the transformer is like a relational machine. The value of the pixels in each patch does not carry
any meaningful information on its own, it only has that by relation to the other patches.
EX
image 4x4 pos0 pos1
          pos2 pos3
                col0 col1 col2 col3
pe matrix pos0    0,    0,  0,  0,
          pos1    1,    0.5,  1,  0.2
          pos2    2,    2,  2,  2,
          pos3    3,    3,  3,  3,  

patch embedding  (pos1) [0.5, 0.5, 0.5, 0.5]
pe                      [1,1,1,1]
patch + pe = [1.5,1.0,1.5,0.7]
each dimension of each patch now has information about its postion. 
Each single values does not tell anything about the position. There is not global map where you can see where it should be placed. The transformer compares to differnet positions
values and calculates how similar they are. Doing this with the whole image it will learn the ordering of the patches. So from the beginning it has no hard coded spatial knowledge,
everything is learned. This is more flexible than CNN but requires more data.

"""




class PositionalEnconding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model))

        #positional encoding
        pe = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if  i % 2 == 0 :
                    pe[pos][i] = np.sin(pos/(10000**(i/d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000**(i-1/d_model)))
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):

        #give every imagne in batch a class token
        cls_in_all_image = self.cls_token.expand(x.size()[0], -1, -1)

        #add cls token to beginning of every patch sequence for each image 
        x = torch.cat((cls_in_all_image,x), dim=1)

        x = x + self.pe

        return x

            