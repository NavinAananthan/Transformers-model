import torch
import torch.nn as nn
import math

'''
dim_embed: refers to the dimension of the model
vocab_size: refers to the no. of words in the vocabulary
'''

class Embeddings(nn.Module):

    def __init__(self, vocab_size: int, dim_embed: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,dim_embed)
        self.sqrt_dim_embed = math.sqrt(dim_embed)

    def forward(self, x):
        x = self.embedding(x)
        x = x * self.sqrt_dim_embed
        return x
    

embeddings = Embeddings(5,5)

tensor = torch.tensor([1,2,3,4])
print(embeddings.forward(tensor))


