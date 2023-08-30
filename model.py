import torch
import torch.nn as nn
import math
from torch import Tensor

'''
dim_embed: refers to the dimension of the model
vocab_size: refers to the no. of words in the vocabulary
'''

class Embeddings(nn.Module):

    def __init__(self, vocab_size: int, dim_embed: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,dim_embed)
        self.sqrt_dim_embed = math.sqrt(dim_embed)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x.long())
        x = x * self.sqrt_dim_embed
        return x
    

'''
Max_position:
dime_embed:
drop_out:
'''
class PositionalEncoding(nn.Module):

    def __init__(self, max_positions: int, dim_embed: int, drop_out: float = 0.1) -> None:
        super().__init__()

        assert dim_embed % 2 == 0

        position = torch.arange(max_positions).unsqueeze(1)
        dim_pair = torch.arange(0, dim_embed, 2)
        div_term = torch.exp(dim_pair * (-math.log(10000.0) / dim_embed))

        pe = torch.zeros(max_positions, dim_embed)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x: Tensor) -> Tensor:
        max_sequence_length = x.size(1)
        x = x + self.pe[:, :max_sequence_length]
        x = self.dropout(x)
        return x



embeddings = Embeddings(6,4)

tensor = torch.tensor([1,1,4,4])
vector = embeddings.forward(tensor)
print(vector)

positionalencoding = PositionalEncoding(10,4)
print(positionalencoding.forward(vector))


