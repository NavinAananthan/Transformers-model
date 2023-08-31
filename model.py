import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F

'''
This function converts words into a corresponding input embeddings

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
This function adds position to those input embeddings when there are two same words present in the sequence to not
change the semantic meanings

Max_position: To create a random zero vector for upto max_positions
dime_embed: Based on the formula to get the dimension of the embedding vector
drop_out: To add some probaility values to the vector values
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



'''
This function is to calculate the attentions when we pass query, key and value
query: Q
key: K
value: V
mask: if we perform masked attention we need to specify if the mask is present or not
'''
def attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None) -> Tensor:
    sqrt_dim_head = query.shape[-1]**0.5
    key_transposed = tensor.t(key)

    scores = torch.matmul(query, key_transposed)
    scores = scores / sqrt_dim_head
    
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    
    # Using dim=-1 is particularly useful when you have varying input tensor shapes
    weight = F.softmax(scores, dim=-1)    
    return torch.matmul(weight, value)







embeddings = Embeddings(5,4)
tensor = torch.tensor([1,2,3,4])
embed = embeddings.forward(tensor)
print(embed)

positionalencoding = PositionalEncoding(10,4)
print(positionalencoding.forward(embed))