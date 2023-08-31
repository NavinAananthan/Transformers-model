# Transformers Model
This repository is based on a Text Translation model built on Transformers architecture

* Input Embedings
   - 
   - Here we pass the input text sequence which is converted to word Ids by mapping it to the vocabulary.
   - Then we convert the word Ids to tensors then we pass it into Embeddings class.
   - Before that we should initialize the class by passig in two parameters Vocab_size the amount of text or input tensors we are going to give input and dim_embed the dimension of the output should we get.
  
      ```
      embeddings = Embeddings(5,5)
      tensor = torch.tensor([1,2,3,4])
      print(embeddings.forward(tensor))

      output:
      tensor
      ([[-2.4605,  0.6936,  1.1314,  1.8564,  0.8002],
        [-1.0799,  2.8328, -1.5969,  0.3128, -2.2947],
        [-0.1975, -1.3198, -1.9824,  4.3277, -3.7367],
        [-2.2290, -4.1239, -1.4516, -0.6692,  1.2785]], 
        grad_fn=<MulBackward0>)
      ```
    - If you give input tensor with same number you would get embedings for the both same number with same vector embeddings that is why we go for positional encoding.


* Positional Encoding
  - 
  - When two same words are present in a sequence we perform positional encoding to find the position of each word based on even and odd positions
  - It uses sin curve for even index and cos curve for odd index
  - Based on that we add that values to the position which generates unique values for each word and values in it so no two words have the same context vector.
  - so we get input of embedding dimension only when it is divisible by 2 and create a ranom tensor with max_length and add replace the zero value with position value.
  - then we register that with non-learnable parameters and apply dropout to it.

      ```
      positionalencoding = PositionalEncoding(10,4)
      print(positionalencoding.pe)
      print(positionalencoding.pe[:,:4])

      output:

      tensor([[[ 0.0000,  1.0000,  0.0000,  1.0000],  
         [ 0.8415,  0.5403,  0.0100,  0.9999],  
         [ 0.9093, -0.4161,  0.0200,  0.9998],  
         [ 0.1411, -0.9900,  0.0300,  0.9996],  
         [-0.7568, -0.6536,  0.0400,  0.9992],  
         [-0.9589,  0.2837,  0.0500,  0.9988],  
         [-0.2794,  0.9602,  0.0600,  0.9982],  
         [ 0.6570,  0.7539,  0.0699,  0.9976],  
         [ 0.9894, -0.1455,  0.0799,  0.9968],  
         [ 0.4121, -0.9111,  0.0899,  0.9960]]])

      tensor([[[ 0.0000,  1.0000,  0.0000,  1.0000],  
         [ 0.8415,  0.5403,  0.0100,  0.9999],  
         [ 0.9093, -0.4161,  0.0200,  0.9998],  
         [ 0.1411, -0.9900,  0.0300,  0.9996]]])
      ```

* Scaled Dot-Product Attention
  - 
  - 