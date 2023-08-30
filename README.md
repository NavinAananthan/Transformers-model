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
  - 