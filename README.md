# all-about-attention

This is how I understand about attention mechanism.

1. The Background of Attention. Why the attention has been made?
   For training model with sequential data, researchers naturally think RNN(LSTM, GRU) structure. And it worked well. There is no doubt about the structure since sequential data has recurrence relation. But the problem of RNN architecture is that it precludes parallelization so that computation takes some times. Also RNN has limit of keeping long memory. So the researchers came up with the transformer model which use self-attention(Dense, Linear layer). There is no recurrence relation in attention, also no order. They put positional encoding for order and residual connection for long memory. 

2. How the self-attention works?
   The goal is to identify and attend to most important features in input.
   - Encode position information
   - Extract Query, Key, Value for search
   - Compute attention weighting
   - Extract features with high attention
   - FYI, Muti-head attention is just Concat(# of heads with self-attention). Also implys that input of attention splits into multiple parts of it.

![Attention score](https://raw.githubusercontent.com/delphinH/all-about-attention/main/attention_score.jpg)


3. How does Transformer look?
![Transformer Architecture](https://raw.githubusercontent.com/delphinH/all-about-attention/main/transformer_arch.PNG)

4. Behind Philosophy

5. Unsolved Questions
* Does the Embedding of input learn from data like CBOW, Skip-gram, or does it just fixed random vector? 

### Reference(Excellent sources)

* [MIT 6.S191 for general capturing concepts](https://youtu.be/ySEx_Bqxvvo)
* [Yannic Kilcher for the philosophy](https://youtu.be/TrdevFK_am4)
* [Jay Alammar with concrete example](https://jalammar.github.io/illustrated-transformer/)
* [My Favorite blog: Lil'Log](https://lilianweng.github.io/posts/2018-06-24-attention/)
* [Attention is all you need paper](https://arxiv.org/abs/1706.03762)
* [Aladdin Persson with implementation pytorch codes](https://youtu.be/M6adRGJe5cQ)