深度解析 文章 Attention is all you need (Transformer 论文)

## Abstract
主要的序列转导模型是基于复杂的循环 (RNN) 或卷积神经(CNN) 网络，包括一个编码器和一个解码器。表现最好的模型还通过注意机制连接编码器和解码器。我们提出了一个新的简单的网络架构，变形金刚 (Transformer)，完全基于注意力机制 (attention mechanisms)，完全摒弃递归和卷积。在两个机器翻译任务上的实验表明，这些模型在质量上更优越，同时更具并行性，并且需要更少的训练时间。我们的模型在WMT 2014英语-德语翻译任务上实现了28.4 BLEU，比现有的最佳结果(包括集合)提高了2个BLEU以上。在WMT 2014英法翻译任务中，我们的模型在8个gpu上训练3.5天后，建立了一个新的单模型最先进的BLEU分数41.0，这是文献中最佳模型训练成本的一小部分。

BLEU 是在机器翻译里大家经常用的一个衡量标准。
## 1 Introduction


## 2 Background


## 3 Model Architecture

#### 3.1 Encoder and Decoder Stacks

#### 3.2 Attention

###### 3.2.1 Scaled Dot-Product Attention

###### 3.2.2 Multi-Head Attention

###### 3.2.3 Application of Attention in our Model

#### 3.3 Position-wise Feed-Forward Networks

#### 3.4 Embeddings and Softmax

#### 3.5 Positional Encoding

## 4 Why Self-Attention

## 5 Training

#### 5.1 Training Data and Batching

#### 5.2 Hardware and Schedule

#### 5.3 Optimizer

#### 5.4 Regularization

## 6 Results

#### 6.1 Machine Translation

#### 6.2 Model Variations

## 7 Conclusion

