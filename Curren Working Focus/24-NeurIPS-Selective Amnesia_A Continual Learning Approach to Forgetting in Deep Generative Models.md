https://arxiv.org/abs/2305.10120

#### Similar Work
###### Data forgetting
1. Focus on *discriminative models* 判别模型
2. *partitioning* of data and model 数据 模型 划分
###### Concept erasure 
少数作品针对文本到图像的扩散模型，并通过利用这些模型的特定设计特征来工作

#### Introduction: continual learning
1. Given parameters for task A, train the model to perform task B without forgetting A
$$
\theta_A \rightarrow \theta_{A,B}
$$
2. Only generate B while forgetting A
$$
\theta_{A,B} \rightarrow \theta_B
$$


[[Diffusion Model]]

[[VAE (Variational Autoencoders)]]

[[Amnesiac Machine Learning, 2020]]

