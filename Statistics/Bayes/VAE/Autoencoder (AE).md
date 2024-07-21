**自编码器** (autoencoder) 也称 **自动编码器**，是一种 人工神经网络，用于学习 无标签数据 的 有效编码；属于一种 无监督学习 [[Unsupervised learning]]

**自编码** (autoencoding) 的目的是：学习对高维度数据做低维度”表示“ （“表征”或“编码”）；因此，通常用于 **降维**。最近，自编吗的概念广泛地用于数据的 **生成模型**。

自 2010 年以来，一些先进的人工智能在 **深度学习** 网络中采用了堆叠式稀疏自编码。

###### 基本结构
自编码器有两个主要部分组成：**编码器**用于将**输入编码**，而**解码器**使用**编码重构输入**。

实现这个功能最简单的方式就是重复原始信号。然而，自编码器通常被迫近似地重构输入信号，重构结果仅仅包括原信号中最相关的部分。

自编码器的思想已经流行了几十年，其首次应用可以追溯到20世纪80年代。自编码器最传统的应用是[降维](https://zh.wikipedia.org/wiki/%E9%99%8D%E7%BB%B4 "降维")或[特征学习](https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E5%AD%A6%E4%B9%A0 "特征学习")，现在这个概念已经推广到用于学习数据的生成模型。21世纪10年代的一些最强大的人工智能在深度神经网络中采用了自编码器。

最简单的自编码器形式是一个**前馈**的、**非循环**的神经网络，用**一层或多层隐藏层**链接输入和输出。**输出层节点数**和**输入层**一致。其目的是**重构输入**（最小化输入和输出之间的差异），而不是在给定输入的情况下预测目标值，所以自编码器属于**无监督学习**。

自编码器由编码器和解码器组成，二者可以被分别定义为 $\phi$ 和 $\psi$，使得：
$$\begin{aligned}
&\phi : \mathcal X \rightarrow \mathcal F\\
&\psi : \mathcal F \rightarrow \mathcal X\\
&\phi, \psi = \mathop{\arg \min}_{\phi,\psi} \| \mathcal X - (\psi \circ \phi)\mathcal X\|^2
\end{aligned}$$
在最简单的情况下，给定一个隐藏层，自编码器的编码阶段接受输入 $x\in \mathbb R^d = \mathcal X$ 并将其映射到 $\mathbf h \in \mathbb R^{p} = \mathcal F$
$$
\mathbf h = \sigma (\mathbf W \mathbf x + \mathbf b)
$$
$\mathbf h$ 通常表示编码、潜变量或潜在表示。
$\sigma$ 是一个逐元素的 激活函数 (例如 sigmoid 或 ReLU)
$\mathbf W$ 是权重矩阵 (通常随机初始化，并在训练期间通过反向传播迭代更新。)
$\mathbf b$ 是偏置向量 (通常随机初始化并在训练期间通过反向传播迭代更新)

自编码器的解码阶段映射 $\mathbf h$ 到 重构 $\mathbf x'$ (与 $\mathbf x$ 形状一致)
$$
\mathbf x' = \sigma' (\mathbf W' \mathbf h + \mathbf b')
$$
其中解码器部分的 $\sigma',\mathbf W',\mathbf b'$ 可能与编码器部分的 $\sigma,\mathbf W,\mathbf b$ 无关

自编码器被训练来最小化重建误差 (如平方误差)，通常被称为“损失“：
$$
\mathcal L(\mathbf x,\mathbf x') = \|\mathbf x- \mathbf x' \|^2 = \| \mathbf x - \sigma'(\mathbf W' (\sigma (\mathbf W\mathbf x + \mathbf b))+\mathbf b')\|^2
$$
其中 $\mathbf x$ 通常在训练集上平均。

如前所述，和其它前馈神经网络一样，自编码器的训练是通过误差的反向传播进行的。

当特征空间 $\mathcal F$ 的维度比输入空间 $\mathcal X$ 低时，特征向量 $\phi(x)$ 可以看作是输入 $x$ 的研所表示，这就是不完备自动编码 (undercomplete autoencoders) 的情况。如果隐藏层大于 (过完备) 或等于输入层的数量，或隐藏单元的容量足够大，自编码器可能学会恒等函数而变得无用。然而，实验结果表明过完备自编码器 (overcomplete autoencoders) 仍然可能学习到有用的特征。在理想情况下，编码的维度和模型容量可以根据待建模数据分布的复杂性来设定，采用这种方式的一种途径是正则化自编码器。
![[Pasted image 20240720195834.png]]

