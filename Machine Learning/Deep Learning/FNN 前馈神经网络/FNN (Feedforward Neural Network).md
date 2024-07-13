前馈神经网络由多层神经元组成，层间的神经元相互连接，层内的神经元不连接。其信息处理机制是:前一层神经元通过层间连接向后一层神经元传递信号，因为信号是从前往后转递的，所以是“前馈的”信息处理网络。这里，神经元是对多个输入信号(实数向量)进行非线性转换产生一个输出信号(实数值)的函数，整个神经网络是对多个输入信号(实数向量)进行多次非线性转换产生多个输出信号 (实数向量)的复合函数。每一个神经元的函数含有参数，神经网络的神经元的参数通过学习得到。当前馈神经网络的层数达到一定数量时(一般大于 2)，又称为深度神经网络(deep neural network，[[DNN]])。
前馈神经网络学习算法是反向传播 (back propagation) 算法，是 随机梯度下降法 的具体实现。学习的损失函数通常在分类时是 交叉熵损失，在回归时是 平方损失，其最小化等价于 极大似然估计。学习的正则化方法包括 早停法(early stopping)、暂退法(dropout)。

## 1 FNN 的模型

神经网络是由神经元连接组成的网络，采用不同类型的神经元以及神经元的不同连接方法可以构建出不同的网络结构，也就是不同的神经网络模型。本节讲述 FNN 的基本模型。首先给出 FNN 的定义，接着介绍具体例子，最后讨论 FNN 的表示能力。

#### 1.1 前馈神经网络定义
###### 1. 神经元
人工神经元 (artificial neuron) 是神经网络的基本单元。
生物神经元一般有多个树突接入，一个轴突接出。输入信号从树突传入，输出信号从轴突传出。当输入信号量达到阈值后，神经元被激活，产生输出信号。

与其对应，人工神经元是以实数向量为输入，实数值为输出的非线性函数，表示多个输入信号 (实数向量) 到一个输出信号 (实数值) 的非线性转换。

**定义**
神经元是如下定义的非线性函数:
$$
y = f(x_1,x_2,\cdots,x_n) = a \left(\sum_{i=1}^n w_i x_i + b \right)
$$
或者写作
$$
y = f(x_1,x_2,\cdots,x_n) = a(z),z = \sum_{i=1}^n w_ix_i + b
$$
其中，$x_1,x_2,\cdots,x_n$ 是输入，取实数值; $y$ 是输出，取实数值； $z$ 是中间结果，又称作净输入 (net input)，也取实数值; $w_1,w_2,\cdots,w_n$ 是权重 (weight), $b$ 是偏置 (bias)，也都取实数值；$z=\sum_{i=1}^n w_i x_i + b$ 是仿射函数; $a(\cdot)$ 是特定的非线性函数，成为激活函数。激活函数有多种形式，比如 $S$ 型函数:
$$
a(z) = \frac{1}{1+e^{-z}}
$$

神经元函数由两部分组成，首先使用仿射函数对输入 $x_1,x_2,\cdots,x_n$ 进行仿射变换，得到净输入 $z$，然后使用激活函数 $a(z)$ 对净输入 $z$ 进行非线性变换，得到输出 $y$。权重 $w_1,w_2,\cdots,w_n$ 与偏置 $b$ 是神经元函数的参数，通过学习得到。

结点表示变量，有向边表示变量之间的依存关系，有向图整体表示神经元函数。结点 $x_1,x_2,\cdots,x_n$ 是神经元的输入变量，结点 $y$ 是神经元的输出变量。通常不显示表示净输入变量 $z$。

![[Pasted image 20240713102317.png]]

神经元也可以用向量表示。设向量
$$
\mathbf x = \begin{bmatrix} x_1 \\x_2 \\ \vdots \\ x_n\end{bmatrix},\mathbf w = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n\end{bmatrix}
$$
为输入和权重，则神经元为函数
$$
y = f(\mathbf x) = a(\mathbf w^\mathrm T \mathbf x + b)
$$
或者写作
$$
y = f(\mathbf x) = a(z), z= \mathbf w^\mathrm T \mathbf x + b
$$
###### 2. 前馈神经网络
前馈神经网络由多层神经元组成，层间的神经元相互连接，层内的神经元不连接，前一层神经元的输出是后一层神经元的输入。整体表示输入信号 (实数向量)到输出信号(实数向量)的多次非线性转换。数学上，前馈神经网络是以实数向量为输入、以实数向量为输出的非线性函数的复合函数(这里，函数都是以向量为输入输出的一般函数的扩展)。前馈神经网络最后的输出也可以是一个实数值，是实数向量的特殊情况。先给出二层前馈神经网络的定义。

**定义 (二层前馈神经网络)**
二层前馈神经网络是如下定义的非线性函数的复合函数。输入是 $x_i,i=1,2,\cdots,n$，输出是 $y_k,k=1,2,\cdots,l$。神经网络由两层。第一层由 $m$ 个神经元组成，其中第 $j$ 个神经元是
$$
h_j^{(1)} = a \left(z_j^{(1)}\right) = a\left(\sum_{i=1}^n w_{ji}^{(1)}x_i + b_j^{(1)} \right), j =1,2,\cdots,m
$$
这里 $x_i$ 是输入，$w_{ji}^{(1)}$ 是权重，$b_{j}^{(1)}$ 是偏置，$z_j^{(1)}$ 是净输入，$a(\cdot)$ 是激活函数。第二层由 $l$ 个神经元组成，其中第 $k$ 个神经元是
$$
y_k = g \left(z_k^{(2)}\right) = g \left(\sum_{j=1}^m w_{kj}^{(2)} h_j^{(1)} +b_k^{(2)} \right),k=1,2,\cdots,l
$$
这里 $h_j^{(1)}$ 是第一层神经元的输出，$w_{kj}^{(2)}$ 是权重，其中 $j=1,2,\cdots,m$，$b_k^{(2)}$ 是偏置，$z_k^{(2)}$ 是净输入，$g(\cdot)$ 是激活函数。神经网络整体是
$$
y_k = g \left [\sum_{j=1}^m w_{kj}^{(2)}a \left(\sum_{i=1}^n w_{ji}^{(1)}x_i + b_j ^{(1)} \right) + b_k^{(2)} \right],k=1,2,\cdots,l
$$
通常情况第二层只有一个神经元，即 $l=1$。

第一层神经元从输入输出的角度不可见，称为隐层。第二层神经元称为输出层。有时把输入也看作是一层，称为输入层。隐层和输出层的激活函数 $a(·)$ 和 $g(·)$ 通常有不同的定义。这里考虑层间的全连接，即前一层的每一个神经元都和后一层的每一个神经元连接。部分连接网终是其特殊情况，相当于未连接边的权重为 0。

![[Pasted image 20240713104510.png]]
二层前馈神经网络也可以用矩阵来表示。
$$
\mathbf h^{(1)} = f^{(1)} (\mathbf x) = a(\mathbf z^{(1)}) = a \left(\mathbf W(1)^{\mathrm T} \mathbf x + \mathbf b^{(1)} \right)
$$
$$
\mathbf y = f^{(2)} (\mathbf x) = a(\mathbf z^{(2)}) = g \left(\mathbf W(2)^{\mathrm T} \mathbf h^{(1)} + \mathbf b^{(2)} \right)
$$
其中
$$
\mathbf x = \begin{bmatrix}x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix},\mathbf z^{(1)} = \begin{bmatrix}z_1^{(1)} \\ z_{2}^{(1)} \\ \vdots \\ z_{m}^{(1)} \end{bmatrix}, \mathbf h^{(1)} = \begin{bmatrix}h_1^{(1)} \\ h_2^{(1)} \\ \vdots \\ h_m^{(1)} \end{bmatrix},\mathbf z^{(2)} = \begin{bmatrix}z_1^{(2)} \\ z_{2}^{(2)} \\ \vdots \\ z_{l}^{(2)} \end{bmatrix} 
$$
$$
\mathbf W^{(1)} = 
\begin{bmatrix}
w_{11}^{(1)} & \cdots & w_{1m}^{(1)}\\
\vdots & & \vdots\\
w_{n1}^{(1)} & \cdots & w_{nm}^{(1)}
\end{bmatrix},\mathbf W^{(2)} = 
\begin{bmatrix}
w_{11}^{(2)} & \cdots & w_{1l}^{(2)}\\
\vdots & & \vdots\\
w_{m1}^{(2)} & \cdots & w_{ml}^{(2)}
\end{bmatrix},\mathbf b^{(1)} = \begin{bmatrix}b_1^{(1)} \\ b_{2}^{(1)} \\ \vdots \\ b_{m}^{(1)} \end{bmatrix} ,\mathbf b^{(2)} = \begin{bmatrix}b_1^{(2)} \\ b_{2}^{(2)} \\ \vdots \\ b_{l}^{(2)} \end{bmatrix} 
$$
**定义 (多层前馈神经网络)**
多层前馈神经网络或前馈神经网络是如下定义的非线性函数的复合函数。输入是 $x_i,i=1,2,\cdots,n$，输出是 $y_k,k=1,2,\cdots,l$。神经网络有 $s$ 层 ($s\ge 2$)。第一层到第 $s-1$ 是隐层。假设其中的第 $t$ 层由 $m$ 个神经元组成。第 $t-1$ 层由 $n$ 个神经元组成，$t=1,2,\cdots,s-1$，第 $t$ 层的第 $j$ 个神经元是
$$ 
h_j^{(t)} = a \left(z_j^{(t)} \right) = a \left(\sum_{t=1}^n w_{ji}^{(t)}h_i^{(t-1)}+b_j^{(t)}\right),j=1,2,\cdots,m
$$
这里 $h_i^{(t-1)},i=1,2,\cdots,n$，是第 $t-1$ 层的输出，设 $h_i^{(0)}=x_i,w_{ji}^{(t)},i=1,2,\cdots,n$，是权重，$b_j^{(t)}$ 是偏置，$z_j^{(t)}$ 是净输入，$a(\cdot)$ 是激活函数。第 $s$ 层是输出层。假设第 $s$ 层由 $l$ 个神经元组成，第 $s-1$ 层由 $m$ 个神经元组成，第 $s$ 层的第 $k$ 个神经元是
$$
y_k = g(z_k^{(s)}) = g \left(\sum_{j=1}^m w_{kj}^{(s)} h_j^{(s-1)} + b_k^{(s)} \right),k=1,2,\cdots,l
$$
这里 $h_j^{(s-1)},j=1,2,\cdots,m$，是第 $s-1$ 层的输出，$w_{kj}^{(s)},j=1,2,\cdots,m$，是权重，$b_k^{(s)}$ 是偏置，$z_k^{(s)}$ 是净输入，$g(\cdot)$ 是激活函数。神经网络整体是
$$
y_k = g \left\{\sum_{j=1}^m w_{kj}^{(s)} \cdots \left[a\left(\sum_{i=1}^{n}w_{ji}^{(1)}x_i + b_j^{(1)} \right) \right] \cdots+ b_k^{(s)}\right\},k=1,2,\cdots,l
$$
层数大于 2 时的前馈神经网络又称为 深度神经网络 DNN，通常情况是第 $s$ 层只有一个神经元，即 $l=1$。

前馈神经网络的矩阵表示如下
$$
\begin{cases}
&\mathbf h^{(1)} = f^{(1)}(\mathbf x) = a(\mathbf z^{(1)}) = a \left(\mathbf {W^{(1)}}^{\mathrm T} \mathbf x+\mathbf b^{(1)} \right)\\
&\mathbf h^{(2)} = f^{(2)}(\mathbf h^{(1)}) = a(\mathbf z^{(2)}) = a \left(\mathbf {W^{(2)}}^{\mathrm T} \mathbf h^{(1)}+\mathbf b^{(2)} \right)\\
&\vdots\\
&\mathbf h^{(s-1)} = f^{(s-1)}(\mathbf h^{(s-1)}) = a(\mathbf z^{(s-1)}) = a \left(\mathbf {W^{(s-1)}}^{\mathrm T}\mathbf h^{(s-2)}+\mathbf b^{(s-1)} \right)
\\
&\mathbf y = \mathbf h^{(s)} = f^{(s)} (\mathbf h^{(s-1)}) = g(\mathbf z^{(s)}) = g \left({\mathbf W^{(s)}}^\mathrm T\mathbf h^{(s-1)} + \mathbf b^{(s)} \right)
\end{cases}
$$
整体神经网络由复合函数 $f^{(s)} (\cdots f^{(2)} (f^{(1)}(\mathbf x))\cdots)$ 表示，也可以写作 $f(\mathbf x; \theta)$，其中 $\mathbf \theta$ 是所有参数组成的向量。

但到目前为止考虑到是一个样本输入到神经网络的情况，这时输入由一个向量表示。也可以是多个样本批量同时输入到神经网络，这时输入样本由一个矩阵表示。可以用矩阵表示扩展，之后介绍。
###### 3. 隐层到神经元
隐层神经元函数由两部分组成：仿射函数和激活函数。这里介绍常用的隐层激活函数，包括 S 型函数、双曲正切函数、整流线形函数。一个神经网络通常采用一种隐层激活函数
S 型函数 (sigmoid function) 又称为逻辑斯谛函数 (logistic function)，是定义式如下的非线性函数。
$$
a(z) = \sigma (z) = \frac{1}{1+e^{-z}}
$$
其中，$z$ 是自变量或输入，$\sigma(z)$ 是因变量或输出。函数的定义域为 $(-\infty,+\infty)$，值域为 $(0,1)$。
![[Pasted image 20240713133605.png]]

S 型函数的导函数是
$$
a'(z) = a(z) (1-a(z))
$$
双曲正切函数 (hyperbolic tangent function) 是定义式如下的非线性函数。
$$
a(z) = \tanh (z) = \frac{e^{z} - e^{-z}}{e^z + e^{-z}}
$$
其中，$z$ 是自变量或输入，$\tanh (z)$ 是因变量或输出。函数定义域为 $(-\infty,+\infty)$，值域为 $(-1,+1)$。
![[Pasted image 20240713133953.png]]
双曲正切函数的导函数是
$$
a'(z) = 1-a(z)^2
$$
双曲正切函数与 S 型函数有以下关系：直观上双曲正切函数将S型函数“放大”两倍，并向下平移 1 个单位。
$$
\tanh (z) = 2\sigma(2z) - 1
$$
整流线性函数 (rectified linear unit, ReLU) 是定义式如下的非线性函数。
$$
a(z) = \text{relu} (z) = \max(0,z)
$$
其中，$z$ 是自变量或输入，$\text{relu}(z)$ 是因变量或输出。函数定义域为 $(-\infty,+\infty)$，值域为 $(0,+\infty)$。
![[Pasted image 20240713135812.png]]
整流线形函数的导函数是
$$
a'(z) = \begin{cases} 1, &z>0\\ 0, &其他\end{cases}
$$
整流线性函数比 $S$ 型函数和双曲正切函数在计算机上的计算效率更高，其导函数也是如此。整流线形函数在当前深度学习中被广泛使用。

对于激活函数 $a(z)$，当导数满足 $\lim_{z\rightarrow -\infty} a'(z) = 0$ 时，称为左饱和 (left saturating) 函数：当其导数满足 $\lim_{z \rightarrow +\infty} a'(z) = 0$ 时，