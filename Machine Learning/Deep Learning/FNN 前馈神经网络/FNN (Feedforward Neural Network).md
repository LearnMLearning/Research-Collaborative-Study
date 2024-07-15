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
###### 3. 隐层的神经元
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

对于激活函数 $a(z)$，当导数满足 $\lim_{z\rightarrow -\infty} a'(z) = 0$ 时，称为左饱和 (left saturating) 函数：当其导数满足 $\lim_{z \rightarrow +\infty} a'(z) = 0$ 时，称为右饱和 (right saturating) 函数：同时满足左饱和、右饱和条件时称为 (两边的) 饱和 (saturating) 函数。整流线形函数时左饱和函数，S 型函数和双曲正切函数是饱和函数。

###### 4. 模型
前馈神经网络可以作为机器学习模型用于不同任务，有几种代表情况。
(1) 用于**回归**。神经网络的输出层只有一个神经元，其输出是一个实数值。神经网络表示为 $y=f(\mathbf x)$，其中 $y\in \mathcal R$。预测时给定输入 $\mathbf x$，计算输出 $y$。

(2) 用于**二类分类**。神经网络的输出层只有一个神经元，其输出是一个概率值。神经网络表示为 $p=P(y=1|\mathbf x)=f(\mathbf x)$，其中 $y\in \{0,1\}$，满足条件
$$
0 < P(y=1 | \mathbf x) < 1, P(y=1|\mathbf x) + P(y=0 | \mathbf x) = 1
$$
预测时给定输入 $\mathbf x$，计算其属于类别 1 的概率。如果概率大于 0.5，则将输入分到类别 1，否则分到类别 0。

(3) 用于 **多类分类 (multi-class classification)**。神经网络的输出层只有一个神经元，神经元的输出是由 $l$ 个概率值组成的概率向量。神经网络表示为 $\mathbf p = [P(y_k = 1|\mathbf x)] = f(\mathbf x)$，其中 $y_k \in \{0,1\},k=1,2,\cdots,l$，满足条件
$$
\sum_{k=1}^i y_k =1,0<P(y_k=1|x) < 1,\sum_{k=1}^l P(y_k = 1|\mathbf x) = 1,k=1,2,\cdots ,l
$$
也就是说 $[y_1,y_2,\cdots,y_l]$ 是只有一个元素为 $1$，其他元素为 $0$ 的向量，这样的向量称为独热向量 (one-hot vector)，$[P(y_1=1|\mathbf x),P(y_2=1|\mathbf x),\cdots,P(y_l=1|\mathbf x)]$ 是定义在独热向量上的概率分布，表示输入 $\mathbf x$ 属于 $l$ 个类别的概率。预测时给定输入 $\mathbf x$，计算其属于各个类别的概率。将输入分到概率最大的类别，这时输入只可能被分到一个类别。

(4) 用于 **多标签分类 (multi-label classification)**。神经网络的输出层有 $l$ 个神经元，每个神经元的输出是一个概率值。神经网络表示为 $\mathbf p = [P(y_k=1)|\mathbf x]=f(\mathbf x)$，其中 $y_k \in \{0,1\},k=1,2,\cdots,l$ 满足条件
$$
0 < P(y_k = 1 | \mathbf x) < 1, P(y_k = 1 |\mathbf x) + P(y_k = 0 |\mathbf x)=1,k=1,2,\cdots,l
$$
$[P(y_1 = 1 | \mathbf x),\cdots,P(y_l=1|\mathbf x)]$ 表示输入 $\mathbf x$ 分别属于 $1$ 个类别的概率。预测时给定输入 $\mathbf x$，计算其属于各个类别的概率。将输入分到概率大于 $0.5$ 的所有类别，这时输入可以被分到多个类别 (赋予多个标签)。

注意，在**回归**中神经网络的输出和模型的输出是相同的，都是实数值; 在**分类**中神经网络的输出和模型的输出是不同的，前者是概率值，后者是类别。现实中经常对神经网络及其表示的模型不严格区分，这一点其他类型的神经网络也一样。

###### 5. 输出层的神经元
输出层神经元由两部分组成：仿射函数和激活函数。输出层激活函数通常使用恒等函数、S型函数、软最大化函数。在回归、二类分类、多类分类、多标签分类中，激活函数有不同的形式。

**回归**时，输出层只有一个神经元，其激活函数是恒等函数，神经元函数是
$$
y = g(z) = z, z = \mathbf {w^{(s)}}^{\mathrm T} \mathbf h ^{(s-1)} + b^{(s)}
$$
这里 $\mathbf w^{(s)}$ 是权重向量，$b^{(s)}$ 是偏置，$g(\mathbf \cdot)$ 是恒等函数，$\mathbf h^{(s-1)}$ 是第 $s-1$ 隐层的输出。称这样的输出层为线性输出层。

**二类分类**时，输出层只有一个神经元，其激活函数是 S 型函数，神经元函数是
$$
P(y=1|\mathbf x) = g(z) = \frac{1}{1+e^{-z}}, z= {\mathbf w^{(s)}}^{\mathrm T} \mathbf h^{(s-1)} + b^{(s)}
$$
这里 $\mathbf w^{(s)}$ 是权重向量，$b^{(s)}$ 是偏置，$g(\mathbf \cdot)$ 是恒等函数，$\mathbf h^{(s-1)}$ 是第 $s-1$ 隐层的输出。

**多类分类** 时，输出层只有一个神经元，其激活函数是软最大化函数 (softmax function)，神经元函数是
$$
P(y_k =1|\mathbf x) = g(z_k) = \frac{e^{z_k}}{\sum_{i=1}^l e^{z_i}},z_k = {\mathbf w_k^{(s)}}^\mathrm T \mathbf h^{(s-1)} + b_k^{(s)}, k =1,2,\cdots,l
$$
这里 $\mathbf w^{(s)}$ 是权重向量，$b^{(s)}$ 是偏置，$g(\mathbf \cdot)$ 是恒等函数，$\mathbf h^{(s-1)}$ 是第 $s-1$ 隐层的输出。称这样的输出层为软最大化输出层。

软最大化函数的名字来自它是最大化 (max) 函数的近似这一事实。如果 $z_k \ll z_j,j\ne k$，那么 $p_k = P(y_k = 1) \approx 1,p_j = P(y_j = 1) \approx 0$。软最大化函数是 $l$ 维实数向量 $\mathbf z$ 到 $l$ 维概率向量 $\mathbf p$ 的映射
$$
z = 
\begin{bmatrix}
z_1 \\
z_2 \\
\vdots\\
z_l
\end{bmatrix} \rightarrow \mathbf p = 
\begin{bmatrix}
p_1 \\
p_2 \\
\vdots\\
p_l
\end{bmatrix}
$$
软最大化函数的偏导数或雅可比矩阵元素是 (推导见附录 $F$)
$$
\frac{\partial p_k}{\partial z_j} = \begin{cases}p_k (1-p_k),&j=k\\-p_jp_k,& j\ne k\end{cases}, \, j,k = 1,2,\cdots ,l
$$
前馈神经网络用于多类分类时，为了提高效率，在预测时经常省去激活函数的计算，选取仿射函数(净输入) $z_k$ 值最大的类别。这样做，分类结果是等价的，因为软最大化函数的分母对各个类别是常量，而分子的指数函数是单调递增函数。实数值 $z_k$ 又称为对数几率(logit)。

多标签分类时，输出层有 $l$ 个神经元，每个神经元的激活函数都是 S 型函数，神经元函数是
$$
P(y_k = 1|\mathbf x) = g(z_k) = \frac{1}{1+e^{-z_k}}, z_k = {\mathbf w_k^{(s)}}^{\mathrm T} \mathbf h^{(s-1)} + b_k^{(s)},k=1,2,\cdots,l
$$
这里 $\mathbf w_k^{(s)}$ 是权重向量，$b_k^{(s)}$ 是偏置，$g(\cdot)$ 是 S 型函数，$\mathbf h^{(s-1)}$ 是第 $s-1$ 隐层的输出。
#### 1.2 前馈神经网络的例子
**例 1**
![[Pasted image 20240713172026.png]]
**解** 第一层 (隐层) 的第一个神经元与例 23.1 相同，仿射函数是 $z=-x_1+2x_2+1$，激活函数是 $a(z) = \frac{1}{1+e^{-z}}$，神经元函数是
$$
h_1^{(1)} = \frac{1}{1+e^{x_1-2x_2-1}}
$$
图 23.3 是该神经元的三维图形。

第一层的第二个神经元的仿射函数是 $z=-4x_1-x_2+2$，激活函数是 $a(z) = \frac{1}{1+e^{-z}}$，神经元函数是
$$
h_2^{(1)} = \frac{1}{1+e^{4x_1+x_2-2}}
$$
![[Pasted image 20240713172305.png]]
第二层（输出层）的神经元仿射函数是 $z=h_1^{(1)} + h_2^{(1)}$，激活函数是 $g(z) = \sigma(z)$，神经元函数是 $f(x_1,x_2)=\sigma (h_1^{(1)} + h_2^{(1)})$。神经网络整体
$$
f(x_1,x_2) = \sigma \left(\frac{1}{1+e^{x_1 -2x_2-1}} + \frac{1}{1+e^{4x_1 + x_2 - 2}} \right)
$$

是一个二类分类模型。图 23.10 是神经网络的三维图形。图形中 “高原” 部分的输出值接近 $1$，“盆地” 部分的输出值接近 0。可以看出，整个二层前馈神经网络能够比第一层的两个神经元表示更复杂的非线性关系。

**例2** 构建一个前馈神经网络实现逻辑表达式 XOR 的功能。
**解** 采用矩阵表示，构建一个二层前馈神经网络，第一层有两个神经元，其激活函数是整流线形函数，第二层有一个神经元，其激活函数是恒等函数，如图 23.11 所示。
![[Pasted image 20240713173510.png]]
第一层的权重矩阵和偏置向量是
$$
\mathbf W^{(1)} = \begin{bmatrix}1&1\\1&1 \end{bmatrix}, \mathbf b^{(1)} = \begin{bmatrix}0\\-1 \end{bmatrix}
$$
第二层的权重矩阵和偏置向量是
$$
\mathbf W^{(2)} = \begin{bmatrix}1\\-2 \end{bmatrix},\mathbf b^{(2)} = [0]
$$

用矩阵表示四种可能的输入：
$$
\mathbf X = \begin{bmatrix}0 & 0 & 0 & 1 \\ 0 & 1 & 0 & 1 \end{bmatrix}
$$
代表批量处理。代入神经网络，第一层输出是
$$\begin{aligned}
\mathbf H^{(1)} &= \text{relu} \left({\mathbf W^{(1)}}^{\mathrm T} \mathbf X + \mathbf B^{(1)}\right)\\
&= \text{relu}\left(\begin{bmatrix}1 & 1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} 0 & 0 & 1 & 1\\ 0 & 1 & 0 & 1\end{bmatrix} + \begin{bmatrix}0 & 0 & 0 & 0 \\ -1 & -1 & -1 & -1\end{bmatrix}\right)\\
&= \text{relu}\left(\begin{bmatrix}0 & 1 & 1 & 2 \\ -1 & 0 & 0 & 1 \end{bmatrix} \right)\\
&= \begin{bmatrix}0 & 1 & 1 & 2\\ 0 & 0 & 0 & 1 \end{bmatrix}
\end{aligned}$$
其中 relu 计算对矩阵的每一个元素进行。第二层输出是
$$\begin{aligned}
\mathbf H^{(2)} &= {\mathbf W^{(2)}}^{\mathrm T} \mathbf H^{(1)} + \mathbf B^{(2)}\\
&= \begin{bmatrix}1 & -2 \end{bmatrix} \begin{bmatrix}0 & 1 & 1 & 2 \\ -1 & 0 & 0 & 1 \end{bmatrix} + \begin{bmatrix}0 & 0 & 0 & 0 \end{bmatrix}\\
&= \begin{bmatrix}0 & 1 & 1 &0 \end{bmatrix}
\end{aligned}$$
作为线性模型的感知机不能实现 XOR 是众所周知的事实，而作为非线性模型的前馈神经网络可以实现 XOR，并且以很简单的方式实现。

###### 3. 前馈神经网络的表示能力
**1. 与其他模型的关系**
前馈神经网络与逻辑斯谛回归模型、感知机、支持向量机等有密切关系。
对于多类分类的一层神经网络，当其输出层激活函数是软最大化函数是，模型等价于多想逻辑斯谛回归模型
$$
P(y_k=1|\mathbf x) = f(\mathbf x) = \frac{e^{\left({\mathbf w_i^{(1)}}^{\mathrm T} \mathbf x + b_i^{(1)}\right)}}{\sum_{i=1}^l e^{\left({\mathbf w_i^{(1)}}^{\mathrm T} \mathbf x + b_i^{(1)}\right)}},k=1,2,\cdots,l
$$
所以，前馈神经网络是逻辑斯谛回归模型的扩展。注意：前馈神经网络通常将所有 $l$ 个类别的权重和偏置作为参数使用，而逻辑斯谛回归通常将前 $l-1$ 个类别的权重和偏置作为自由参数使用，

对于二类分类的一层神经网络，当其输出层激活函数是双曲正切函数时，
$$
f(\mathbf x) = \tanh ({\mathbf w^{(1)}}^{\mathrm T} + b^{(1)})
$$
模型可以与感知机对应。感知机模型的定义是
$$
y = \begin{cases}+1, & \text{sign} (\mathbf w^{\mathrm T} \mathbf x + b) \ge 0\\
-1, & 其他\end{cases}
$$
所以，可以认为前馈神经网络是感知机的扩展。这也是前馈神经网络又被称为多层感知机的原因。

对于二类分类的多层神经网络没当其输出层激活函数是双曲正切函数时，
$$
f(\mathbf x) = \tanh({\mathbf w^{(s)}}^{\mathrm T} \mathbf h^{(s-1)} + b^{(s)})
$$
$$
\mathbf h^{(s-1)} = f^{(s-1)} [\cdots f^{(2)}(f^{(1)}(\mathbf x))\cdots]
$$
模型可以与非线性支持向量机对应。非线性支持向量机模型的定义是
$$
y = \begin{cases}
+1, & \text{sign} (\mathbf w^{\mathrm T} \phi(x) + b)\\
-1, & 其他
\end{cases}
$$
其中，$\phi(x)$ 是从输入空间到特征空间的非线性映射，$\mathbf w$ 和 $b$ 是模型的参数。前馈神经网络的前 $s-1$ 层函数 $f^{(s-1)}[\cdots f^{(2)}(f^{(1)}(\mathbf x))\cdots]$ 与映射函数 $\phi(\mathbf x)$ 对应。支持向量机学习是凸优化问题，保证可以找到全局最优，而前馈神经网络学习是非凸优化问题，不能保证找到全局最优。前馈神经网络比支持向量机有更多的参数可以调节。

**2. 函数近似能力**
前馈神经网络具有强大的函数近似能力。通用近似定理 (universal approximation theorem) 指出，存在一个二层前馈神经网络，具有一个线性输出层和一个隐层，其中隐层含有充分数量的神经元，激活函数为挤压函数，这个网络可以以任意精度近似任意一个在紧的定义域上连续函数。从这个意义上，前馈神经网络的函数近似能力是通用的。

设有实函数 $G(x):\mathcal R \rightarrow[0,1]$，如果 $G(x)$ 是非减函数，且满足 $\lim_{x\rightarrow -\infty} G(x) = 0,\lim_{x\rightarrow +\infty} G(x) = 1$，则称函数 $G(x)$ 为挤压函数 (squashing function)。S 型函数是一种挤压函数。

后续理论研究发现，定理的条件可以放宽，当激活函数是多项式函数以外的其他函数时，或者当被近似函数是波莱尔可测函数时，定理的结论依然成立。波莱尔可测函数包括连续函数、分段连续函数、阶梯函数。

下面的定理是通用近似定理的一个具体形式。

**定理** 对任意连续函数 $h:[0,1]^n \rightarrow \mathcal R$ 和任意 $\varepsilon >0$，存在一个二层前馈神经网络；
$$\begin{aligned}
f(\mathbf x) &= \mathbf \alpha^{\mathrm T} \sigma (\mathbf W^{\mathrm T}\mathbf x + \mathbf b)\\
&= \sum_{j} \alpha_j \sigma \left(\sum_{i} w_{ji}x_i + b_j \right)
\end{aligned}$$
使得对于任意 $x\in [0,1]^n$，有 $|\mathbf h(x)-f(\mathbf x)|<\varepsilon$ 成立。这里隐层的激活函数是 S 型函数。

下面给出定理的直观解释。假设 $h(x)$ 是一个连续函数，定义域是区间 $[0,1]$，值域是区间 $[0,1]$，可以用二层前馈神经网络 $f(x)$ 以任意精度近似 $h(x)$。（为了简单，这里假设 $h(x)$ 取正值，也可以取负值）
![[Pasted image 20240714110020.png]]
如图 23.15 所示，可以用阶梯函数以任意精度近似函数 $h(x)$。如图 (b) 所示，假设阶梯函数第 $i$ 个分段函数是
$$
s_i(x) = \begin{cases}\alpha_i, &x_{i-1} < x \le x_i \\ 0, &其他 \end{cases}
$$
则该分段函数可以由以下二层神经网络近似。
$$
f_i(x) = \alpha_i \cdot \sigma (w \cdot x - x_{i-1}) - \alpha_i \cdot \sigma(w\cdot x - x_i)
$$
其中隐层有两个神经元，其激活函数是 S 型函数，输出层是线性的。这里参数 $x_{i-1}$ 和 $x_i$ 保证与分段函数的区间一致，参数 $\alpha_i$ 保证趋近分段函数，参数 $w$ 控制与分段函数的趋近程度。这样，阶梯函数的每一分段函数 $s_i(x)$ 都可以用一个二层神经网络 $f_i(x)$ 近似，函数 $h(x)$ 整体也可以由所有 $f_i(x)$ 相加得到的二层神经网络 $f(x)$ 近似。

通用近似定理叙述的是理论存在性，并不意味着现实可行性。定理 23.1 中近似连续函数的二层前馈神经网络的隐层神经元的个数可能是非常大的，甚至是指数级的，参数个数也是如此，现实中不会有足够多的数据训练这样的网络。经验上，当前馈神经网络的层数增大时,也就是变成深度神经网络时，可以解决这个问题。

###### 3. 函数等价性
前馈神经网络 $\mathbf y = f(\mathbf x; \mathbf \theta)$ 由大量的等价函数，即参数 $\mathbf \theta$ 不同但对相同的输入 $\mathbf x$ 产生相同的输出 $\mathbf y$，而且等价函数的个数是指数级的。

假设某个隐层有 $m$ 个神经元，其所有参数由向量表示。这一层有 $m!$ 个神经元的排列,每一个排列决定一个参数向量，因此有 $m!$ 种不同的参数向量。改变神经元的排列，参数向量发生变化，但神经网络的输入输出的映射关系不变，这时隐层有 $m!$ 个等价的参数向量。

假设某个隐层有 $m$ 个神经元，其所有参数由向量表示，激活函数是双曲正切函数。双曲正切函数是奇函数，即满足 $\tanh(-z)= -\tanh(z)$。若这一层的某一个神经元的参数以及相连的后一层神经元的参数都反号，则对相同的输入，神经网络的输出不变。这时隐层(与后一层一起)有 $2^m$ 个等价的参数向量。如果同时考虑神经元的不同排列，这个隐层共有 $m!2^m$ 个等价的参数向量。

当神经网络有多个隐层的时候，整体的等价函数的个数由各层的等价参数向量个数的乘积决定。›
###### 4. 网络的深度
前馈神经网络的深度指网络的层数，复杂度指神经元的个数。复杂度也代表神经网络的参数个数，因为参数个数与神经元个数成正比。深度神经网络与 “浅度神经网络” 可以用同等的表示能力，但深度神经网络比浅度神经网络有更低的复杂度。这一点可以由逻辑门电路理论间接论证。

![[Pasted image 20240714112659.png]]
![[Pasted image 20240714112711.png]]

## 2 前馈神经网络的学习算法
#### 2.1 前馈神经网络学习
###### 1. 一般形式
前馈神经网络的学习和预测时如下的监督学习问题。给定训练数据集
$$
\mathcal T = \{ (\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\}
$$
其中，$(\mathbf x_i,y_i),i=1,2,\cdots,N$，表示样本，由输入 $\mathbf x_i$ 与输出 $y_i$ 的对组成：$N$ 表示样本容量。学习一个前馈神经网络模型 $f(\mathbf x; \hat {\mathbf \theta})$，其中 $\hat{\mathbf \theta}$ 是估计的神经网络的参数向量。用学到的模型对新的输入 $\mathbf x_{N+1}$ 给出新的输出 $y_{N+1}$。

学习时通常假设神经网络的架构已经确定，包括网络的层数、每层的神经元数、神经元激活函数的类型。所以网络的参数已确定，需要从数据中学习或估计的是参数值。

学习问题可以形式化为以下的优化问题:
$$
\hat {\mathbf \theta} = \mathop{\text{argmin}}_{\mathbf \theta} \left[\sum_{i=1}^{N} L(f(\mathbf x_i; \mathbf\theta),y_i) + \lambda \cdot \Omega(f)\right]
$$
其中，$L(\cdot)$ 是损失函数，$\Omega(\cdot)$ 是正则项，$\lambda \ge 0$ 是系数。当损失函数是对数损失函数、没有正则化时，问题变成极大似然估计。这时前馈神经网络学习的一般形式。
$$
\hat {\mathbf \theta} = \mathop{\text{argmin}}_{\mathbf \theta} \left[-\sum_{i=1}^N \log P_\theta (y_i|\mathbf x_i) \right]
$$
这里 $P_{\mathbf \theta}(y|\mathbf x)$ 表示输入 $\mathbf x$ 给定条件下输出 $y$ 的条件概率，由神经网络决定：$\mathbf \theta$ 是神经网络的参数。

###### 2. 具体形式
针对不同的问题，前馈神经网络学习的一般形式可以转化为不同的具体形式。
当问题是**回归**时，模型的输入是实数向量 $\mathbf x$，输出是实数值 $y$。神经网络 $f(\mathbf x;\mathbf \theta)$ 决定输入给定条件下输出的条件概率分布 $P_{\mathbf \theta}(y|\mathbf x)$。假设条件概率分布 $P_{\mathbf \theta} (y|\mathbf x)$ 遵循高斯分布：
$$
P_{\mathbf \theta}(y|\mathbf x) \sim N (f(\mathbf x;\mathbf\theta),\sigma^2)
$$
其中，$y\in(-\infty,+\infty)$，$f(\mathbf x;\mathbf \theta)$ 是均值，$\sigma^2$ 是方差。学习问题 (极大似然估计) 变为优化问题：
$$
f_X(x;\mu,\sigma)=\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac 12\left(\frac{x-\mu}{\sigma} \right)^2} 
$$
$$\begin{aligned}
\hat \theta &= \mathop{\mathrm{argmin}}_{\theta} -\sum_{i=1}^N\log\left[\frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac 12 \left(\frac{y_i-f(\mathbf x_i ; \theta)}{\sigma} \right)^2}  \right]\\
&= \mathop{\mathrm{argmin}}_{\theta} \left[ \sum_{i=1}^N \frac{1}{2\sigma^2}(y_i - f(\mathbf x_i;\theta))^2 + \frac{N}{2}\log 2\pi + N\log \sigma\right]
\end{aligned}$$
假设方差 $\sigma^2$ 固定不变，有等价的优化问题：
$$
\hat \theta = \mathop{\mathrm{argmin}}_{\theta} \sum_{i=1}^N \frac12 (y_i - f(\mathbf x_i; \mathbf \theta))^2
$$
从另一个角度看，前馈神经网络用于回归时，使用平方损失(square loss) 作为损失函数， 学习进行的是平方损失的最小化。

当问题是**二类分类**时，模型的输入是实数向量 $\mathbf x$，输出是类别 $y\in \{0,1\}$，神经网络 $f(\mathbf x; \mathbf \theta)$ 决定输入给定条件下类别的条件概率分布：
$$
p = P_{\theta} (y=1|\mathbf x) = f(\mathbf x; \mathbf \theta)
$$
假设条件概率分布 $P_{\theta} (y=1|\mathbf x)$ 遵循贝努利分布，学习问题 (极大似然估计) 变为优化问题：
$$
\hat {\theta} = \mathop{\mathrm{argmin}}_{\theta} \left\{-\sum_{i=1}^N [y_i \log f(\mathbf x;\mathbf \theta) + (1-y_i) \log (1-f(\mathbf x; \mathbf \theta))] \right\}
$$
这时损失函数是交叉熵 (cross entropy) 损失。离散分布的交叉熵的一般定义是 $-\sum_{k=1}^l P_k \log Q_k$，表示经验分布和预测分布的差异，其中 $Q_k$ 是预测分布的概率，$P_k$ 是经验分布的概率。

当问题是 **多类分类** 时，模型的输入是实数向量 $\mathbf x$，输出类别是 $y_k \in \{0,1\},k=1,2,\cdots,l,\sum_{k=1}^ly_k=1$，神经网络 $f(\mathbf x;\mathbf \theta)$ 表示输入给定条件下类别的条件概率分布
$$
\mathbf p = P_\theta(y_k = 1|\mathbf x) = f(x;\mathbf \theta)
$$
假设条件概率分布 $P_{\mathbf \theta}(y_k=1|\mathbf x)$ 遵循类别分布 (categorical distribution)，学习问题变为优化问题：
$$
\hat {\mathbf \theta} = \mathop{\text{argmin}}_{\theta} \left\{-\sum_{i=1}^N \left[\sum_{k=1}^l y_{ik}\log f(\mathbf x;\mathbf \theta) \right] \right\}
$$
其中，$y_{ik} \in \{0,1\},\sum_{k=1}^l y_{ik}=1,k=1,2,\cdots,l,i=1,2,\cdots,N$。所以，前馈神经网络用语二类分类和多类分类时以交叉熵为损失函数，进行的是交叉熵的最小化。
#### 2.2 前馈神经网络学习的优化算法
###### 1. 非凸优化问题
前馈神经网络学习变成给定网络架构 $f(\mathbf x;\mathbf \theta)$，训练数据集 $\mathcal T$ 的条件下，最小化目标函数 $L(\mathbf \theta)$，得到最优参数 $\hat{\mathbf \theta}$ 的优化问题 (最小化问题)

前馈神经网络学习的目标函数一般是非凸函数，优化问题是非凸优化。从前馈神经网络的等 价性 可以得知，一个神经网路通常有大量等价的参数向量，所以其学习的优化问题有大量等价的局部最优点(最小点)。

![[Pasted image 20240714165429.png]]
图 示意了神经网络的非凸优化问题。参数向量是 $(\theta_1,\theta_2)$，目标函数是 $L(\theta_1,\theta_2)$，全局最小点是深蓝色，局部最小点是浅蓝色。因为目标函数非凸，有许多局部最小点。

###### 2. 梯度下降法和随机梯度下降法
深度学习包括 前馈神经网络学习，均使用迭代优化算法，包括梯度下降法(gradient descent)和随机梯度下降法 (stochastic gradient descent)，后者更为常用(附录A给出梯度下降法的一般介绍)。
优化目标函数写作
$$
L(\mathbf \theta) = \frac 1N \sum_{i=1}^N L_i (\theta) = \frac 1N \sum_{i=1}^N L(f(\mathbf x_i;\mathbf \theta),y_i)
$$
其中，$L_i(\mathbf \theta)$ 是第 $i$ 个样本的损失函数

**梯度下降法**首先随机初始化参数向量 $\mathbf \theta$；之后针对所有样本，通过以下公式更新参数向量 $\mathbf \theta$；不断迭代，直到收敛为止。
$$
\theta \leftarrow \theta - \eta \frac{\partial L(\theta)}{\partial \theta}
$$
或写作
$$
\theta \leftarrow \theta - \eta \frac 1N \sum_{i=1}^M \frac{\partial L_i(\theta)}{\partial \theta}
$$
其中，$\eta >0$ 是学习率，$\frac{\partial L(\theta)}{\partial \theta}$ 是所有样本的损失函数的梯度向量，$\frac{\partial L_i(\theta)}{\partial \theta}$ 是第 $i$ 个样本的损失函数的梯度向量。

梯度下降的基本想法如下。由于负梯度方向 $-\frac{\partial L(\theta)}{\partial \theta}$ 是使函数值下降的方向，所以每一次迭代以负梯度更新参数向量 $\theta$ 的值，从而达到减少函数值 $L(\theta)$ 的目的，函数极小值满足 $\nabla L(\theta)= \mathbf  0$。在迭代过程中，梯度向量趋近 $\mathbf 0$ 向量，参数向量 $\mathbf \theta$ 也趋近极小点。学习率控制参数更新的幅度。学习率的大小需要适当，学习率过小，参数向量每次更新的幅度会过小，迭代的次数会增加;学习率过大，参数向量每次更新的幅度会过大，产生振荡，迭代的次数也会增加。图23.18 显示梯度下降的过程。

![[Pasted image 20240714171717.png]]
**随机梯度下降法** 首先随机打乱样本顺序，将样本分成 $m$ 个组 (小批量)，每一组有 $n$ 个样本 (假设 $m=\lfloor N/n \rfloor$)；接着随机初始化参数向量；之后针对每组样本，通过以下公式更新参数向量，并遍历所有样本组；不断迭代，直到收敛为止。
$$
\theta \leftarrow \theta - \eta \frac 1n \sum_{j=1}^n \frac {\partial L_j(\theta)}{\partial \theta}
$$
其中，$\eta > 0$ 学习率，$\frac{\partial L_j(\theta)}{\partial \theta}$ 是一个组中的第 $j$ 个样本的损失函数的梯度向量。 算法23.2 给出随机梯度下降的具体算法。当 $n$ 是 $1$ 时，每次参数更新只使用一个样本，是一种特殊的随机梯度下降。当 $n$ 是整体样本容量 $N$ 时，随机梯度下降变为梯度下降 (当前深度学习采用的 Adam 等优化算法，在随机梯度下降的迭代过程中，自适应地调整梯度向量。第29章对 Adam 等算法予以介绍)。

![[Pasted image 20240714172853.png]]

随机梯度下降可以进行分布式并行计算，进一步提高学习效率，特别是当训练数据量大时非常有效。具体地，每组样本分配到不同的工作服务器 (worker)上，各台工作服务器基于自己的数据并行更新参数向量，参数服务器 (paramet ers erver) 再将所有工.作服务器的参数更新结果汇总求平均，得到一轮的训练结果。
#### 2.3 反向传播算法
 基于梯度下降或随机梯度下降的学习算法的核心是针对给定样本，计算损失函数对神经网络所有参数的梯度 $\frac{\partial L}{\partial \theta}$，更新神经网络的所有参数 $\theta$。反向传播(back propagation) 算法也称 误差反向传播(error back propagation)算法，提供了一个高效的梯度计算以及参数更新方法。只需要依照网络结构进行一次正向传播 (forward propagation) 和一次反向传播(backward propagation)，就可以完成梯度下降的一次迭代。在梯度下降的每一步，参数已在前一步更新，正向传播旨在基于当前的参数重新计算神经网络所有变量(比如，神经元的输出)，反向传播旨在基于当前的变量重新计算损失函数对所有参数的梯度，这样就可以根 据梯度下降公式(23. 40)和公式(23. 41)更新神经网络的所有参数。
 考虑一个 $s$ 层的神经网络，其中 $t$ 层 (隐层) 的神经元定义如下：
$$
h_j^{(t)} = a \left(z_j^{(t)}\right), j = 1,2,\cdots,m
$$
$$
z_j^{(t)} = \sum_{i=1}^n w_{ji}^{(t)}h_i^{(t-1)} + b_j^{(t)}
$$
第 $t+1$ 层 (隐层) 的神经元定义如下：
$$
h_k^{(t+1)} = a ()
$$
#### 2.4 在计算图上的实现
#### 2.5 算法的实现技巧

## 3 前馈神经网络学习的正则化

#### 3.1 深度学习中的正则化
#### 3.2 早停法
#### 3.3 暂退法
