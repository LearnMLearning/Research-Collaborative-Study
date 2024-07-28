Latent space: preserve the information of data within a low- dimensional space
潜在空间：在一个低维空间内很好的保存了数据的信息

在了解 VAE 之前，先了解 **[[Autoencoder (AE)]]**

打开了但是还没看的网页们

https://zh.wikipedia.org/wiki/变分自编码器

https://zhuanlan.zhihu.com/p/348498294

https://zhuanlan.zhihu.com/p/575984592

https://zhuanlan.zhihu.com/p/249296925

https://zhuanlan.zhihu.com/p/531763049

https://zhuanlan.zhihu.com/p/88750084

https://zhuanlan.zhihu.com/p/686158547

https://zhuanlan.zhihu.com/p/563543020

https://www.zhihu.com/question/649097976/answer/3444409568

https://www.reddit.com/r/MachineLearning/comments/1bmi8gq/d_is_vae_still_worth_it/?rdt=37283
## 0 前导知识 辅助理解
#### 目标：生成模型
VAE 是一个生成模型，我们的最终目标是如何构建一个生成模型。**生成模型其实就是一个联合概率分布 $P(X)$ 或 $P(X,Y)$**。当有了这个联合概率分布后，我们就可以从中采样生成不同的样本。

## 1 Introduction
#### 1.1 Motivation
生成建模有吸引力的原因有很多
	1. 表达物理定律和约束，同时表达我们不知道或不关心的细节
	2. 自然地表达了世界的因果关系。
VAE可以看作是两个耦合但独立的参数化模型:
	1. The encoder or recognition model
	2. The decoder or generative model
	1. 识别模型向生成模型提供其对**潜在随机变量的后验近似值**，它需要在“期望最大化”学习的迭代中更新其参数。
	2. 生成模型是识别模型学习数据的**有意义表示(可能包括类标签)** 的各种脚手架。根据贝叶斯规则，识别模型是生成模型的近似逆。
**VAE** comparing to Variational Inference (**VI**)
	**VI**: 其中每个数据案例都有单独的变分分布，这对于大型数据集来说效率很低。识别模型使用一组参数来模拟输入变量和潜在变量之间的关系，因此被称为“平摊推理 (amortized inference)”。这种识别模型可以是任意复杂的，但仍然相当快，因为通过构造，它可以使用从输入到潜在变量的单个前馈传递来完成。然而，我们付出的代价是，这种采样会在学习所需的梯度中产生**采样噪声**。
	**VAE:** VAE 框架的最大贡献是实现了我们可以通过使用现在所知的 "**重新参数化技巧** (reparameterization trick)" 来抵消这种方差。

VAE 受到 Helmholtz Machine (Dayan et al.， 1995)的启发，这可能是**第一个使用识别模型**的模型。然而，它的**唤醒-睡眠算法效率低下**，**没有优化单个目标**。相反，VAE学习规则遵循**对最大似然目标的单一近似**。

VAEs 将图形模型和深度学习结合在一起
	**生成模型**是形式为 $p(\mathbf x|\mathbf z)$ 的贝叶斯网络，或者有多个潜在层，则是 $p(\mathbf x|\mathbf z_L) p(\mathbf z_L|\mathbf z_{L-1}) \dots p(\mathbf z_1|\mathbf z_0)$.
	**识别模型**是形式为 $q(\mathbf z|\mathbf x)$ 的贝叶斯网络，或者是一个层次结构，例如 $q(\mathbf z_0|\mathbf z_1)\dots q(\mathbf z_L|\mathbf x)$.
但在每个条件中可能隐藏一个复杂的 (深度) 神经网络，例如 $\mathbf z|\mathbf x \sim f(\mathbf x, \mathbf \epsilon)$，$f$ 是一个神经网络映射，$\epsilon$ 是一个噪声随机变量。它的学习算法是经典的 (amortized, variational) 期望最大化的混合，但通过重参数化技巧，最终通过嵌入其中的深度神经网络的许多层**反向传播**。

VAE 向多方向拓展
	[[Dynamical Models]] 动态模型 (Johnson et al., 2016)
	[[Models with attention]] 注意力模型 (Gregor et al., 2015)
	[[Models with Multiple Levels of Stochastic Latent Variables]] 具有多层随机潜在变量的模型 (Kingma et al., 2016)
	**[[GAN]]** 生成对抗网络 (Goodfellow et al., 2014) 该种生成建模范式获得了极大的关注：VAEs 和 GANs 似乎有互补的特性：虽然GANs 可以生成**高主观感知质量**的图像，但与基于似然的生成模型相反。它们往往缺乏对数据的充分支持。与其他基于似然的模型一样，VAEs生成的样本更分散，但就似然准则而言，它是更好的密度模型。因此，已经提出了许多的混合模型，试图代表两个世界的最佳状态 (Dumoulin et al., 2017; Grover et al., 2018; Rosca et al., 2018)

作为一个社区，我们似乎已经接受了这样一个事实，即**生成模型**和**无监督学习**在构建智能机器方面发挥着重要作用。我们希望VAE能为这个难题提供一块有用的拼图。
#### 1.2 Aim
变分自编码器(VAEs)框架(Kingma and Welling, 2014;Rezende et al.， 2014)提供了一种利用**随机梯度下降**联合学习**深度潜变量模型**和相应推理模型的原则性方法。该框架具有广泛的应用，从**生成建模**、**半监督学习**到**表示学习**。

这项工作是我们早期工作的扩展版本(Kingma和Welling, 2014)，使我们能够更详细地解释这个主题，并讨论一系列重要的后续工作。这并非是对所有相关工作的全面审查。我们假定读者具有基本的**代数**、**微积分**和**概率论**知识。

在本章中，我们讨论了背景材料:**概率模型**，有向图模型，有向图模型与神经网络的结合，在完全观察模型和深潜变量模型(DLVMs)中的学习。在第2章中，我们将解释VAEs的基础知识。在第3章中，我们解释了高级推理技术，然后在第4章中解释了高级生成模型。有关数学符号的更多信息，请参阅A.1节。
#### 1.3 Probabilistic Models and Variational Inference
在机器学习领域，我们经常对从数据中学习各种自然和人工现象的概率模型感兴趣。概率模型是对这种现象的数学描述。它们对于理解这些现象、预测未来的未知以及各种形式的辅助或自动决策都很有用。因此，概率模型形式化了知识和技能的概念，是机器学习和人工智能领域的核心结构。

由于概率模型包含未知数，而数据很少描绘出未知的完整图景，我们通常需要在模型的各个方面假设某种程度的不确定性。这种不确定性的程度和性质是用(条件)概率分布来表示的。模型可以由连续值变量和离散值变量组成。在某种意义上，概率模型的最完整形式指定了模型中变量之间的所有相关性和高阶依赖性，以这些变量的**联合概率分布**的形式。

我们用 $\mathbf x$ 作为向量来表示所有观察到的变量的集合我们要对这些变量的联合分布进行建模。请注意，为了简化符号并避免混乱，我们使用小写粗体(例如x)来表示观察到的随机变量的底层集合，即平面化和连接，以便该集合表示为单个向量。有关符号的更多信息，请参见A.1节。

我们假设观察到的变量 $\mathbf x$ 是来自未知底层过程的随机样本，其真实(概率)分布 $p^∗(\mathbf x)$ 是未知的。我们尝试用一个选定的模型 $p_{\theta}(\mathbf x)$ 来近似这个潜在的过程，参数为 $\theta$:
$$
\mathbf x \sim p_\theta(\mathbf x)
$$
最常见的是，*Learning* 是寻找参数 $\theta$ 值的过程，使模型给出的概率分布函数 $p_\theta(\mathbf x)$ 近似于数据的真实分布，表示为 $p^*(\mathbf x)$，使得对于任何观测到的 $\mathbf x$:
$$
p_{\theta}(\mathbf x) \approx p^*(\mathbf x)
$$
自然，我们希望 $p_{\theta}(\mathbf x)$足够灵活，能够适应数据，这样我们就有机会获得一个足够精确的模型。同时，我们希望能够将关于数据分布的知识整合到已知的先验模型中。
###### 1.3.1 Conditional Models
通常，例如在分类或回归问题的情况下，我们对学习无条件模型 $p_{\theta}(\mathbf x)$ 不感兴趣，而是对近似潜在条件分布 $p^∗(\mathbf y|\mathbf x)$ 的条件模型 $p_{\theta}(\mathbf y|\mathbf x)$ 感兴趣:变量 $\mathbf y$ 值的分布，以观察到的变量 $\mathbf x$ 的值为条件。在这种情况下，$\mathbf x$ 通常被称为模型的输入。与无条件情况一样，选择一个模型 $p_{\theta}(\mathbf y|\mathbf x)$，并将其优化为接近未知的底层分布，这样对于任意 $\mathbf x$ 和 $\mathbf y$:
$$
p_{\mathbf \theta}(\mathbf y|\mathbf x) \approx p^*(\mathbf y | \mathbf x)
$$
条件建模的一个相对常见和简单的例子是图像分类，其中 $\mathbf x$ 是图像，$\mathbf y$ 是图像的类别，由我们希望预测的人标记。在这种情况下，$p_{\theta}(\mathbf y|\mathbf x)$ 通常被选择为分类分布，其参数由 $\mathbf x$ 计算。

当预测的变量是**非常高维**的，如图像、视频或声音时，条件模型变得更加难以学习。一个例子是图像分类问题的反面:预测图像上的分布，以**类标签**为条件。另一个同时具有**高维输入**和**高维输出**的例子是时间序列预测，如文本或视频预测。

为了避免符号混乱，我们通常会假设**无条件建模**，但应该始终记住，在这项工作中引入的方法在几乎所有情况下**都同样适用于条件模型**。模型所依赖的数据可以被视为模型的输入，类似于模型的参数，但**明显的区别**是，人们不会对它们的值进行优化。

#### 1.4 Parameterizing Conditional Distributions with Neural Networks
可微前馈神经网络，从这里就叫 *neural networks*，是一种特别灵活的计算可扩展的函数逼近器。基于具有多个“隐藏”人工神经元层的神经网络的模型学习通常被称为深度学习(Goodfellow et al.2016;LeCun et al.， 2015)。一个特别有趣的应用是概率模型，即在**概率模型**中使用神经网络的概率密度函数(PDFs)或概率质量函数(PMFs)。基于神经网络的概率模型在计算上是**可扩展**的，因为它们允许基于**随机梯度的优化**，正如我们将解释的那样，允许扩展到**大型模型**和**大型数据集**。我们将深度神经网络表示为一个向量函数: $\mathrm{NeuralNet}(\cdot)$。

在撰写本文时，深度学习已被证明可以很好地解决各种分类和回归问题，如(LeCun et al.， 2015;Goodfellow et al.， 2016)。以神经网络为基础的图像分类为例，LeCun et al.， 1998，神经网络参数化类标签 $y$ 上的分类分布 $p_{\theta}(y|\mathbf x)$，以图像 $\mathbf x$ 为条件。
$$\begin{aligned}
\mathbf p &= \mathrm{NeuralNet} (\mathbf x)\\
p_{\theta} (y|\mathbf x) &= \mathrm{Categorical} (y;\mathbf p)
\end{aligned}$$
其中 $\mathrm{NeuralNet}(\cdot)$的最后一次操作通常是一个 $\mathrm{softmax}()$ 函数，使得 $\sum_i p_i = 1$。
#### 1.5 Directed Graphical Models and Neural Networks
我们使用有向概率模型 (*directed* probabilistic models)，也称为 *probabilistic graphical models* (PGMs)，或贝叶斯网络 (*Bayesian networks*)。有向图模型是一种概率模型，其中所有的变量被拓扑组织成一个有向无环图。这些模型的变量的联合分布被分解为先验分布和条件分布的乘积:
$$
p_\theta(\mathbf x_1,\dots,\mathbf x_M) = \prod_{j=1}^M p_{\theta} (\mathbf x_j | Pa(\mathbf x_j))
$$
其中，$Pa(\mathbf x_j)$为有向图中节点 $j$ 的父变量集合。对于非根节点，我们以父节点为条件。对于根节点，父节点的集合是空集合，因此分布是无条件的。

传统上，每个条件概率分布 $p_{\theta}(\mathbf x_j |Pa(\mathbf x_j))$ 被参数化为查找表或线性模型 (Koller和 Friedman, 2009)。正如我们上面所解释的，更灵活的**参数化条件分布**的方法是使用神经网络。在这种情况下，神经网络将有向图中一个变量的父变量作为输入，并产生该变量的分布参数 $\mathbf \eta$:
$$\begin{aligned}
\eta &= \mathrm{NeuralNet} (Pa(\mathbf x))\\
p_{\theta} (\mathbf x| Pa(\mathbf x)) & = p_{\theta} ( \mathbf x | \eta)
\end{aligned}$$
现在我们将讨论如果在数据中观察到所有变量，如何**学习这些模型的参数**。

#### 1.6 Learning in Fully Observed Models with Neural Nets 用神经网络在完全观察模型中学习
如果在数据中观察到有向图模型中的所有变量，那么我们可以计算和微分模型下数据的对数概率，从而实现相对简单的优化。
###### 1.6.1 Dataset
我们通常收集一个由 $N\ge 1$ 个数据点组成的数据集 $\mathcal D$:
$$
\mathcal D = \{\mathbf x^{(1)},\mathbf x^{(2)},\dots, \mathbf x^{(N)} \} \equiv \{\mathbf x^{(i)}\}_{i=1}^N \equiv \mathbf x^{(1:N)}
$$
假设数据点是来自**不变的底层分布**的独立样本。换句话说，假设数据集由来自**同一(不变)系统**的**不同的、独立的**测量数据组成。在这种情况下，观测值 $\mathcal D = \{x^{(i)}\}_N^{i=1}$ 被称为 $i.i.d$ (independently and identically distributed)，表示 **独立和同分布**。在 $i.i.d$ 假设下，给定参数的数据点的概率被分解为单个数据点概率的乘积。因此，模型赋予数据的对数概率为:
$$
\log p_{\theta} (\mathcal D) = \sum_{\mathbf x \in \mathcal D} \log p_{\theta} (\mathbf x)
$$
###### 1.6.2 Maximum Likelihood and Minibatch SGD
概率模型最常用的准则是 *maximum log-likelihood* 最大对数似然(ML)。正如我们将解释的那样，对数似然准则的**最大化**相当于数据和模型分布之间的 Kullback Leibler (KL) 散度的**最小化**。


![[Pasted image 20240717135847.png]]

在ML准则下，我们试图找到参数 $θ$，使模型分配给数据的**对数概率**的**总和或平均值**最大化。对于大小为 $N_{\mathcal D}$ 的 $i.i.d$ 数据集 $\mathcal D$，最大似然目标是最大化式 $\log p_{\theta} (\mathcal D) = \sum_{\mathbf x \in \mathcal D} \log p_{\theta} (\mathbf x)$ 给出的对数概率。

利用微积分的**链式法则**和自动微分工具，我们可以有效地计算出该目标的梯度，即目标的一阶导数w.r.t. 其参数 $\theta$。我们可以使用这样的梯度迭代爬坡到 ML 目标的局部最优。如果我们使用所有数据点来计算这样的梯度 $\nabla_\theta \log p_\theta(\mathcal D)$ ，那么这就是所谓的**批量梯度下降** (*batch* gradient descent)。然而，对于大数据集大小$N_{\mathcal D}$，计算这个导数是一个昂贵的操作，因为它随 $N_{\mathcal D}$ 线性扩展。

一种更有效的优化方法是 **随机梯度下降(SGD)** (章节A.3)，它使用大小为$N_\mathcal M$的随机绘制的小批量数据$\mathcal M \subset \mathcal D$。有了这样的小批量，我们可以形成 ML 准则的**无偏估计** (指在大量重复抽样的情况下，估计量的值 (数学期望)等于未知参数的真值) 量:
$$
\frac{1}{N_{\mathcal D}} \log p_\theta (\mathcal D) \simeq \frac{1}{N_{\mathcal M}}\log p_{\theta} ( \mathcal M) = \frac{1}{N_{\mathcal M}} \sum_{\mathbf x \in \mathcal M} \log p_{\theta} (\mathbf x)
$$
这一标志意味着一方是另一方的无偏估计者 *unbiased estimator*。因此，一边(在本例中是右手边)是一个随机变量，由于某些噪声源，当**对噪声分布进行平均**时，两边是**相等**的。在这种情况下，噪声源是随机抽取的小批数据 $\mathcal M$ 。无偏估计量 $\log p_\theta(\mathcal M)$ 是可微的，得到无偏随机梯度:
$$
\frac{1}{N_\mathcal D} \nabla_{\theta} \log p_{\theta} (\mathcal D) \simeq \frac{1}{N_{\mathcal M}} \nabla_{\theta} \log p_{\theta} (\mathcal M) = \frac{1}{N_{\mathcal M}} \sum_{\mathbf x \in \mathcal M} \nabla_{\theta} \log p_{\theta} (\mathbf x)
$$
这些梯度可以插入到**随机梯度优化器**中;进一步讨论见A.3节。简而言之，我们可以通过在随机梯度的方向上重复采取小步骤来优化目标函数。
###### 1.6.3 Bayesian inference
从贝叶斯的角度来看，我们可以通过最大后验(MAP)估计来改进机器学习(见第 A.2.1 节)，或者更进一步，对参数的完整近似后验分布进行推断(见第 A.1.4 节)。

#### 1.7 Learning and Inference in Deep Latent Variable Models 深度潜在变量模型的学习与推理
###### 1.7.1 Latent Variables
我们可以将前一节讨论的完全观察到的**有向模型**扩展为**具有潜在变量**的**有向模型**。潜在变量是模型的一部分，但我们没有观察到，因此不是数据集的一部分。我们通常用 $\mathbf z$ 来表示这样的**潜在变量**。在对观测变量 $\mathbf x$ 进行无条件建模的情况下，有向图形模型将表示观测变量 $\mathbf x$ 和潜在变量 $\mathbf z$ 上的联合分布 $p_{\theta}(\mathbf x,\mathbf z)$。观测变量 $p_{\theta}(\mathbf x)$的 **边际分布** 由下式给出:
$$
p_{\theta} (\mathbf x) = \int p_\theta (\mathbf x, \mathbf z) \, \mathrm d \mathbf z
$$
当作为 $\theta$ 的函数时，这也称为(单数据点)边际似然 *marginal likelihood* 或模型证据 *model evidence*。
这种关于 $\mathbf x$ 的隐式分布是非常灵活的。如果 $\mathbf z$ 是离散的，且 $p_{\theta}(\mathbf x|\mathbf z)$ 是高斯分布，则 $p_{\theta}(\mathbf x)$ 是混合高斯分布。对于连续的 $\mathbf z$, $p_{\theta}(\mathbf x)$可以看作是一个无限的混合，它比离散的混合可能更强大。这种边际分布也称为**复合概率分布**。
###### 1.7.2 Deep Latent Variable Models
我们使用术语深潜变量模型 *deep latent variable model* (DLVM)来表示其分布由神经网络参数化的潜在变量模型 $p_{\theta}(\mathbf x,\mathbf z)$。这样的模型可以以某些上下文为条件，如$p_{\theta}(\mathbf x,\mathbf z|\mathbf y)$。DLVM 的一个重要优点是，即使有向模型中的**每个因素**(先验分布或条件分布)**相对简单**(例如条件高斯分布)，边际分布 $p_{\theta}(\mathbf x)$ 也可以非常复杂，即包含几乎任意的依赖关系。这种表达性使得深层潜变量模型对于近似复杂的底层分布 $p^*(\mathbf x)$具有吸引力。 

也许最简单、最常见的DLVM是一个被指定为分解的DLVM，它的结构如下:
$$
p_{\theta} (\mathbf x,\mathbf z) = p_{\theta} (\mathbf z) p_{\theta} (\mathbf x|\mathbf z)
$$
其中 $p_{\theta}(\mathbf z)$ 和/或 $p_{\theta}(\mathbf x|\mathbf z)$是指定的。分布 $p(\mathbf z)$ 通常被称为 $\mathbf z$ 上的先验分布，因为它不以任何观测值为条件。
###### 1.7.3 Example DLVM for multivariate Bernuolli data
(Kingma and Welling, 2014)中使用的二元数据 $\mathbf x$ 的简单示例DLVM具有球形高斯潜在空间和因式伯努利观测模型:
$$\begin{aligned}
p(\mathbf z) &= \mathcal N(\mathbf z;0,\mathbf I)\\
\mathbf p &= \mathrm{DecoderNeural Net}_{\theta}(\mathbf z)\\
\log p(\mathbf x | \mathbf z) &= \sum_{j=1}^D \log p (x_j | \mathbf z) = \sum_{j=1}^D \log \mathrm{Bernoulli} (x_j;p_j)\\
&= \sum_{j=1} x_j \log p_j + (1-x_j)\log (1-p_j)
\end{aligned}$$
其中$\forall p_j \in \mathbf p: 0\le p_j \le 1$ (例如通过 sigmoid 非线性作为DecoderNeuralNetθ(.)的最后一层来实现)，其中 $D$ 是 $\mathbf x$ 的维数，而伯努利 $(.;p)$ 为伯努利分布的概率质量函数(PMF)。
#### 1.8 Intractabilities 棘手之处
DLVM 中最大似然学习的**主要困难**是模型下数据的**边际概率通常难以处理**。这是由于计算边际似然(或模型证据)的方程 $p_{\theta} (\mathbf x) = \int p_{\theta} (\mathbf x, \mathbf z) \, d \mathbf z$ 中的积分，没有解析解或有效估计量。由于这种难处，我们不能区分它的**参数和优化**它，因为我们可以与充分观察模型。
$p_{\theta}(\mathbf x)$的难处理性与后验分布 $p_{\theta}(\mathbf z|\mathbf x)$ 的难处理性有关。注意，联合分布 $p_{\theta}(\mathbf x, \mathbf z)$ 的计算效率很高，密度是通过基本恒等式联系起来的:
$$
p_{\theta} (\mathbf z | \mathbf x) = \frac{p_{\theta} (\mathbf x,\mathbf z)}{p_{\theta} (\mathbf x)}
$$
 由于 $p_{\theta}(\mathbf x,\mathbf z)$ 是易于计算的，一个易于处理的边际似然 $p_{\theta}(\mathbf x)$ 导致一个易于处理的后验 $p_{\theta} (\mathbf z|\mathbf x)$，反之亦然。这两个问题在 DLVM 中都很难解决。
 
近似推理技术 (参见A.2节) 允许我们在 DLVM 中近似后验 $p_{\theta}(\mathbf z|\mathbf x)$ 和边际似然 $p_{\theta}(\mathbf x)$。传统的推理方法相对昂贵。例如，这种方法通常需要每个数据点的**优化循环**，或者产生不好的后验近似值。我们不想做这种昂贵的步骤。

同样，神经网络参数化的(有向模型) $p(\theta |\mathcal D)$ 的后验通常难以精确计算，需要近似推理技术。
## 2 Variational Autoencoders 变分自编码器
在本章中，我们将解释变分自编码器(VAEs)的基础知识。
![[Pasted image 20240717164849.png]]
图2.1:VAE 学习观察到的 $\mathbf x$ 空间(其经验分布 $q_{\mathcal D}(\mathbf x)$ 通常是复杂的)与潜在 $\mathbf z$ 空间(其分布可能相对简单(如球形，如图所示)之间的随机映射。**生成模型**学习一个**联合分布** $p_{\theta}(\mathbf x,\mathbf z)$，它通常(但并不总是)被分解为 $p_{\theta}(\mathbf x,\mathbf z) = p_{\theta}(\mathbf z)p_{\theta}(\mathbf x|\mathbf z)$，具有**潜在空间**上的先验分布 $p_{\theta}(\mathbf z)$ 和**随机解码器** $p_{\theta}(\mathbf x|\mathbf z)$。随机编码器 $q_{\phi}(\mathbf z|\mathbf x)$也称为推理模型 *Inference model* ，它近似于生成模型的真实但难以处理的后验 $p_{\phi}(\mathbf z|\mathbf x)$。
#### 2.1 Encoder or Approximate Posterior
在前一章中，我们介绍了深度潜变量模型(DLVM)，以及估计这种模型中的对数似然分布和后验分布的问题。变分自编码器(VAEs)框架提供了一种计算效率高的方法来优化 DLVM，并结合相应的推理模型使用SGD进行优化。

为了将 DLVM 的后验推理和学习问题转化为可处理的问题，我们引入了一个参数推理模型 *Inference Model* $q_{\phi}(\mathbf z|\mathbf x)$。这个模型也被称为**编码器** *Encoder* 或 识别模型 *Recoginition model* 。用 $\phi$ 表示该推理模型的参数，也称为**变分参数** *variational parameters*。我们优化变分参数 $\phi$ 使:
$$
q_{\phi} (\mathbf z|\mathbf x) \approx p_{\theta} (\mathbf z | \mathbf x)
$$
正如我们将解释的那样，这种对后验的近似帮助我们优化边际似然。

像DLVM一样，推理模型可以是(几乎)任何有向图形模型:
$$
q_{\phi} (\mathbf z | \mathbf x) = q_{\phi} (\mathbf z_1,\dots,\mathbf z_M |\mathbf x) = \prod_{j=1}^M q_{\phi} (\mathbf z_j| Pa(\mathbf z_j),\mathbf x)
$$
其中 $Pa (\mathbf z_j)$ 是变量 $\mathbf z_j$ 在有向图中的父变量集合。与 DLVM 类似，分布 $q_{\phi}(\mathbf z|\mathbf x)$ 可以使用深度神经网络参数化。在这种情况下，变分参数 $\phi$ 包括神经网络的权重和偏差。例如:
$$\begin{aligned}
(\mathbf \mu ,\log \mathbf \sigma) &= \mathrm{EncoderNeural Net}_{\phi} (\mathbf x)\\
q_{\phi} (\mathbf z|\mathbf x) & = \mathcal N(\mathbf z;\mathbf \mu ;\mathrm{diag}(\mathbf \sigma))
\end{aligned}$$
通常，我们使用单个编码器神经网络对数据集中的所有数据点执行**事后推理**。这可以与更**传统的变分推理方法形成对比**，其中变分参数不是共享的，而是每个数据点单独迭代优化的。在VAEs中使用**的跨数据点共享变分参数**的策略也称为**平摊变分推理** *amortized variational inference* (Gershman和Goodman, 2014)。通过平摊推理，我们可以避免每个数据点的优化循环，并利用SGD的效率。
#### 2.2 Evidence Lower Bound (ELBO)
与其他变分方法一样，变分自编码器的优化目标是**证据下界** *Evidence Lower Bound* ，简称为ELBO。这个目标的另一个术语是**变分下界** *variational lower bound*。典型地，ELBO是通过 Jensen 不等式推导出来的。这里，我们将使用另一种推导方法，避免使用 Jensen 不等式，从而更深入地了解它的**紧密性**。

对于任意选择推理模型 $q_{\phi}(\mathbf z|\mathbf x)$，包括变分参数 $\phi$ 的选择，有:
$$\begin{aligned}
\log p_{\theta} (\mathbf x) &= \mathbb E_{q_{\phi}(\mathbf z | \mathbf x)} \left[\log p_{\theta} (\mathbf x) \right]\\
&= \mathbb E_{q_{\phi}(\mathbf z |\mathbf x)}\left[\log \left[ \frac{p_{\theta}(\mathbf x,\mathbf z)}{p_{\theta} (\mathbf z|\mathbf x)}\right] \right]\\
&= \mathbb E_{q_{\phi}(\mathbf z | \mathbf x)}\left[\log \left[\frac{p_{\theta} (\mathbf x,\mathbf z)}{q_{\phi}(\mathbf z |\mathbf x)}\frac{q_{\phi} (\mathbf z|\mathbf x)}{p_{\theta}(\mathbf z |\mathbf x)} \right] \right]\\
&= \underbrace{\mathbb E_{q_\phi (\mathbf z |\mathbf x)} \left[\log\left[\frac{p_{\theta} ( \mathbf x,\mathbf z)}{q_{\phi}(\mathbf z | \mathbf x)}\right] \right]}_{=\mathop{\mathcal L_{\theta,\phi}(\mathbf x)}}+ \underbrace{\mathbb E_{q_{\phi}(\mathbf z |\mathbf x)} \left[\log \left[\frac{q_{\phi} (\mathbf z | \mathbf x)}{p_{\theta} ( \mathbf z |\mathbf x)} \right] \right]}_{=D_{KL}(q_\phi)(\mathbf z | \mathbf x)||p_{\mathbf \theta} (\mathbf z | \mathbf x)}
\end{aligned}$$
KL 散度 (Kullback-Leibler divergence)，非负
$$
D_{KL} (q_\phi (\mathbf z | \mathbf x) || p_{\theta} (\mathbf z | \mathbf x)) \ge 0
$$
取等条件为当且仅当 $q_{\phi(\mathbf z | \mathbf x)}$ 等于真实后验分布

第一个形式是 *variational lower bound* (*evidence lower bound*) (ELBO):
$$
\mathcal L_{\theta,\phi} (\mathbf x) = \mathbb E_{q_\phi(\mathbf z|\mathbf x)} [\log p_{\theta} (\mathbf x,\mathbf z) - \log q_{\phi} (\mathbf z|\mathbf x)]
$$
由于KL散度的非负性，ELBO是数据的对数似然的下界。
$$\begin{aligned}
\mathcal L_{\theta,\phi} (\mathbf x) &= \log p_{\theta} (\mathbf x) - D_{KL} (q_\phi(\mathbf z | \mathbf x) ||p_{\theta} (\mathbf z | \mathbf x))\\
& \le \log p_{\theta} (\mathbf x)
\end{aligned}$$
有趣的是，KL 散度 $D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z|\mathbf x))$ 决定了两个“距离”:
1. 根据定义，近似后验与真实后验的 KL 散度; 
2. ELBO $L_{\theta,\phi}(\mathbf x)$与边际似然 $\log p_\theta(\mathbf x)$ 之间的间隙; 这也被称为约束的紧密性。就 KL 散度而言，$q_{\phi}(\mathbf z|\mathbf x)$ 越接近真实(后验)分布 $p_{\theta}(\mathbf z|\mathbf x)$，则差距越小。

使用 Jenson 不等式：
$$\begin{aligned}
\log p_{\theta} (\mathbf x) &= \log \int  p_{\theta} (\mathbf x,\mathbf z) \,d \mathbf z\\
&= \log \int \frac{p_{\theta}(\mathbf x,\mathbf z)q_{\phi}(\mathbf z| \mathbf x)}{q_{\phi}(\mathbf z| \mathbf x)} \, d \mathbf z\\
&= \log \mathbb E_{q_{\phi}(\mathbf z |\mathbf x)}\left[ \frac{p_{\theta} (\mathbf x,\mathbf z)}{q_{\phi}(\mathbf z|\mathbf x)}\right]\\
&\ge  \mathbb E_{q_{\phi}(\mathbf z|\mathbf x) }\left[\log\left[\frac{p_{\theta} (\mathbf x,\mathbf z)}{q_{\phi}(\mathbf z|\mathbf x)} \right ]\right] (\mathrm{Jessen's \, inequality} ) = \mathcal L_{\theta,\phi}(\mathbf x)\, (\text{ELBO})\\
\end{aligned}$$
通过最大化这个下界，我们可以逼近对数似然 $\log p_{\theta} (\mathbf x)$
###### 2.2.1 Two for One
通过查看公式 $\mathcal L_{\theta,\phi}(\mathbf x) = \log p_{\theta} (\mathbf x) - D_{KL} (q_{\phi} (\mathbf z |\mathbf x)|| p_{\theta} (\mathbf z | \mathbf x))$，可以理解，最大化ELBO $L_{\theta,\phi}(\mathbf x)$ w.r.t.参数 $\theta$ 和 $\phi$，将同时优化我们关心的两件事:
1. 它将近似最大化边际似然 $p_{\theta}(\mathbf x)$。这意味着我们的生成模型将变得更好。 
2. 它将最小化近似 $q_{\phi}(\mathbf z|\mathbf x)$ 与真实后验 $p_{\theta}(\mathbf z|\mathbf x)$ 的KL散度，因此 $q_{\phi}(\mathbf z|\mathbf x)$ 变得更好。

#### 2.3 Stochastic Gradient-Based Optimization of the ELBO
ELBO 的一个重要性质是，它允许使用随机梯度下降(SGD)对所有参数($\phi$和 $\theta$)进行联合优化。我们可以从 $\phi$ 和 $\theta$ 的随机初始值开始，随机优化它们的值，直到收敛。
给定具有i.i.d数据的数据集，ELBO 目标是单个数据点 ELBO 的总和(或平均值):
$$
\mathcal L_{\theta,\phi} (\mathcal D) = \sum_{\mathbf x \in \mathcal D} \mathcal L_{\theta,\phi} ( \mathbf x)
$$
一般来说，单个数据点 ELBO 及其梯度 $\nabla_{\theta,\phi}\mathcal L_{\theta,\phi}(\mathbf x)$ 是难以处理的。然而，有良好的无偏估计量 $\nabla_{\theta,\phi}\mathcal L_{\theta,\phi}(\mathbf x)$存在，我们将会证明，这样我们仍然可以执行小批量 SGD。
生成模型参数 $\theta$ 下 ELBO 的无偏梯度很容易得到:
$$\begin{aligned}
\nabla_{\theta}\mathcal L_{\theta,\phi} (\mathbf x) &= \nabla_{\theta} \mathbb E_{q_\phi(\mathbf z | \mathbf x)} [\log p_{\theta} (\mathbf x,\mathbf z)-\log q_{\phi} (\mathbf z|\mathbf x)]\\
&= \mathbb E_{q_\phi (\mathbf z | \mathbf x)} [\nabla_{\theta} (\log p_\theta(\mathbf x,\mathbf z)-\log q_{\phi}(\mathbf z | \mathbf x))]\\
&\simeq \nabla_{\theta}(\log p_{\theta}(\mathbf x,\mathbf z) - \log q_{\phi}(\mathbf z | \mathbf x))\\
&= \nabla_{\theta} (\log p_{\theta}(\mathbf x , \mathbf z))
\end{aligned}$$
最后一行 (eq.(2.17))是第二行(eq.(2.15))的简单蒙特卡罗估计量 (Monte Carlo estimator) ，其中最后两行 (eq.(2.16) 和 eq.(2.17)) 中的 $\mathbf z$ 是来自 $q_{\phi}(\mathbf z| \mathbf x)$ 的随机样本。

变分参数 $\phi$ 的无偏梯度更难获得，因为 ELBO 的期望是在分布 $q_{\phi}(\mathbf z|\mathbf x)$ 的基础上取的，它是 $\phi$ 的函数。也就是说，一般来说:
$$\begin{aligned}
\nabla_{\phi} \mathcal L_{\theta,\phi} (\mathbf x) &= \nabla_{\phi} \mathbb E_{q_\phi(\mathbf z| \mathbf x)} [\log p_{\theta} (\mathbf x , \mathbf z) - \log q_\phi (\mathbf z|\mathbf x)]\\
&\ne \mathbb E_{q_\phi (\mathbf z | \mathbf x)}[\nabla_{\phi}(\log p_{\theta} (\mathbf x , \mathbf z) - \log q_\phi (\mathbf z|\mathbf x))]
\end{aligned}$$
在连续潜变量的情况下，我们可以使用 reparameterization trick 来计算 $\nabla_{\theta,\phi} \mathcal L_{\theta,\phi}(\mathbf x)$ 的无偏估计，我们现在将讨论。这种随机估计允许我们使用 SGD 来优化 ELBO;参见算法1。关于离散潜在变量的变分方法的讨论，请参见第2.9.1节。

ELBO的随机优化。由于噪声来源于小批量抽样和 $p(\epsilon)$ 抽样，这是一个双重随机优化过程。我们也把这个过程称为 *Auto-Encoding Variational Bayes* 自动编码变分贝叶斯(AEVB)算法。

![[Pasted image 20240718141732.png]]

#### 2.4 Reparameterization Trick 重参数化技巧
对于连续潜变量和可微编码器和生成模型，可以通过变量的变化直接对 ELBO进行w.r.t. $\phi$ 和 $\theta$ 的微分，也称为重参数化技巧 (Kingma和Welling, 2014和Rezende等人，2014)。
###### 2.4.1 Change of variables
首先，我们将随机变量 $\mathbf z \sim q_{\phi}(\mathbf z|\mathbf x)$ 表示为另一个随机变量 $\epsilon$ 的可微(可逆)变换，给定 $\mathbf z$ 和 $\phi$ :
$$
\mathbf z = \mathbf g(\mathbf \epsilon, \mathbf \phi,\mathbf x)
$$
$\epsilon$ 独立 $\phi,\mathbf x$ 分布
###### 2.4.2 Gradient of expectation under change of variable
给定这样的变量变化，期望可以用 $\epsilon$ 来表示，
$$
\mathbb E_{q_\phi (\mathbf z |\mathbf x)}[f(\mathbf z)] = \mathbb E_{p(\epsilon)} [f(\mathbf z)]
$$
其中 $\mathbf z = \mathbf g(\epsilon, \phi,\mathbf  x)$，期望算子与梯度算子互换，我们可以得到一个简单的蒙特卡罗估计量:
$$\begin{aligned}
\nabla_{\phi} \mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [f(\mathbf z)] &= \nabla_{\phi} \mathbb E_{p(\epsilon)} [f(\mathbf z)]\\
&= \mathbb E_{p(\epsilon)}[\nabla_{\phi}f(\mathbf z)]\\
&\simeq \nabla_{\phi} f(\mathbf z)
\end{aligned}$$
其中最后一行，$\mathbf z = \mathbf g(\phi, \mathbf x,\epsilon)$，随机噪声样本 $\epsilon \sim p(\epsilon)$。见图2.3 的说明和进一步的说明，图3.2 为二维玩具问题的后验结果的说明。
![[Pasted image 20240718143237.png]]
图2.3: 重新参数化技巧的说明。变分参数 $\phi$ 通过随机变量 $\mathbf z \sim q_{\phi}(\mathbf z|\mathbf x)$ 影响目标$f$。我们希望通过计算梯度 $\nabla_\phi f$ 来利用 SGD 优化目标。在原始形式(左)中，我们不能微分 $f$ w.r.t. $\phi$，因为我们不能直接通过随机变量 $\mathbf z$ 反向传播梯度。我们可以通过将变量重新参数化为 $\phi$， $\mathbf x$和新引入的随机变量 $\epsilon$ 的确定性可微函数来“外部化” $\mathbf z$ 的随机性。这允许我们“反向通过 backprop through $\mathbf z$”，并计算梯度 $\nabla_{\phi} f$。
###### 2.4.3 Gradient of ELBO
在重新参数化的情况下，我们可以用一个 w.r.t. $p (\epsilon)$ 代替期望w.r.t. $q_{\phi}(\mathbf z|\mathbf x)$。ELBO可以重写为:
$$\begin{aligned}
\mathcal L_{\theta,\phi} (\mathbf x) &= \mathbb E_{q_{\phi}(\mathbf z |\mathbf x)} [\log p_{\theta} (\mathbf x,\mathbf z) - \log q_{\phi} (\mathbf z | \mathbf x)]\\
&= \mathbb E_{p(\epsilon)} [\log p_{\theta} (\mathbf x,\mathbf z)-\log q_\phi (\mathbf z|\mathbf x)]
\end{aligned}$$
因此，我们可以形成单个数据点 ELBO 的简单蒙特卡罗估计量 $\tilde {\mathcal L}_{\theta,\phi}(\mathbf x)$，其中我们使用来自 $p(\epsilon)$ 的单个噪声样本 $\epsilon$:
$$\begin{aligned}
\epsilon &\sim p(\epsilon)\\
\mathbf z &= \mathbf g(\phi,\mathbf x,\epsilon)\\
\tilde {\mathcal L}_{\theta,\phi} ( \mathbf x)& = \log p_{\theta}(\mathbf x,\mathbf z) - \log q_{\phi} (\mathbf z | \mathbf x)
\end{aligned}$$
这一系列操作可以在TensorFlow等软件中表示为符号图，并且可以毫不费力地区分参数 $\theta$ 和 $\phi$。得到的梯度 $\nabla_\phi \tilde {\mathcal L}_{\theta,\phi} (\mathbf x)$ 用于使用小批量 SGD 优化 ELBO。参见算法1。该算法最初被 Kingma 和 Welling(2014) 称为 Auto-Encoding Variational Bayes (AEVB)算法。更一般地说，可重构 ELBO 估计被称为随机梯度变分贝叶斯(SGVB)估计。这个估计器也可以用来估计模型参数的后验，如(Kingma和Welling, 2014)的附录中所解释的那样。

**Unbiasedness 无偏性**
该梯度是精确单数据点 ELBO 梯度的无偏估计;当对噪声 $\epsilon \sim p(\epsilon)$ 进行平均时，该梯度等于单数据点 ELBO 梯度:
$$\begin{aligned}
\mathbb E_{p(\epsilon)} \left[\nabla_{\theta,\phi} \tilde{\mathcal L}_{\theta,\phi}(\mathbf x;\epsilon) \right] &= \mathbb E_{p(\epsilon)} [\nabla_{\theta,\phi} (\log p_{\theta} (\mathbf x,\mathbf z) - \log q_{\phi} (\mathbf z|\mathbf x))]\\
&= \nabla_{\theta,\phi} (\mathbb E_{p(\epsilon)}[\log p_{\theta} ( \mathbf x,\mathbf z) - \log q_{\phi} (\mathbf z | \mathbf x)])\\
&= \nabla_{\theta,\phi} \mathcal L_{\theta,\phi} (\mathbf x)
\end{aligned}$$
###### 2.4.4 Computation of $\log q_{\phi}(\mathbf z|\mathbf x)$
计算 ELBO 的(估计量)需要计算密度 $\log q_\phi(\mathbf z|\mathbf x)$，给定一个值 $\mathbf x$，并给定一个值$\mathbf z$ 或等效的 $\mathbf \epsilon$。这个对数密度是一个简单的计算，只要我们选择正确的变换 $\mathbf g()$。
注意，我们通常知道密度 $p(\epsilon)$，因为这是所选噪声分布的密度。只要 $\mathbf g(\cdot)$ 是可逆函数，则 $\epsilon$ 和 $\mathbf z$ 的密度关系式为:
$$
\log q_{\phi} (\mathbf z|\mathbf x)  = \log p(\epsilon) - \log d_{\phi} (\mathbf x,\mathbf \epsilon)
$$
其中第二项是雅可比矩阵$(\partial \mathbf z/\partial \epsilon)$ 的行列式绝对值的对数:
$$
\log d_{\phi} (\mathbf x,\mathbf \epsilon) = \log \left| \det \left(\frac{\partial \mathbf z}{\partial \mathbf \epsilon} \right)\right|
$$
我们称它为 $\epsilon$ 到 $\mathbf z$ 变换的对数行列式。我们使用符号 $\log d_{\phi}(\mathbf x, \epsilon)$ 来表明这个对数行列式，类似于 $\mathbf g()$，是 $\mathbf x$， $\epsilon$ 和 $\phi$ 的函数。雅可比矩阵包含 $\epsilon$ 到 $\mathbf z$ 变换的所有一阶导数:
$$
\frac{\partial \mathbf z}{\partial \epsilon} = \frac{\partial (z_1,\dots,z_k)}{\partial (\epsilon_1,\dots,\epsilon_k)} = \begin{pmatrix}
\frac{\partial z_1}{\partial \epsilon_1} & \cdots & \frac{\partial z_1}{\partial \epsilon_k}\\
\vdots & \ddots & \vdots\\
\frac{\partial z_k}{\partial \epsilon_1} & \cdots & \frac{\partial z_k}{\partial \epsilon_k}
\end{pmatrix}
$$
正如我们将展示的，我们可以构建非常灵活的转换 $\mathbf g()$，其中 $\log d_{\phi}(\mathbf x,\epsilon)$ 很容易计算，从而产生高度灵活的推理模型 $q_{\phi}(\mathbf z|\mathbf x)$。
#### 2.5 Factorized Gaussian posteriors 因式高斯后验
一个常见的选择是一个简单的 因式高斯编码器 factorized Gaussian encoder $q_{\phi}(\mathbf z|\mathbf x) = \mathcal N (\mathbf z; \mathbf \mu,\mathrm{diag}(\mathbf \sigma^2))$:
$$\begin{aligned}
(\mu,\log \sigma) &= \mathrm{EncoderNeuralNet}_{\phi}(\mathbf x)\\
q_{\phi}(\mathbf z| \mathbf x) & = \prod_i q_{\phi} (z_i |\mathbf x) = \prod_i \mathcal N(z_i;\mu_i;\sigma_i^2)
\end{aligned}$$
其中 $\mathcal N (z_i; \mu_i, \sigma_i^2)$ 为单变量高斯分布 (univariate Gaussian distribution) 的 PDF。重新参数化 (reparameterization) 后，我们可以写:
$$\begin{aligned}
\epsilon & \sim \mathcal N (0,\mathbf I)\\
(\mu,\log \sigma) &= \mathrm{EncoderNeuralNet}_\phi (\mathbf x)\\
\mathbf z &= \mathbf \mu + \sigma \odot \epsilon 
\end{aligned}$$
其中 $\odot$ 是元素积。$\mathbf \epsilon$ 到 $\mathbf z$ 变换的雅可比矩阵为:
$$
\frac{\partial \mathbf z}{\partial \epsilon} = \mathrm{diag} (\mathbf \sigma),
$$
即，一个元素 $\mathbf \sigma$ 在对角线上的对角矩阵。一个对角线 (或者更一般地说，三角形) 矩阵的行列式是它的对角线项的乘积。因此，雅可比矩阵的对数行列式是:
$$
\log d_{\phi}(\mathbf x,\mathbf \epsilon) = \log \left|\det \left(\frac{\partial \mathbf z}{\partial \epsilon} \right) \right| = \sum_{i}\log \sigma_i
$$
后验密度为:
$$\begin{aligned}
\log q_{\phi} (\mathbf z|\mathbf x) &= \log p(\epsilon)- \log d_{\phi}(\mathbf x,\epsilon)\\
&= \sum_i \log \mathcal N(\epsilon_i;0,1) - \log \sigma_i
\end{aligned}$$
当 $z = g(\epsilon,\phi, \mathbf x)$
###### 2.5.1 Full-covariance Gaussian posterior 全协方差高斯后验
**因式高斯后验** 可以推广为具有 **全协方差的高斯**:
$$
q_{\phi}(\mathbf z|\mathbf x) = \mathcal N (\mathbf z;\mu,\mathbf \Sigma)
$$
该分布的重新参数化由下式给出:
$$\begin{aligned}
\epsilon &\sim \mathcal N(0,\mathbf I)\\
\mathbf z &= \mu + \mathbf L \epsilon
\end{aligned}$$
其中 $\mathbf L$ 是一个下(或上)三角矩阵，在对角线上有非零结点。非对角线元素定义了 $\mathbf z$ 中元素的相关性(协方差)。

**全协方差高斯函数参数化**的原因是**雅可比行列式非常简单**。这种情况下的雅可比矩阵是平凡的: $\frac{\partial \mathbf z}{\partial \epsilon} = \mathbf L$。注意，三角形矩阵的行列式是其对角元素的乘积。因此，在此参数化中:
$$
\log \left|\det \left(\frac{\partial \mathbf z}{\partial \epsilon} \right) \right| = \sum_i \log |L_{ii}|
$$
后验的对数密度是
$$
\log q_{\phi}(\mathbf z|\mathbf x) = \log p(\mathbf \epsilon) - \sum_i \log |L_{ii}|
$$
此参数化对应于 $\mathbf z$ 协方差的 Cholesky 分解 $\mathbf \Sigma = \mathbf L \mathbf L^{\mathrm T}$:
$$\begin{aligned}
\mathbf \Sigma &= \mathbb E \left[(\mathbf z - \mathbb E[\mathbf z])(\mathbf z -\mathbb E[\mathbf z])^{\mathrm T} \right]\\
&= \mathbb E [\mathbf L \mathbf \epsilon (\mathbf L \epsilon)^{\mathrm{T}}] = \mathbf L \mathbb E[\mathbf \epsilon \mathbf \epsilon^{\mathrm T}] \mathbf L^{\mathrm T}\\
&= \mathbf L \mathbf L^{\mathrm T}
\end{aligned}$$
注意 $\mathbb E[\epsilon \epsilon^{\mathrm T}] = \mathbf I$ 因为 $\epsilon \sim \mathcal N(0,\mathbf I)$。

符号 $\epsilon \sim \mathcal N (0,\mathbf I)$ 表示随机变量 $\epsilon$ 服从均值为 0，协方差矩阵为单位矩阵 $\mathbf I$ 的多元正态分布。具体来说，这意味着：
1. **均值为0**：$\mathbb E[\epsilon] = 0$。
2. **协方差矩阵为单位矩阵 $\mathbf I$**
用数学语言来说，如果 $\epsilon=[\epsilon_1,\epsilon_2,\dots,\epsilon_n]^{\mathrm T}$ 是一个 $n$ 维向量，那么：
- $\mathbb E[\epsilon_i]=0$，表示每个变量 $\epsilon_i$ 的均值是 0.
- $\mathrm{Cov}(\epsilon_i,\epsilon_j) = \delta_{ij}$，其中$\delta_{ij}$是克罗内克 delta 函数，当 $i=j$ 时 $\delta_{ij}=1$，否则 $\delta_{ij}=0$。这表示 $\epsilon_i$ 和 $\epsilon_j$ 在 $i\ne j$ 时相互独立，并且每个分量的方差是 1。
![[Pasted image 20240719153424.png]]
![[Pasted image 20240719153433.png]]
![[Pasted image 20240719153730.png]]
![[Pasted image 20240719153747.png]]
构建矩阵 $\mathbf L$ 的一种方法是，它具有所需的性质，即三角形和非零对角线元素，是通过如下构造它:
$$
(\mathbf \mu, \log \mathbf \sigma,\mathbf L') \leftarrow \mathrm{EncoderNeuralNet}_{\phi}(\mathbf x)
$$
$$
\mathbf L \leftarrow \mathbf L_{mask} \odot \mathbf L' + \mathrm{diag} (\mathbf \sigma)
$$
然后按照上面的描述进行 $\mathbf z = \mathbf \mu + \mathbf L \mathbf \epsilon$。$\mathbf L_{mask}$ 是一个掩模矩阵，对角线上和上面都是 $0$，对角线下都是 $1$ 。注意，由于屏蔽 $\mathbf L$ ，雅可比矩阵 $(\partial \mathbf z/\partial \mathbf \epsilon)$ 是三角形的，其对角线上的值为σ。因此，对数行列式与因式高斯情况相同:
$$
\log \left |\det \left(\frac{\partial \mathbf z}{\partial \mathbf \epsilon} \right) \right| = \sum_{i} \log \sigma_i
$$
更一般地说，我们可以用一系列(**可微**的和**非线性**的)变换代替 $\mathbf z = \mathbf L\mathbf \epsilon + \mathbf \mu$;只要链中每一步的雅可比矩阵是三角形且对角线项不为零，则对数行列式保持简单。这一原理被 Kingma 等人于2016年探索并在第3章讨论的逆自回归流 *inverse autoregressive flow* (IAF)所使用。

用全协方差高斯推理模型和分解伯努利生成模型计算以VAE为例的单数据点ELBO的无偏估计。$\mathbf L_{mask}$是一个掩模矩阵，对角线上和上面都是0，对角线下都是1。
![[Pasted image 20240719174827.png]]
#### 2.6 Estimation of the Marginal Likelihood 边际似然估计
在训练了一个VAE之后，我们可以使用Rezende et al.， 2014最初提出的重要抽样 *important sampling* 技术来估计模型下数据的概率。数据点的边际似然可以表示为:
$$
\log p_{\theta} ( \mathbf x) = \log \mathbb E_{q_\phi (\mathbf z | \mathbf x)} [p_{\theta}(\mathbf x,\mathbf z) / q_{\phi}(\mathbf z|\mathbf x)]
$$
取 $q_{\phi}(\mathbf z|\mathbf x)$ 的随机样本，其蒙特卡罗估计量为:
$$
\log p_{\theta} (\mathbf x) \approx \frac 1 L \sum_{l=1}^L p_\theta (\mathbf x,\mathbf z^{(l)})/q_\phi (\mathbf z^{(l)}|\mathbf x)
$$
其中每个 $\mathbf z^{(l)} \sim q_{\phi}(\mathbf z|\mathbf x)$ 是来自推理模型的随机样本。通过使 $L$ 变大，近似值成为更好的边际似然估计，事实上，由于这是一个蒙特卡罗估计，对于 $L\rightarrow \infty$，它收敛于实际的边际似然。

注意，当设置 $L = 1$ 时，它等于 VAE 的 ELBO 估计。我们也可以使用 eq.(2.57) 的估计量作为我们的目标函数;这是重要性加权自编码器 *importance weighed autoencoders* (Burda et al.， 2015) (IWAE)中使用的目标。在那篇论文中，还表明目标随着 $L$ 值的增加而变得越来越紧密。后来由Cremer等人，2017表明，IWAE目标可以被重新解释为具有特定推理模型的 ELBO 目标。这些优化更紧密边界的方法的缺点是，**重要性加权估计对于高维潜在空间具有臭名昭著的糟糕缩放特性**。
#### 2.7 Marginal Likelihood and ELBO as KL Divergences 边际似然和 ELBO 作为 KL 散度
提高 ELBO 潜在紧密性的一种方法是**增加生成模型的灵活性**。这可以通过 ELBO 和 KL 散度之间的联系来理解。
对于大小为 $N_{\mathcal D}$ 的 i.i.d. 数据集 $\mathcal D$，最大似然准则为:
$$\begin{aligned}
\log p_{\theta} (\mathcal D) &= \frac{1}{N_{\mathcal D}} \sum_{\mathbf x \in \mathcal D} \log p_{\theta} (\mathbf x)\\
&= \mathbb E_{q_{\mathcal D}(\mathbf x)} [\log p_{\theta} (\mathbf x)]
\end{aligned}$$
式中，$q_{\mathcal D}(\mathbf x)$ 为经验(数据)分布，为**混合分布**:
$$
q_{\mathcal D} (\mathbf x) = \frac 1N \sum_{i=1}^{N} q_{\mathcal D}^{(i)}(\mathbf x)
$$
其中，对于连续数据，每个分量 $q_{\mathcal{D}}^{(i)}(\mathbf x)$ 通常对应于以值 $\mathbf x^{(i)}$为中心的 狄拉克 $δ$ Dirac delta 分布，对于离散数据，则对应于全概率质量集中在值 $\mathbf x^{(i)}$ 的离散分布。数据和模型分布之间的Kullback Leibler (KL)散度可以重写为负对数似然加上一个常数:
$$\begin{aligned}
D_{KL} (q_{\mathcal D} (\mathbf x) || p_{\theta} (\mathbf x)) &= -\mathbb E_{q_\mathcal D(\mathbf x)} [\log p_{\theta} (\mathbf x)] + \mathbb E_{q_\mathcal D (\mathbf x)} [\log q_{\mathcal D} (\mathbf x)]\\
&= -\log p_{\theta} (\mathcal D) + \mathrm{constant}
\end{aligned}$$
其中 $\mathrm{constant} = - \mathcal H(q_{\mathcal D}(\mathbf x))$。所以上面 KL 散度的最小化等价于数据 log-likelihood $\log p_\theta (\mathcal D)$的最大化。

将经验数据分布 $q_{\mathcal D}(\mathbf x)$ 与推理模型相结合，得到数据 $\mathbf x$ 与潜变量 $\mathbf z$ 的联合分布: $q_{\mathcal D,\phi}(\mathbf x, \mathbf z) = q_{\mathcal D}(\mathbf x)q(\mathbf z|\mathbf x)$。

$q_{\mathcal D,\phi} (\mathbf x, \mathbf z)$ 与 $p_{\theta}(\mathbf x,\mathbf z)$ 的 KL 散度可以写成负的 ELBO，加上一个常数:
$$\begin{aligned}
&D_{KL} (q_{\mathcal D,\phi}(\mathbf x,\mathbf z)||p_{\theta}(\mathbf x,\mathbf z))\\
&= -\mathbb E_{q_{\mathcal D}(\mathbf x)} \left[\mathbb E_{q_\phi(\mathbf z |\mathbf x)}[\log p_{\theta}(\mathbf x,\mathbf z) - \log q_{\phi}(\mathbf z |\mathbf x)]-\log q_{\mathcal D}(\mathbf x)] \right]\\
&= -\mathcal L_{\theta,\phi}(\mathcal D) + \mathrm{constant}
\end{aligned}$$
其中 $\mathrm{constant} =−\mathcal H(q_{\mathcal D}(\mathbf x))$。所以 ELBO 的最大化，等价于 KL 散度的最小化 $D_{KL}(q_{\mathcal D,\phi}(\mathbf x, \mathbf z)||p_{\theta}(\mathbf x, \mathbf z))$ ML 和 ELBO 目标之间的关系可以用以下简单的公式来概括:
$$\begin{aligned}
&D_{KL}(q_{\mathcal D,\phi}(\mathbf x,\mathbf z) || p_{\theta} (\mathbf x,\mathbf z))\\
&= D_{KL} (q_{\mathcal D}(\mathbf x) ||p_{\theta}(\mathbf x)) + \mathbb E_{q_{\mathcal D}(\mathbf x)} [D_{KL}(q_{\mathcal D,\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z|\mathbf x))]\\
&\ge D_{KL}(q_{\mathcal D}(\mathbf x)||p_{\theta}(\mathbf x))
\end{aligned}$$
另一个观点是，ELBO 可以被看作是增强空间中 *in an augmented space* 的最大似然目标。对于某些固定选择的编码器 $q_{\phi}(\mathbf z|\mathbf x)$，我们可以将联合分布 $p_{\theta}(\mathbf x,\mathbf z)$ 看作是原始数据 $\mathbf x$ 和与每个数据点相关的(随机)辅助特征 $\mathbf z$ 上的增量经验分布。然后，模型 $p_{\theta}(\mathbf x,\mathbf z)$ 定义了原始数据和辅助特征的联合模型。见图2.4。

![[Pasted image 20240719221113.png]]
图2.4:最大似然(ML)目标可以看作是 $D_{KL}(q_{\mathcal D},\phi(\mathbf x)||p_{\theta}(\mathbf x))$ 的最小化，而 ELBO 目标可以看作是$D_{KL}(q_{\mathcal D},\phi(\mathbf x,\mathbf z)||p_{\theta}(\mathbf x,\mathbf z))$ 的最小化，其上界是 $D_{KL}(q_\mathcal D,\phi(\mathbf x)||p_{\theta}(\mathbf x))$。如果不可能完美拟合，那么由于 KL 散度的方向，$p_{\theta} (\mathbf x,\mathbf z)$ 通常会比$q_{\mathcal D,\phi}(\mathbf x,\mathbf z)$具有更高的方差。
#### 2.8 Challenges
###### 2.8.1 Optimization issues
在我们的工作中，与(Bowman et al.，2015)和(Sønderby et al.， 2016a)的研究结果一致，我们发现具有未修改下界目标的随机优化可能陷入不希望的稳定平衡。在训练开始时，似然项$\log p (\mathbf x|\mathbf z)$ 相对较弱，因此初始吸引状态是 $q(\mathbf z|\mathbf x)\approx p(\mathbf z)$，从而导致难以逃脱的稳定平衡。在(Bowman et al.， 2015)和(Sønderby et al.， 2016a)中提出的解决方案是使用一个优化调度，其中潜在成本 $D_{KL}(q(\mathbf z|\mathbf x)||p(\mathbf z))$ 的权重在多个 epoch 中从 0 慢慢退火到 1

(Kingma et al.， 2016)中提出的另一种方法是自由比特 *free bits* 方法:对ELBO目标的修改，确保平均而言，每个潜在变量或每组潜在变量编码一定最小数量的信息比特。

潜在维度分为 $K$ 组。然后我们使用以下 minibatch 目标，它确保每个子集 $j$ (平均每个minibatch $\mathcal M$)使用少于 $\lambda$ nats的信息是不利的:
$$
\tilde {\mathcal L} \lambda = \mathbb E_{\mathbf x\sim \mathcal M}\left[\mathbb E_{q(\mathbf z|\mathbf x)}\left[\log p(\mathbf x|\mathbf z) \right] \right] - \sum_{j=1}^K \mathrm{maximum}(\lambda,\mathbb E_{\mathbf x\sim \mathcal M}[D_{KL}(q(z_j|\mathbf x)||p(z_j))])
$$
由于增加潜在信息通常有利于物镜的第一项(不受影响)(通常称为负重构误差 *negative reconstruction error*)，因此在实践中，对于所有 $j$，这导致 $\mathbb E_{\mathbf x \sim \mathcal M} [D_{KL}(q(\mathbf z_j|\mathbf x)| p(\mathbf z_j))]\ge \lambda$。Kingma等人在2016年发现，该方法在相当大的范围内( $\lambda \in [0.125,0.25,0.5,1,2]$)都能很好地工作，从而显著提高了基准结果的对数似然。
###### 2.8.2 Blurriness of generative model 生成模型的模糊性


#### 2.9 Related prior and concurrent work


###### 2.9.1 Score function estimator


## 3 Beyond Gaussian Posteriors 超越高斯后验
#### 3.1 Requirements for Computational Tractability 
#### 3.2 Improving the Flexibility of Inference Models 
###### 3.2.1 Auxiliary Latent Variables
###### 3.2.2 Normalizing Flows

#### 3.3 Inverse Autoregressive Transformations 
#### 3.4 Inverse Autoregressive Flow (IAF) 

#### 3.5 Related Work
## 4 Deeper Generative Models
#### 4.1 Inference and Learning with Multiple latent Variables
###### 4.1.1 Choice of ordering
#### 4.2 Alternative methods for increasing expressivity
#### 4.3 Autoregressive Models
#### 4.4 Invertible transformations with tractable Jacobian determinant
#### 4.5 Follow-Up Work
###### 4.5.1 Representation Learning
###### 4.5.2 Understanding of data, and artificial creativity
**Chemical Design**

**Natural Language Synthesis**

**Astronomy**

**Image (Re-)Synthesis**

###### 4.5.3 Other relevant follow-up work
## 5 Conclusion
有向概率模型是现代人工智能的一个重要方面。通过用可微的深度神经网络参数化条件分布，这种模型可以变得非常灵活。

在完全观测的情况下，向**最大似然目标优化**这些模型是直截了当的。然而，人们通常对**具有潜在变量**的灵活模型更感兴趣，例如**深度潜在变量模型**或**具有随机参数的贝叶斯模型**。在这两种情况下，都需要**执行近似后验估计**，其中 **变分推理(VI)** 方法是合适的。在VI中，**推理**被视为对**新引入的变分参数的优化**问题，通常针对ELBO，**模型证据的下界**或数据的**边际似然**进行优化。现有的后验推理方法要么效率相对较低，要么不适用于以神经网络为组成部分的模型。我们的主要贡献是一个**高效和可扩展**的**基于梯度**的**变分后验推理**和**近似最大似然**学习的**框架**。

本文描述了变分自编码器(VAE)及其一些扩展。VAE是具有连续潜变量的深度潜变量模型(DLVM)和关联推理模型的组合。DLVM是一种基于数据的生成模型。推理模型，也称为编码器或识别模型，近似生成模型的潜在变量的后验分布。生成模型和推理模型都是由深度神经网络全部或部分参数化的有向图形模型。通过对证据下界(ELBO)进行随机梯度上升，对模型参数(包括神经网络的权重和偏置等参数)进行联合优化。ELBO是数据的边际似然的下界，也称为变分下界。随机梯度，必要的执行SGD，通过一个基本的重参数化技巧获得。VAE框架现在是用于各种概率建模和人工创造力应用的常用工具，并且在大多数主要的深度学习软件库中都有基本的实现。

为了学习灵活的推理模型，我们提出了逆自渐进流(IAF)，这是一种允许缩放到高维潜在空间的归一化流。进一步探索的一个有趣方向是与具有计算廉价逆的变换进行比较，例如NICE (Dinh等人，2014)和Real NVP (Dinh等人，2016)。在VAE框架中应用这样的转换可以潜在地导致相对简单的VAE，并结合了强大的后验、先验和解码器。这样的架构可以潜在地与纯粹的自回归架构竞争或超越(Van den Oord等人，2016)，同时允许更快的合成。

所提出的VAE框架仍然是文献中唯一允许离散和连续观察变量的框架，允许有效的平摊潜在变量推理和快速合成，并且可以在数据的对数似然方面产生接近最先进的性能。
## Appendix
#### A.1 Notation and definitions
###### A.1.1 Notation

| Example(s)                                            | Description                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathbf x,\mathbf y,\mathbf z$                       | With characters in bold we typically denote random *vectors*. We also use this notation for collections of random variables variables.                                                                                                                                                                                                                      |
| $x,y,z$                                               | With characters in italic we typically denote random *scalars*, i.e. single real-valued numbers.                                                                                                                                                                                                                                                            |
| $\mathbf X,\mathbf Y,\mathbf Z$                       | With bold and capitalized letters we typically denote random *matrices*.                                                                                                                                                                                                                                                                                    |
| $Pa(\mathbf z)$                                       | The parents of random variable $\mathbf z$ in a directed graph.                                                                                                                                                                                                                                                                                             |
| $\mathrm{diag}(\mathbf x)$                            | Diagonal matrix, with the values of vector $\mathbf x$ on the diagonal.                                                                                                                                                                                                                                                                                     |
| $\mathbf x \odot \mathbf y$                           | Element-wise multiplication of two vectors. The resulting vector is $(x_1y_1,\dots,x_Ky_K)^{\mathrm T}$.                                                                                                                                                                                                                                                    |
| $\theta$                                              | Parameters of a (generative) model are typically denoted with the Greek lowercase letter $\theta$ (theta).                                                                                                                                                                                                                                                  |
| $\phi$                                                | Variational parameters are typically denoted with the bold Greek letter $\phi$ (phi).                                                                                                                                                                                                                                                                       |
| $p(\mathbf x),p(\mathbf z)$                           | Probability density functions (PDFs) and probability mass functions (PMFs), also simply called *distributions*, are denoted by $p(\cdot)$, $q(\cdot)$ or $r(\cdot)$.                                                                                                                                                                                        |
| $p(\mathbf x,\mathbf y,\mathbf z)$                    | Joint distributions are denoted by $p(\cdot , \cdot)$                                                                                                                                                                                                                                                                                                       |
| $p(\mathbf x$\|$\mathbf z)$                           | Conditional distributions are denoted by $p(\cdot$\|$\cdot)$                                                                                                                                                                                                                                                                                                |
| $p(\cdot;\theta),p_{\theta}(\mathbf x)$               | The parameters of a distribution are denoted with $p(\cdot ;\theta)$ or equivalently with subscript $p_{\theta}(\cdot)$.                                                                                                                                                                                                                                    |
| $p(\mathbf x = \mathbf a),p(\mathbf x \le \mathbf a)$ | We may use an (in-)equality sign within a probability distribution to distinguish between function arguments and value at which to evaluate. So $p(\mathbf x = \mathbf a)$ denotes a PDF or PMF over variable $\mathbf x$ evaluated at the value of variable a. Likewise, $p(\mathbf x \le \mathbf a)$ denotes a CDF evaluated at the value of $\mathbf a$. |
| $p(\cdot),q(\cdot)$                                   | We use different letters to refer to different probabilistic models, such as $p(\cdot)$ or $q(\cdot)$. Conversely, we use the *same* letter across different marginals / conditionals to indicate they relate to the same probabilistic model.                                                                                                              |

###### A.1.2 Definitions
| Term                                   | Description                                                                                                                        |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Probability density func- tion (PDF)   | A function that assigns a probability *density* to each possible value of given *continuous* random variables.                     |
| Cumulative distribution function (CDF) | A function that assigns a cumulative probability density to each possible value of given univariate *continuous* random variables. |
| Probability mass function (PMF)        | A function that assigns a probability *mass* to given *discrete* random variable.                                                  |

###### A.1.3 Distributions
We overload the notation of distributions (e.g. $p(x) = \mathcal N (\mathbf x; \mathbf \mu, \mathbf \Sigma)$) with two meanings: (1) a distribution from which we can sample, and (2) the probability density function (PDF) of that distribution.

| Term                                                                                                      | Description                                                                               |
| --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| $\mathrm{Categorical}(x;\mathbf p)$                                                                       | Categorical distribution, with parameter $\mathbf p$ such that $\sum_i p_i = 1$.          |
| $\mathrm{Bernuolli}(\mathbf x;\mathbf p)$                                                                 | Multivariate distribution of independent Bernoulli.                                       |
| $\mathrm{Normal}(\mathbf x;\mathbf \mu, \mathbf \Sigma)=\mathcal N(\mathbf x;\mathbf \mu,\mathbf \Sigma)$ | Multivariate Normal distribution with mean $\mathbf \mu$ and covariance $\mathbf \Sigma$. |
**Chain Rule of Probability**
$$
p(\mathbf a,\mathbf b) = p(\mathbf a) p(\mathbf b|\mathbf a)
$$
**Bayes' Rule**
$$
p(\mathbf a|\mathbf b) = \frac{p(\mathbf b|\mathbf a )p(\mathbf a)}{p(\mathbf b)}
$$

###### A.1.4 Bayesian Inference
Let $p(\theta)$ be a chosen marginal distribution over its parameters $\theta$, called a prior distribution. Let $\mathcal D$ be observed data, $p(\mathcal D|\theta) \equiv p_{\theta}(\mathcal D)$ be the probability assigned to the data under the model with parameters $\theta$. Recall the chain rule in probability:
$$
p(\theta,\mathcal D) = p(\theta|\mathcal D) p(\mathcal D) = p(\theta) p(\mathcal D|\theta)
$$
Simply re-arranging terms above, the posterior distribution over the parameters $\theta$, taking into account the data $\mathcal D$, is:
$$
p(\theta|\mathcal D) = \frac{p(\mathcal D | \theta)p(\theta)}{p(\mathcal D)} \varpropto p(\mathcal D|\theta)p(\theta)
$$
where the proportionality ($\varpropto$) holds since $p(\mathcal D)$ is a constant that is not dependent on parameters $\theta$. The formula above is known as Bayes’ rule, a fundamental formula in machine learning and statistics, and is of special importance to this work.

A principal application of Bayes’ rule is that it allows us to make predictions about future data $\mathbf x'$, that are optimal as long as the prior $p(\theta)$ and model class $p_{\theta}(\mathbf x)$ are correct:
$$
p(\mathbf x = \mathbf x' | \mathcal D) = \int p_{\theta} (\mathbf x = \mathbf x') p (\theta | \mathcal D)\, d \theta
$$
#### A.2 Alternative methods for learning in DLVMs
###### A.2.1 Maximum A Posteriori
从贝叶斯的角度来看，我们可以通过最大后验 (MAP) 估计来改进最大似然目标，从而最大化对数后验w.r.t $\theta$。使用i.i.d data $\mathcal D$，这是:
$$\begin{aligned}
L^{MAP} (\theta) &= \log p (\theta|\mathcal D)\\
&= \log p(\theta) + L^{ML}(\theta) + \text{constant}
\end{aligned}$$
方程(A.5)中的先验 $p(\theta)$ 对于越来越大的 $N$ 的影响是递减的。因此，在对大型数据集进行优化的情况下，我们经常选择简单地使用最大似然准则，从目标中省略先验，这在数值上相当于设置$p(\theta) = \text{constant}$。
###### A.2.2 Variational EM with local variational parameters
期望最大化(EM)是在部分观测模型中学习参数的一般策略(Dempster et al.， 1977)。有关使用MCMC的 EM 的讨论，请参见 A.2.3 节。这种方法可以解释为 ELBO 上的坐标上升(Neal and Hinton, 1998)。对于 i.i.d 数据，传统的变分EM方法估计**局部变分参数** $\mathbf \phi^{(i)}$，即数据集中每个数据点 $i$ 单独的一组变分参数。相反，VAE采用具有**全局变分参数**的策略。

EM从一些(随机的)初始选择 $\mathbf \theta$ 和 $\mathbf \phi^{(1:N)}$ 开始。然后迭代地应用更新:$$\begin{aligned}
\forall i &= 1, \dots,N : \phi^{(i)} \leftarrow \mathop{\mathrm{argmax}}_{\phi} \,\mathcal L(\mathbf x^{(i)};\mathbf \theta,\mathbf \phi)\\
\mathbf \theta & \leftarrow \mathop{\mathrm{argmax}}_{\theta} \sum_{i=1}^N \mathcal L(\mathbf x^{(i)};\mathbf \theta,\mathbf \phi)
\end{aligned}$$
直至收敛。为什么是这样？注意在 E-step:
$$
\begin{aligned}
&\mathop{\mathrm{argmax}}_{\phi} \, \mathcal L(\mathbf x;\mathbf \theta,\mathbf \phi)\\
&= \mathop{\mathrm{argmax}}_{\phi} [\log p_{\theta} (\mathbf x) - D_{KL} (q_{\phi}(\mathbf z | \mathbf x)||p_{\theta} ( \mathbf z | \mathbf x))]\\
&= \mathop{\mathrm{argmin}}_{\phi} \, D_{KL} (q_{\phi} 
(\mathbf z | \mathbf x) ||p_{\mathbf \theta}(\mathbf z |\mathbf x))
\end{aligned}
$$
因此，E 步明显地最小化了 $q_{\phi}(\mathbf z|\mathbf x)$ 与真实后验的KL散度。

其次，请注意，如果 $q_{\phi}(\mathbf z|\mathbf x)$ 等于 $p_{\theta}(\mathbf z|\mathbf x)$，则 ELBO 等于边际似然，但对于 $q_{\phi}(\mathbf z|\mathbf x)$ 的任何选择，M步 优化边际似然的边界。该界的紧性由 $D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_\theta(\mathbf z|\mathbf x))$ 定义。


###### A.2.3 MCMC-EM


#### A.3 [[Stochastic Gradient Descent]] 随机梯度下降
我们使用有向模型，其中每个数据点的目标是标量，并且由于组成它们的神经网络的可微性，目标是可微的，其参数 $\theta$。由于逆模自动微分(也称为**反向传播算法**(Rumelhart et al.， 1988))的显著效率，可微分标量目标的值和梯度(即偏导数向量)可以以相等的时间复杂度计算。在SGD中，我们迭代更新参数 $θ$:
$$
\theta_{t+1} \leftarrow \theta_t + \alpha_t \cdot \nabla_\theta \tilde L (\theta,\xi) 
$$
其中 $\alpha_t$ 是学习率或预条件，$\tilde L(\theta，\xi)$ 是目标 $L(\theta)$ 的无偏估计，即 $\mathbb E_{\xi \sim p(\xi)} \tilde L(\theta,\xi) = L(\theta)$。随机变量 $\xi$ 可以是一个数据点索引，从 $\{1,\cdots,N\}$，但也可以包含不同类型的噪声，例如 VAE 中的后验抽样噪声。在实验中，我们通常使用 Adam 和 Adamax 优化方法来选择 $\alpha_t$ (Kingma和Ba, 2015);这些方法不受目标的不断重新缩放和单个梯度的不断重新缩放的影响。因此，$L (\theta, \xi)$ 只需要达到比例的无偏性。我们迭代地应用 eq. (a .15)，直到满足停止条件。一个简单而有效的准则是，当一组数据出现的概率开始下降时，立即停止优化;这一标准被称为**早停** (*early stopping*)。


## Related Works
[[24-Feature Unlearning for Pre-trained GANs and VAEs]]


inference

variational inference (VI)

Auto encoder

variational autoencoder