## 1 Introduction
#### 1.1 Motivation
Many reasons why generative modeling is attractive
	1. Express physical laws and constraints while the details that we don't know or care about
	2. Naturally expresses **causal relations** of the world.
The VAE can be viewed as two coupled, but independently parameterized models:
	1. The encoder or recognition model
	2. The decoder or generative model
	1. 识别模型向生成模型提供其对**潜在随机变量的后验近似值**，它需要在“期望最大化”学习的迭代中更新其参数。
	2. 生成模型是识别模型学习数据的**有意义表示(可能包括类标签)** 的各种脚手架。根据贝叶斯规则，识别模型是生成模型的近似逆。
**VAE** comparing to Variational Inference (**VI**)
	**VI**: 其中每个数据案例都有单独的变分分布，这对于大型数据集来说效率很低。识别模型使用一组参数来模拟输入变量和潜在变量之间的关系，因此被称为“平摊推理 (amortized inference)”。这种识别模型可以是任意复杂的，但仍然相当快，因为通过构造，它可以使用从输入到潜在变量的单个前馈传递来完成。然而，我们付出的代价是，这种采样会在学习所需的梯度中产生**采样噪声**。
	**VAE:** VAE 框架的最大贡献是实现了我们可以通过使用现在所知的 "**重新参数化技巧** (reparameterization trick)" 来抵消这种方差。

VAE受到Helmholtz Machine (Dayan et al.， 1995)的启发，这可能是**第一个使用识别模型**的模型。然而，它的**唤醒-睡眠算法效率低下**，**没有优化单个目标**。相反，VAE学习规则遵循**对最大似然目标的单一近似**。

VAEs 将图形模型和深度学习结合在一起
	**生成模型**是形式为 $p(\mathbf x|\mathbf z)$ 的贝叶斯网络，或者有多个潜在层，则是 $p(\mathbf x|\mathbf z_L) p(\mathbf z_L|\mathbf z_{L-1}) \dots p(\mathbf z_1|\mathbf z_0)$.
	**识别模型**是形式为 $q(\mathbf z|\mathbf x)$ 的贝叶斯网络，或者是一个层次结构，例如 $q(\mathbf z_0|\mathbf z_1)\dots q(\mathbf z_L|\mathbf x)$.
但在每个条件中可能隐藏一个复杂的 (深度) 神经网络，例如 $\mathbf z|\mathbf x \sim f(\mathbf x, \mathbf \epsilon)$，$f$ 是一个神经网络映射，$\epsilon$ 是一个噪声随机变量。它的学习算法是经典的 (amortized, variational) 期望最大化的混合，但通过重参数化技巧，最终通过嵌入其中的深度神经网络的许多层**反向传播**。

VAE 向多方向拓展
	[[Dynamical Models]] 动态模型 (Johnson et al., 2016)
	[[Models with attention]] 注意力模型 (Gregor et al., 2015)
	[[Models with Multiple Levels of Stochastic Latent Variables]] 具有多层随机潜在变量的模型 (Kingma et al., 2016)
	**[[GAN]]** 生成对抗网络 (Goodfellow et al., 2014) 该种生成建模范式获得了极大的关注：VAEs 和 GANs 似乎有互补的特性：虽然GANs 可以生成高主观感知质量的图像，但与基于似然的生成模型相反。它们往往缺乏对数据的充分支持。与其他基于似然的模型一样，VAEs生成的样本更分散，但就似然准则而言，它是更好的密度模型。因此，已经提出了许多的混合模型，试图代表两个世界的最佳状态 (Dumoulin et al., 2017; Grover et al., 2018; Rosca et al., 2018)

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
可微前馈神经网络，从这里就叫 *neural networks*，是一种特别灵活的计算可扩展的函数逼近器。基于具有多个“隐藏”人工神经元层的神经网络的模型学习通常被称为深度学习(Goodfellow et al.2016;LeCun et al.， 2015)。一个特别有趣的应用是概率模型，即在概率模型中使用神经网络的概率密度函数(PDFs)或概率质量函数(PMFs)。基于神经网络的概率模型在计算上是可扩展的，因为它们允许基于随机梯度的优化，正如我们将解释的那样，允许扩展到大型模型和大型数据集。我们将深度神经网络表示为一个向量函数: $\mathrm{NeuralNet}(\cdot)$。

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

#### 1.6 Learning in Fully Observed Models with Neural Nets
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

一种更有效的优化方法是 **随机梯度下降(SGD)** (章节A.3)，它使用大小为$N_\mathcal M$的随机绘制的小批量数据$\mathcal M \subset \mathcal D$。有了这样的小批量，我们可以形成 ML 准则的无偏估计量:
$$
\frac{1}{N_{\mathcal D}} \log p_\theta (\mathcal D) \simeq \frac{1}{N_{\mathcal M}}\log p_{\theta} ( \mathcal M) = \frac{1}{N_{\mathcal M}} \sum_{\mathbf x \in \mathcal M} \log p_{\theta} (\mathbf x)
$$
这一标志意味着一方是另一方的无偏估计者 *unbiased estimator*。因此，一边(在本例中是右手边)是一个随机变量，由于某些噪声源，当**对噪声分布进行平均**时，两边是**相等**的。在这种情况下，噪声源是随机抽取的小批数据 $\mathcal M$ 。无偏估计量 $\log p_\theta(\mathcal M)$ 是可微的，得到无偏随机梯度:
$$
\frac{1}{N_\mathcal D} \nabla_{\theta} \log p_{\theta} (\mathcal D) \simeq \frac{1}{N_{\mathcal M}} \nabla_{\theta} \log p_{\theta} (\mathcal M) = \frac{1}{N_{\mathcal M}} \sum_{\mathbf x \in \mathcal M} \nabla_{\theta} \log p_{\theta} (\mathbf x)
$$
这些梯度可以插入到随机梯度优化器中;进一步讨论见A.3节。简而言之，我们可以通过在随机梯度的方向上重复采取小步骤来优化目标函数。
###### 1.6.3 Bayesian inference
从贝叶斯的角度来看，我们可以通过最大后验(MAP)估计来改进机器学习(见第 A.2.1 节)，或者更进一步，对参数的完整近似后验分布进行推断(见第 A.1.4 节)。

#### 1.7 Learning and Inference in Deep Latent Variable Models
###### 1.7.1 Latent Variables
我们可以将前一节讨论的完全观察到的有向模型扩展为具有潜在变量的有向模型。潜在变量是模型的一部分，但我们没有观察到，因此不是数据集的一部分。我们通常用z来表示这样的潜在变量。在对观测变量x进行无条件建模的情况下，有向图形模型将表示观测变量x和潜在变量z上的联合分布pθ(x,z)。观测变量pθ(x)的边际分布由下式给出:
###### 1.7.2 Deep Latent Variable Models
###### 1.7.3 Example DLVM for multivariate Bernuolli data
#### 1.8 Intractabilities

## 2 Variational Autoencoders
#### 2.1 Encoder or Approximate Posterior
#### 2.2 Evidence Lower Bound (ELBO)
###### 2.2.1 Two for One

#### 2.3 Stochastic Gradient-Based Optimization of the ELBO
#### 2.4 Reparameterization Trick
###### 2.4.1 Change of variables
###### 2.4.2 Gradient of expectation under change of variable
###### 2.4.3 Gradient of ELBO
###### 2.4.4 Computation of $\log_{q_{\phi}}(\mathbf z|\mathbf x)$

#### 2.5 Factorized Gaussian posteriors
###### 2.5.1 Full-covariance Gaussian posterior

#### 2.6 Estimation of the Marginal Likelihood

#### 2.7 Marginal Likelihood and ELBO as KL Divergences
#### 2.8 Challenges
###### 2.8.1 Optimization issues
###### 2.8.2 Blurriness of generative model

#### 2.9 Related prior and concurrent work
###### 2.9.1 Score function estimator
## 3 Beyond Gaussian Posteriors
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
[[Feature Unlearning for Pre-trained GANs and VAEs 2024]]


inference

variational inference (VI)

Auto encoder

variational autoencoder