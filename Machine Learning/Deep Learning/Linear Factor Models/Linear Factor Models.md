深度学习的许多前沿研究涉及建立输入的概率模型 $p_{\mathrm{model}}(\mathbf x)$。原则上，这种模型可以在给定任何其他变量的情况下，使用概率推理来预测其环境中的任何变量。这些模型中的许多还具有潜在变量，如 $p_{\mathrm{model}}(\mathbf x) = \mathbb E_\mathbf hp_\mathrm{model}(\mathbf x | \mathbf h)$ 。这些潜在变量提供了另一种表示数据的方法。基于潜变量的分布式表示可以获得我们在深度前馈和循环网络中看到的所有表示学习的优点

在本章中，我们描述了一些最简单的具有潜在变量的概率模型:**线性因子模型**。这些模型有时被用作混合模型的基础(Hinton et al.， 1995;Ghahramani and Hinton, 1996;Roweis et al.， 2002)或更大的深度概率模型(Tang et al.， 2012)。它们还展示了构建生成模型所必需的许多基本方法，更高级的深度模型将进一步扩展这些方法。

线性因子模型通过使用随机线性解码器函数 (stochastic linear decoder) 来定义，该函数通过向 $\mathbf h$ 的线性变换添加噪声来生成 $\mathbf x$。

这些模型很有趣，因为它们使我们能够发现具有简单联合分布的解释性因素。使用线性解码器的简单性使这些模型成为第一批被广泛研究的潜在变量模型。

线性因子模型描述数据生成过程如下。首先，我们从一个分布中抽样解释因子 (explanatory factors) $\mathbf h$
$$
\mathbf h \sim p(\mathbf h)
$$
其中 $p(\mathbf h)$ 是一个 factorial distribution，$p(\mathbf h) =\prod_ip(h_i)$，因此很容易从中抽样。接下来，我们对给定因子的实值可观察变量进行采样


#### 1 [[Probabilistic PCA and Factor Analysis]]


#### 2 [[Independent Component Analysis (ICA)]]


#### 3 [[Slow Feature Analysis (SFA)]]


#### 4 [[Sparse Coding]]


#### 5 [[Manifold Interpretation of PCA]]
