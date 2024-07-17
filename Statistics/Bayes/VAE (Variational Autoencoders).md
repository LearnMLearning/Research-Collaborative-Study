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
	**[[GAN]]** 生成对抗网络 (Goodfellow et al., 2014) 该种生成建模范式获得了极大的关注：VAEs 和 GANs 似乎有互补的特性：虽然GANs 可以生成高主观感知质量的图像，但与基于似然的生成模型相反。它们往往缺乏对数据的充分支持。与其他基于似然的模型一样，VAEs生成的样本更分散，但就似然准则而言，它是更好的密度模型。因此，
#### 1.2 Aim
#### 1.3 Probabilistic Models and Variational Inference
#### 1.4 Parameterizing Conditional Distributions with Neural Networks
#### 1.5 Directed Graphical Models and Neural Networks
#### 1.6 Learning in Fully Observed Models with Neural Nets
#### 1.7 Learning and Inference in Deep Latent Variable Models
#### 1.8 Intractabilities

## 2 Variational Autoencoders
#### 2.1 Encoder or Approximate Posterior
#### 2.2 Evidence Lower Bound (ELBO)

#### 2.3 Stochastic Gradient-Based Optimization of the ELBO
#### 2.4 Reparameterization Trick]
#### 2.5 Factorized Gaussian posteriors
#### 2.6 Estimation of the Marginal Likelihood
#### 2.7 Marginal Likelihood and ELBO as KL Divergences
#### 2.8 Challenges
#### 2.9 Related prior and concurrent work
## 3 Beyond Gaussian Posteriors
#### 3.1 Requirements for Computational Tractability
#### 3.2 Improving the Flexibility of Inference Models
#### 3.3 Inverse Autoregressive Transformations
#### 3.4 Inverse Autoregressive Flow (IAF)

#### 3.5 Related Work
## 4 Deeper Generative Models
#### 4.1 Inference and Learning with Multiple latent Variables
#### 4.2 Alternative methods for increasing expressivity
#### 4.3 Autoregressive Models
#### 4.4 Invertible transformations with tractable Jacobian determinant
#### 4.5 Follow-Up Work
## 5 Conclusion
## Appendix
#### A.1 Notation and definitions
#### A.2 Alternative methods for learning in DLVMs
#### A.3 Stochastic Gradient Descent

## Related Works
[[Feature Unlearning for Pre-trained GANs and VAEs 2024]]


inference

variational inference (VI)

Auto encoder

variational autoencoder