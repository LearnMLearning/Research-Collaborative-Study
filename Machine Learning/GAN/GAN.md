![[Pasted image 20240715223410.png]]
## 1. GAN 基本模型
本节首先介绍 GAN 基本模型的定义，然后给出其学习算法，最后给出相关理论分析结果。
#### 1.1 模型
目标是从已给训练数据中学习生成数据的模型，用模型自动生成新的数据，包括图像.语音数据。一个直接的方法是假设已给数据是由一个概率分布产生的数据，通过**极大似然估计学习**这个概率分布，即**概率密度函数**。当**数据分布非常复杂**时，很难给出适当的概率密度函数的定义，以及有效地学习概率密度函数。生成对抗网络 GAN 不直接**定义和学习数据生成的概率分布**，而是通过导入**评价生成数据“真假”的机制**来解决这个问题。

GAN 由一个生成网络(generator)和一个判别网络(discriminator)组成，相互进行博弈(对抗)，**生成网络**生成**数据(假数据)**，判别网络判别数据是**已给数据(真数据)** 还是**生成数据(假数据)**。学习的过程就是**博弈**的过程。生成网络和判别网络不断提高自己的能力，当最终达到**纳什均衡([[Nash Equilibrium]])** 时，生成网络可以以假乱真地生成数据，判别网络不能判断数据的真假。

这里假设生成网络和判别网络是深度神经网络，都有足够强的学习能力。训练数据并没有直接用于生成网络的学习，而是用于判别网络的学习。判别网络能力提高之后用于生成网络能力的提高，生成网络能力提高之后再用于判别网络能力的提高，不断循环。

图 28.1 显示 GAN 的框架。假设已给训练数据 $\mathcal D$ 遵循分布 $P_\text{data}(\mathbf x)$，其中 $a$ 是样本。生成网络用 $\mathbf x = G(\mathbf z;\mathbf \theta)$ 表示, 其中 $\mathbf z$ 是输入向量(种子), $\mathbf x$ 是输出向量(生成数据),$\mathbf \theta$ 是网络参数。判别网络是一个二类分类器，用 $P(1|\mathbf x)= D(\mathbf x;\varphi)$表示，其中 $\mathbf x$ 是输入向量, $P(1|\mathbf x)$ 和 $1- P(1|\mathbf x)$ 是输出概率，分别表示输入 $\mathbf x$ 来自训练数据和生成数据的概率，$\mathbf \varphi$ 是网络参数。种子 $\mathbf z$ 遵循分布 Pseed(z)，如标准正态分布或均匀分布。生成网络生成的数据分布表示为 $P_\text{gen}(\mathbf x)$，由 $P_{\text{seed}}(\mathbf z)$ 和 $x=G(\mathbf z;\mathbf \theta)$ 决定。

![[Pasted image 20240716003623.png]]
如果生成网络参数 $\mathbf \theta$ 固定，可以通过最大化以下目标函数学习判别网络参数 $\mathbf \varphi$，使其具备判别真假数据的能力。
$$
\mathop{\max}_{\varphi}\{E_{\mathbf x \sim P_{\text{data}(\mathbf x)}}[\log D(\mathbf x;\varphi)] + E_{\mathbf z\sim P_{\text{seed}(\mathbf z)}} [\log(1-D(G(\mathbf z;\mathbf \theta);\bar \varphi))] \}
$$
如果判别网络参数 $\varphi$ 固定，那么可以通过最小化以下目标函数学习生成网络参数 $\mathbf \theta$，使其具备以假乱真地生成数据的能力。
$$
\mathop{\min}_{\theta} \{E_{\mathbf z\sim P_{\text{seed}(\mathbf z)}}[\log (1-D(G(\mathbf z;\mathbf \theta);\bar {\mathbf \varphi}))] \}
$$
判别网络和生成网络形成博弈关系，可以定义以下的极大极小问题，也就是 GAN 的学习目标函数。
$$
\min_{\mathbf \theta} \mathop{\max}_{\varphi}\{E_{\mathbf x \sim P_{\text{data}(\mathbf x)}}[\log D(\mathbf x;\varphi)] + E_{\mathbf z\sim P_{\text{seed}(\mathbf z)}} [\log(1-D(G(\mathbf z;\mathbf \theta);\varphi))] \}
$$
后续定理证明这个极小极大问题的解 $\varphi^*$ 和 $\theta^*$存在，也就是纳什均衡存在。GAN的学习算法就是求极小极大值的最优解方法。

可以对 GAN 做这样一个比喻。生成网络是仿造者，判别网络是鉴别者。仿造者制作赝品;鉴别者既得到真品又得到赝品，判断作品的真伪。仿造者与鉴别者之间展开博弈，各自不断提高自己的能力，最终仿造者制作出的赝品真假难辨，鉴别者无法判断作品的真伪。注意在这个过程中鉴别者间接地把自己的判别方法告诉了仿造者，所以两者之间既有对抗关系又有“合作”关系。

#### 1.2 学习算法
**GAN 学习算法**
![[Pasted image 20240716005005.png]]

这里不进行 $\log(1 - D(G(\mathbf z; \mathbf \theta); \mathbf \varphi))$ 的最小化，而是进行 $\log(D(G(\mathbf z;\theta);\varphi)$ 的最大化。这是因为在学习的初始阶段，生成网络较弱，判别网络很容易区分训练数据和生成数据，最小化 $\log(1 - D(G(\mathbf z; \mathbf \theta);\mathbf \phi))$ 会使学习很难进行下去。因此，判别网络和生成网络的学习都使用**梯度上升法**。

判别网络训练时从训练数据和生成数据中同采样 $M$ 个样本，也就是各以 $0.5$ 的概率选取训练数据和生成数据。判别网络学习迭代 $S$ 次后，生成网络学习迭代 $1$次。这样可以保证训练判别网络有足够能力时再训练生成网络。$M$ 和 $S$ 是超参数，要在具体应用中调节。
#### 1.3 理论分析


## 2. 图像生成中的应用
#### 2.1 转置卷积

###### 1. 转置卷积的定义

###### 2. 转置卷积的大小

###### 3. 转置卷积上的采样

#### 2.2 DCGAN


[[Feature Unlearning for Pre-trained GANs and VAEs 2024]]
