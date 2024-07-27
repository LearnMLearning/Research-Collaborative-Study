###### Challenges
1. **Differentiability**
	Gradient-based methods in DNN require computing the gradient of the deleted data while GBDT is not differentiable.
2. **Fine-tuning**
	DNN: not increase parameter size
	GBDT: more tree, more parameters, high inference costs
3. **Tree generation**
	RandF: independently
	GBDT: depend on the residual of the trees of all previous iterations
###### Baseline
Retrain

#### Training
$D_{tr} = \{y_i,\mathbf x_i\}_{i=1}^N,N$ is the number of training instances.
For the $i^{th}$ instance: $\mathbf x_i$ is the feature vector
$y_i \in \{1,2,\dots,K\}$
$$
p_{i,k} = \mathbf {Pr}(y_i = k|\mathbf x_i)=\frac{e^{F_{i,k}(\mathbf x_i)}}{\sum_{s=1}^Ke^{F_{i,s}(\mathbf x_i)}},i = 1,2,\dots,N
$$
![[Pasted image 20240727090320.png]]
![[Pasted image 20240727090335.png]]
###### Robust LogitBoost
![[Pasted image 20240727090343.png]]
![[Pasted image 20240727090411.png]]

###### MART
![[Pasted image 20240727090420.png]]
![[Pasted image 20240727090435.png]]
###### Inference
$K\times M$ regression tree $f_{i,k}$
![[Pasted image 20240727090703.png]]
[[softmax]]
![[Pasted image 20240727090850.png]]

###### Discretize features into integers as a preprocessing
![[Pasted image 20240727091026.png]]
PingLi, Christopher J.C.Burges, and Qiang Wu. Mcrank: Learning to rank using multiple classification and gradient boosting. In Advances in Neural Information Processing Systems (NIPS), pages 897–904, Vancouver, Canada, 2007.

Ping Li and Weijie Zhao. Package for fast abc-boost. CoRR, abs/2207.08770, 2022.

#### Unlearning
###### Unlearning in GBDT
![[Pasted image 20240727091517.png]]
###### Unlearning in one tree
![[Pasted image 20240727091624.png]]

![[Pasted image 20240727091636.png]]
Thinking of HedgeCut
###### Time Complexity
![[Pasted image 20240727092247.png]]

#### Incremental Update
###### Split Gain
![[Pasted image 20240727093306.png]]
To remove the term $D_{tr}$ from the time complexity: incremental computation method: *Can we save the summation of the derivatives when we train the data, then we only need to compute the derivatives for $D_{un}$ and subtract them from the saved derivative sum?*
all derivatives with the same feature value (feature value is in $\{0,1,\dots,B-1\}$)
$J$ is the number of terminal nodes
$|x_i|$ is the number of non-empty features for training instances
###### Derivatives
If a split change and subtree is retrained, the derivatives for all data in the retrained subtree will be changed.重新计算这些导数的成本不再仅仅取决于 $D_{un}$ —— 如果需要在树的根部重新训练，它可能会像 $D_{tr} \backslash D_{un}$ 一样大，重新计算所有的导数是十分耗时的。

注意，我们放宽了学习问题(第1节)-不需要从头构建与重新训练的树完全相同的树。我们可以在训练中把更新的导数和保存的导数和合并起来吗?从GBDT训练的角度来看，每棵树都是基于之前所有迭代中学习到的残差来构建的。修改一棵树意味着它将在接下来的迭代中连锁地影响其他树。然而，在遗忘情景中，所有的树都已经被训练得很好了。我们的目标是去除 $D_{un}$ 的影响，这是训练数据集 $D_{tr}$ 的一个小子集。直观地说，$D_{tr}\backslash D_{un}$ 中数据的导数变化应该是最小的，因为我们只稍微改变树来忘记 $D_{un}$ - 树应该仍然可以很好地工作在 $D_{tr} \backslash D_{un}$上。因此，我们可以从保存的和中减去更新的导数，以在增益计算中近似更新的和。通过这种松弛，我们能够增量地计算分离增益和导数。
[[Amnesiac Machine Learning, 2020]]

此外，根据这种直觉，我们进一步以惰性更新方式更积极地减少导数计算成本:我们只在每次迭代时更新导数。这导致使用更陈旧的导数进行增益计算，但实质上节省了导数更新的成本 $\frac{1}{\lambda}$。我们在第3.5节中对提议的松弛和延迟更新进行了经验评估。
#### Random Sampled Split Candidates
[[21-SIGMOD-HedgeCut]]
![[Pasted image 20240727095046.png]]
我们必须评估更好的贴合所带来的优势是否值得花费昂贵的再培训。一种可能的解决方案是启发式 (heuristically) 地为增益差设置一个(相对)阈值。然而，由于增益的分布是未知的，并且随不同的数据集而变化，因此找到这样一个通用阈值并不是微不足道的。

在本文中，我们遵循另一个方向，通过限制分割值来减少频繁的再训练。原始训练过程枚举 ${0,1,\dots,B-1}$ 中所有的变量变量的可能拆分。我们随机抽取了若干个分离的候选值作为样本，并且只在训练和学习中对这些候选值进行增益计算。图4 描述了一个示例。原来的训练间隔是 $5$ 。当我们使用所有的方差值(采样率为100%)作为分割候选值时，$6$ 的增益比原来的分割 $5$高。因此，我们将重新训练该节点的子树。当采样率为 $50\%$ 时，分割 $5$ 仍然是所有候选对象中的最佳分割点，分割保持不变，我们节省了再训练的成本。此外，由于我们在更少的候选对象上计算增益，因此改进了训练和忘记时间。

随机抽样的分割候选的空间开销是最小的:我们只需要将抽样的分割候选存储为辅助数据。此外，作为随机抽样分割可候选优化的副产品，我们大大减少了用于增量分割增益计算的辅助数据的内存占用(在2.3节中讨论)，从 $O(J\cdot |x_i|\cdot B)$ 到 $O(J\cdot |x_i| \cdot \alpha B)$——我们有更少的分割候选，因此需要维护的求和也更少。

#### Random Layers
[[21-ICML-DaRE]]
![[Pasted image 20240727095056.png]]
##