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
DaRE
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
## Experimental Evaluation
实验评估的目的是调查我们的遗忘框架的整体性能以及我们提出的优化的影响。具体来说，实验的目标是回答以下问题:
- 建议的遗忘框架有多有效?未学习的模型与从头开始训练的模型是否有本质上的分歧? 
- 当有不同大小的数据需要遗忘时，遗忘的成本是多少? 
- 所提出的取消学习方法能否取消非随机抽样的训练姿态，即用于添加后门的对抗性姿态? 
- 建议的优化对遗忘时间和测试精度有何影响?

###### 默认参数设置
$J = 20,$$D_{un} = 0.1\%,$$lazy\_update\_freq = 20,$$sample\_rate = 0.1,$$L_r = 0$

###### Data
![[Pasted image 20240727100533.png]]
#### Model Functionality After Unlearning
C2W: 在再训练中正确预测而在学习中错误预测的测试实例的比率
W2C: 在再训练中被错误预测的测试实例，并通过遗忘来纠正\
W2W: 两个模型有不同错误预测的情况
在$D_{un}$ Size列中，$1$ 表示我们只学习 $1$ 个训练实例，$0.1\%$对应 $|D_{un}| = 0.1\%\cdot |D_{tr}|$。我们在下文中采用这种符号。
![[Pasted image 20240727100608.png]]
#### Unlearning Cost
除了模型的功能，遗忘的代价在实际应用中也是至关重要的。表4说明了训练和遗忘时间。令人惊讶的是，虽然DaRE-RF训练更简单的随机森林，但我们更复杂的GBDT方法使用的训练时间更少。一个潜在的原因是DaRE-RF是用python编写的，而我们的框架是基于高度优化的c++代码基础。除了由于实施造成的时差，我们还可以比较相对的趋势。对于只学习一个单独的训练实例，DaRE-RF比我们的方法花费更少的(相对)时间。然而，当未学习数据的数量增加时，DaRE-RF的未学习时间随着 $D_{un}$的大小线性增加。同时，我们提出的方法是批处理的，时间成本只随着需要遗忘的实例数量呈次线性增加。因此，我们的方法需要更少的时间来忘记 $0.5\%$ 和 $1\%$ 的数据。

![[Pasted image 20240727101223.png]]
此外，我们在 表5 中报告了用于取消学习的辅助数据结构的总分配内存和内存。正如我们的理论分析所期望的那样，在更大的数据集上取消学习需要更多的内存作为辅助存储。请注意，在我们的实验中，为了方便起见，我们将所有内容加载到内存中。从系统的角度来看，辅助数据结构不需要同时放在内存中。我们只需要为一个树加载辅助数据结构，其他树的数据可以卸载到磁盘上。我们相信这不会成为应用我们的遗忘技术的瓶颈。

![[Pasted image 20240727101335.png]]

#### Verifying by Backdoor Attacking
现在我们已经证实了学习训练数据随机子集的有效性。在这个后门攻击测试中，我们通过在训练数据集中加入有毒数据来添加后门。具体来说，数据投毒 (data poisoning) 是这样做的:

我们随机选择一个训练数据子集，先设置几个特征作为该数据集中最大的特征值，并将其标签改为 $1$。

在 表6 中，我们在 $3$ 种设置下对模型进行评估，在干净数据上进行训练，在中毒数据上进行训练，在中毒数据上进行训练的模型中去学习后门。我们分别报告清洗数据和后门数据的测试精度。正如预期的那样，由于GBDT作为一个更复杂的模型，优于随机森林，当模型在干净数据上训练时，我们提出的方法在干净数据上具有更高的测试精度。当我们在中毒数据上训练模型时，后门安装成功，两种方法对后门数据的测试准确率均达到100%。在干净数据上的测试精度与我们在干净数据上训练模型的精度几乎相同。在我们删除后门数据后，如果删除后门，后门(最后一列)的测试精度应该与第4列相同。这两种方法都可以通过取消学习来消除有毒数据的影响。后门测试证明，我们的方法确实可以在这种对抗性攻击场景中忘记 $D_{un}$的影响——$D_{un}$不是随机选择的。
![[Pasted image 20240727102131.png]]

#### Verifying by Membership Inference Attack
隶属度推理攻击(MIA) $[21,34,38]$ 用于预测数据实例是否在训练数据集中。我们用多类数据集测试了我们的学习方法，因为MIA在二元分类上效果不佳 $[21]$。MIA模型 $[38]$ 是根据每个类别的预测概率进行训练的。我们使用原始训练数据子集(标记为“in”)和测试数据子集(标记为“out”)的预测概率-其他未使用的数据将被利用来验证攻击准确性。然后，我们执行取消学习，并在未使用的数据上生成未学习的新模型的预测类概率。取消学习后，未学习的数据被标记为“out”，而其他数据的标签保持不变。如果我们成功地学习了数据，我们应该期望对未学习数据的攻击精度与原始“输出”数据相同。表7 给出了攻击准确率，即隶属度推理攻击的准确率成功区分了数据实例是否在训练集中。in是训练集中的数据;“in-un”是训练集中不会被遗忘的数据;“un (in)”是将被遗忘的数据，并标记为in。注意它们现在在训练集中;“out”是测试集中的那个，在训练和遗忘过程中没有被触及;“un (out)”是未学习的数据，现在被标记为out。我们可以观察到，在遗忘之后，未学习数据的行为与“输出”数据的行为一样。我们可以得出结论，从 MIA 的角度来看，数据是被遗忘的。

![[Pasted image 20240727102520.png]]

#### Ablation Study 消融实验
###### Random Layers
我们在图6中研究了随机层的影响。随着随机层数的增加，学习时间减少。遗忘时间由许多与随机层无关的其他方面组成，例如导数计算和增益计算。因此，学习时间不会随着随机层数的增加而呈指数递减。随机层主要降低了再训练的成本，当再训练很少发生时，例如在HIGGS中，再训练成本的降低并不显著。

此外，随机层也会影响模型的精度(图7)。当随机层数小于4时，大多数数据集的精度变化最小。字母的准确性明显下降。根据经验，我们建议使用1或2个随机层，这通常会在不损失测试精度的情况下减少最佳的边际时间。
###### Random sampled split candidates
随机抽样的分裂候选中的抽样率控制了训练(和学习)期间可能分裂的数量。我们在图8中报告了采样率对取消学习时间、准确性和重新训练节点数量的影响。由于篇幅限制，我们只在 AdultIncome 上展示这个实验。其他数据集的结果也显示出类似的趋势。左上角的图显示，当样本率较低时，考虑的分裂候选数较少，并且遗忘时间如预期的那样减少。下一个问题是，这种采样会导致测试精度损失吗?如右上图所示，我们将采样的拆分候选解决方案与从头开始在 $D_{tr} \backslash D_{un}$上重新训练的情况进行比较。测试精度几乎相同，没有观察到明显的损失。在图的底部，我们给出了重新训练节点的累计数量(左下)和重新训练节点的总数(右下)。当我们选取更少的分裂候选样本时，分裂变得更加稳健。因此，需要重新训练的节点数量下降。在我们实验中使用的所有数据集中，5% 的样本率总是有利于时间，而不会出现明显的准确性损失。
###### Lazy update frequency
在第2.3节中，我们讨论通过从存储的和中减去更新的导数来近似导数和。更积极地说，我们不需要在每次迭代中更新导数。在这个实验中，我们评估了延迟更新频率的影响。就像上面的采样率实验一样，图9还报告了学习时间、测试准确性和重新训练节点的数量。更新频率= inf意味着我们不更新导数。在遗忘时间方面，当我们更新导数的频率较低时，预计会有更好的遗忘时间。当AdultIncome的更新频率较低时，测试的准确性只会略有下降。这里，当班级数量较大时，下降效应更为显著，因此，我们也在右列中给出了Letter上的结果。再训练节点的数量与AdultIncome相当，而在Letter上分歧很大。这也解释了当我们更新导数频率较低时精度下降的原因。作为准确性和取消学习时间之间的权衡，禁用衍生更新可以将取消学习时间减少到17%左右，在Letter数据集上的准确性损失约为0.5%。频率选择取决于数据集的分布以及应用程序用例，即可以容忍多少精度损失。



