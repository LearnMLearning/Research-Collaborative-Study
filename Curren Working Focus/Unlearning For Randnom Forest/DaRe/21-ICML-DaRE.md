首先明晰随机森林以及决策树基本概念

随机森林（Random Forest）是一种集成学习方法，用于分类和回归任务。它通过构建多个决策树并结合它们的预测结果来提高模型的准确性和稳定性。以下是随机森林的标准结构和关键组成部分：
# 随机森林
### 标准随机森林模型的组成

1. **多个决策树**：
    - 随机森林由多个决策树（通常称为基学习器或弱学习器）组成。每棵树都是在不同的子样本和特征子集上训练的。
    - 通过结合这些树的预测结果，可以减少单棵树的过拟合风险。
2. **Bootstrap 采样**：
    - 在构建每棵树时，从训练数据集中使用有放回抽样（即 Bootstrap 采样）方法随机抽取样本。这意味着每棵树的训练数据集可能包含重复的样本。
    - 这种方法保证了每棵树都不同，从而提高了整体模型的泛化能力。
3. **随机选择特征**：
    - 在构建每棵树的每个节点时，从所有特征中随机选择一个特征子集来决定最佳分割。这进一步增加了树之间的差异，增强了模型的稳健性。
4. **多数投票或平均**：
    - 对于分类任务，随机森林使用多数投票法将所有树的预测结果结合起来，即选择票数最多的类别作为最终预测结果。
    - 对于回归任务，随机森林使用所有树的预测结果的平均值作为最终预测结果。

### 随机森林的优点
- **高准确性**：通过结合多个决策树的结果，随机森林通常比单个决策树模型具有更高的准确性。
- **抗过拟合**：随机森林通过随机采样和特征选择，减少了单个决策树的过拟合风险。
- **可处理高维数据**：随机森林能够处理大量特征，并且具有自动处理缺失值的能力。
- **计算效率高**：随机森林可以并行训练多个决策树，因此在多核处理器上具有较高的计算效率。
### 主要参数
- **n_estimators**：森林中树的数量。较多的树可以提高模型的性能，但也会增加计算成本。
- **max_features**：在分裂每个节点时考虑的特征数量。较少的特征可以增加树之间的差异，但可能会降低单棵树的性能。
- **max_depth**：每棵树的最大深度。限制树的深度可以防止过拟合。
- **min_samples_split**：分裂一个节点所需的最小样本数。较高的值可以防止过拟合。
- **min_samples_leaf**：叶子节点所需的最小样本数。较高的值可以防止过拟合。
### 实现示例（使用 scikit-learn）
以下是使用 scikit-learn 库在 Python 中实现随机森林分类器的示例：
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 加载数据集
data = load_iris()
X = data.data
y = data.target

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建随机森林分类器
clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

# DaRE RF (Data Removal-Enabled forests)
RF variant
DaRE forests leverage several techniques to make deletions efficient:
1. **Only retrain portions** of the model where the structure must change to match the updated database;
2. Consider at most $k$ randomly-selected **thresholds** per attribute;
3. Introduce **random nodes** at the **top of each tree** that minimally depend on the data and thus rarely need to be retrained.
### Retraining Minimal Subtrees 再训练最小子树
##### Decision nodes
Store and update
1. $|D|$ counts for the number of instances
2. $|D_{.,1}|$ (Positive instances)
3. $|D_l|,|D_{l,1}|$ for a set of $k$ thresholds per attribute.
##### Leaf nodes
Store and update 
1. $|D|$  
2. $|D_{.,1}|$ (Positive instances)
3. A list of training instances
![[Pasted image 20240723154759.png]]
------
##### When deleting a training instance $(x,y) \in D$
1. Statistics are updated and used to check if a particular subtree needs retraining.
2. Recompute the split criterion for each attribute-threshold pair. 
3. If a different threshold obtains an improved split criterion, then retrain the subtree rooted at this node.
4. If not retrain (no improvement), update label counts and instance list, make deletion.

![[Pasted image 20240723154909.png]]
### Sampling Valid Thresholds 采样有效阈值
Premise: The optimal threshold for a continuous attribute will always lie between two training instances with adjacent feature values containing opposite labels.
	If the two training instances have the same label, the split criterion improves by increasing or decreasing $v$, called **valid threshold** 有效阈值
	$\exists (x_1,y_1),(x_2,y_2):x_{1,a}=v_1,x_{2,a} = v_2,y_1\ne y_2$ 

### Random Splits
Retrain when $|D_l|=0$ or $|D_r|=0$. But low dependence on data because top of the tree.
$d_{max}$ as the number of layers of the random nodes.

R-DaRE (Random DaRE) $d_{\mathrm{rmax}}=0$
G-DaRe (Greedy DaRE) $d_{\mathrm{rmax}}\ne 0$

### Metric
##### AP (average precision) positive label $< 1 \%$
如前所述，recall 召回率和 precision 精度之间的权衡意味着当我们评估和比较不同的检测算法时，必须同时考虑它们。一种同时考虑召回率和精确率的常用方法是average precision 平均精确率。引理1 清楚地表明，precision 精确度和 recall 召回率是相关的。我们可以将 precision 精度表示为 recall 召回率的函数，用 $p(r)$ 表示。

**Definition** 
$p(r)$ 在 $r = 0$ 到 $r = 1$ 的整个区间内的平均值，就是 average precision
$$
\frac{1}{1-0} \int_0^1 p(r) \, dr = \int_0^1 p(r) \, dr
$$
##### AUC (Area Under Curve)  $[1\%,20\%]$
被定义为 ROC 曲线下的面积
**什么是 ROC 曲线？**
**ROC曲线**（Receiver Operating Characteristic Curve）是用来评估二分类模型表现的工具。它展示了不同阈值下，模型的**真阳性率**（True Positive Rate，TPR）和**假阳性率**（False Positive Rate，FPR）的变化关系。
**真阳性率（TPR）** 也称为敏感度或召回率，是模型正确识别出正类样本的比例，公式为：
$$
\mathrm{TPR} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}
$$
其中，$\mathrm{TP}$ 是真阳性数，$\mathrm{FN}$ 是假阴性数。
**假阳性率（FPR）** 是模型将负类样本错误识别为正类样本的比例，公式为：
$$
\mathrm{FPR} = \frac{\mathrm{FP}}{\mathrm{FP} + \mathrm{TN}}
$$
其中，$\mathrm{FP}$ 是假阳性数，$\mathrm{TN}$ 是真阴性数。

**什么是AUC?**
**AUC**（Area Under the Curve）指的是ROC曲线下面积。AUC值介于0到1之间，反映了模型在区分正负样本方面的能力。
- **AUC = 1**：完美分类器，能够完美区分正负样本。
- **AUC = 0.5**：随机分类器，表现与随机猜测无异。
- **AUC < 0.5**：模型表现不如随机猜测，通常表示模型存在问题。
###### 为什么使用AUC of ROC来比较模型？
1. **阈值独立性**：ROC曲线展示了模型在所有可能的分类阈值下的表现，而不是仅仅在一个特定阈值下，因此AUC提供了对模型性能的综合评估。
2. **不平衡数据**：在处理类别不平衡的数据时，AUC比准确率更有优势，因为它考虑了真阳性率和假阳性率，而准确率可能会被主要类别的样本数所掩盖。
3. **直观性**：AUC的数值范围为0到1，简单明了，数值越接近1，模型性能越好。

##### 教程
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 示例数据（请替换为你的数据）
X = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]])
y = np.array([0, 0, 1, 1, 0])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测概率
y_prob = model.predict_proba(X_test)[:, 1]

# 计算ROC曲线和AUC值
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```
每个实验重复五次。

表5报告了使用5倍交叉验证调优后每个模型在测试集上的预测性能。我们使用值 $[10,25,50,100,250]$ 来调整森林中的树木数量，并使用值 $[1,3,5,10,20]$ 来调整最大深度。在每次分割时要考虑的随机选择属性的最大数目设置为$⌊\sqrt p⌋$。对于G-DaRE模型，我们还使用值 $[5,10,25,50]$ 来调整每个属性 $k$ 的阈值。我们分别使用50%、25%、2.5%和2.5%的训练数据来调整Twitter、Synthetic、Click-Through Rate和Higgs数据集，100%用于所有其他数据集。所有超参数的选定值如表6所示。

我们发现随机树和额外树模型的预测性能始终比SKLearn和G-DaRE模型差。我们还发现，自举对SKLearn模型的影响可以忽略不计。最后，我们观察到G-DaRE模型的预测性能与SKLearn RF几乎相同，它们在9/14数据集上的得分在0.2%以内，在1/14数据集上的得分在0.4%以内，并且G-DaRE RF在外科手术、航班延误、奥运会和信用卡数据集上的得分明显优于SKLearn RF。

表6所示。为G-DaRE和R-DaRE模型选择的超参数(使用误差容限为0.1%，0.25%，0.5%和1.0%)。树的数量(T)、最大深度(dmax)和每个属性考虑的阈值数量(k)是通过使用贪婪构建的模型(即G-DaRE RF)进行5倍交叉验证得到的。为了构建R-DaRE模型，在前一步中找到的T、dmax和k的值是固定的，并且drmax的值是通过从0开始增加1来找到的，直到它的5倍交叉验证分数超过指定的容错范围(与G-DaRE模型的交叉验证分数相比)。

5倍交叉验证（5-fold cross-validation）是一种常用的模型评估方法，用于评估机器学习模型在数据集上的表现。其基本步骤如下：

1. **数据划分**：将原始数据集平均分成5个相等大小的子集（folds）。
2. **训练和验证**：
    - 进行5次训练和验证，每次选择不同的一个子集作为验证集，其他4个子集作为训练集。
    - 具体地，第1次用第1个子集作为验证集，剩下的4个子集作为训练集；第2次用第2个子集作为验证集，剩下的4个子集作为训练集；以此类推，直到第5次用第5个子集作为验证集。
3. **性能评估**：每次训练和验证后，记录模型在验证集上的表现（例如准确率、精确率、召回率等）。
4. **结果汇总**：计算5次验证结果的平均值，作为模型的整体性能评估指标。

这种方法的优点在于：

- **充分利用数据**：每个样本都被用作训练和验证。
- **减少过拟合风险**：通过多次训练和验证，可以得到更稳定和可靠的性能评估结果。

总之，5倍交叉验证是一种在有限数据情况下评估模型表现的有效方法。

![[Pasted image 20240723201007.png]]
