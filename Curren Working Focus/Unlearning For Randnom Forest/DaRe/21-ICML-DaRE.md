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
https://zh.wikipedia.org/wiki/ROC曲线


