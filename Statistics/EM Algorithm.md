**最大期望算法 Expectation- maximization algorithm**

**最大期望算法**（**Expectation-maximization algorithm**，又译**期望最大化算法**）在统计中被用于寻找，依赖于不可观察的隐性变量的概率模型中，参数的最大似然估计。

在[统计](https://zh.wikipedia.org/wiki/%E7%BB%9F%E8%AE%A1 "统计")[计算](https://zh.wikipedia.org/wiki/%E8%AE%A1%E7%AE%97 "计算")中，**最大期望（EM）算法**是在[概率模型](https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B "概率模型")中寻找[参数](https://zh.wikipedia.org/wiki/%E5%8F%82%E6%95%B0 "参数")[最大似然估计](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1 "最大似然估计")或者[最大后验估计](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E5%90%8E%E9%AA%8C%E6%A6%82%E7%8E%87 "最大后验概率")的[算法](https://zh.wikipedia.org/wiki/%E7%AE%97%E6%B3%95 "算法")，其中概率模型依赖于无法观测的[隐变量](https://zh.wikipedia.org/wiki/%E9%9A%90%E5%8F%98%E9%87%8F "隐变量")。最大期望算法经常用在[机器学习](https://zh.wikipedia.org/wiki/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0 "机器学习")和[计算机视觉](https://zh.wikipedia.org/wiki/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89 "计算机视觉")的[数据聚类](https://zh.wikipedia.org/wiki/%E6%95%B0%E6%8D%AE%E8%81%9A%E7%B1%BB "数据聚类")（Data Clustering）领域。最大期望算法经过两个步骤交替进行计算，第一步是计算期望（E），利用对隐藏变量的现有估计值，计算其最大似然估计值；第二步是最大化（M），最大化在E步上求得的[最大似然值](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1 "最大似然估计")来计算参数的值。M步上找到的参数估计值被用于下一个E步计算中，这个过程不断交替进行。

###### EM 简单教程
EM 是一个在已知部分相关变量的情况下，估计未知变量的迭代技术。EM 的算法流程如下：
1. 初始化分布参数
2. 重复直到收敛：
	1. E步骤：根据参数的假设值，给出未知变量的期望估计，应用于缺失值。
	2. M步骤：根据未知变量的估计值，给出当前的参数的极大似然估计。
###### 最大期望过程说明
我们用 $\mathbf y$ 表示能够观察到的不完整的变量值，用$\mathbf x$表示无法观察到的变量值，这样$\mathbf x$和$\mathbf y$一起组成了完整的数据。$\mathbf x$可能是实际测量丢失的数据，也可能是能够简化问题的隐藏变量，如果它的值能够知道的话。例如，在[混合模型](https://zh.wikipedia.org/wiki/%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B "混合模型")中，如果“产生”样本的混合元素成分已知的话最大似然公式将变得更加便利（参见下面的例子）。

**估计无法观测的数据**
让 $p$ 代表矢量 $\theta:p(\mathbf y,\mathbf x|\theta)$ 定义的参数的全部数据的[几率密度函数](https://zh.wikipedia.org/wiki/%E6%A9%9F%E7%8E%87%E5%AF%86%E5%BA%A6%E5%87%BD%E6%95%B8 "几率密度函数")（连续情况下）或者[几率质量函数](https://zh.wikipedia.org/wiki/%E6%A9%9F%E7%8E%87%E8%B3%AA%E9%87%8F%E5%87%BD%E6%95%B8 "几率质量函数")（离散情况下），那么从这个函数就可以得到全部数据的[最大似然值](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1 "最大似然估计") [MLE]，另外，在给定的观察到的数据条件下未知数据的[条件分布](https://zh.wikipedia.org/wiki/%E6%9D%A1%E4%BB%B6%E5%88%86%E5%B8%83 "条件分布")可以表示为：
$$
p(\mathbf x|\mathbf y,\theta) = \frac{p(\mathbf y,\mathbf x|\theta)}{p(\mathbf y|\theta)} = \frac{p(\mathbf y|\mathbf x,\theta)p(\mathbf x|\theta)}{\int p(\mathbf y|\mathbf x,\theta)p(\mathbf x|\theta) \, d\mathbf x}
$$
