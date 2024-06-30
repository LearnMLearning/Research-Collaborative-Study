分类问题中，假设有 $K$ 个类，样本点属于第 $k$ 类的概率为 $p_k$，则概率分布的基尼指数定义为
$$
\text{Gini}(p) = 1-\sum_{k=1}^Kp_i^2
$$
$$
G_{D,\mathcal Y}(a,v) = \sum_{b \in \{l,r\}} \frac{|D_b|}{|D|} \left(1-\sum_{y\in\mathcal Y} \left(\frac{|D_{b,y}|}{|D_b|} \right)^2 \right)
$$
