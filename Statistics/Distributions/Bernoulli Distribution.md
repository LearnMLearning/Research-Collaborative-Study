伯努利试验 只有两种可能的结果 （成功 / 不成功）
而 试验结果的分布 就是 伯努利分布：
###### 定义：
随机变量 $X$ 代表 **一次** 伯努利试验 **是否** 成功，并且 $\begin{cases}\mathbb P(X=1)=p\\ \mathbb P(X=0) = 1-p \end{cases}$ 
则，随机变量 $X$ 是伯努利分布，记作 $X \sim \text{Ber}(p)$
**PMF:**
$$
p_X(x;p) = p^x (1-p)^{1-x}, x\in \{0,1\}
$$
**例子:**
抛硬币，抛到人物面记作1（事件成功）；抛到字面记作0（事件失败）

**属性**
1. 期望 (Mean)
	$E[x] = \sum_{x=0,1} xp(x) = p$
2. 方差 (Variance)
	$\text{Var}[x] = E[x^2] - E[x] = p(1-p)$

###### Binomial 与 Bernoulli 之间的转化
$X_i$：是 第$i$次 伯努利试验的结果，$X_i=1$ 代表成功，$X_i=0$ 代表失败
$X_i$ 就是伯努利分布
$X = X_1 + X_2 + \cdots + X_n$，$X$ 就是二项分布

###### 属性
1. 期望 (Mean)
	$E[x] = \sum E[X_i] = np$
2. 方差 (Variance)
	$\mathrm{Var}[X] = \sum \mathrm{Var}[X_i] = np (1-p)$
