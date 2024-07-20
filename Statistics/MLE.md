最大似然估计 (maximum likelihood estimation) 极大似然估计

#### 似然函数 likelihood function
在[数理统计学](https://zh.wikipedia.org/wiki/%E6%95%B0%E7%90%86%E7%BB%9F%E8%AE%A1%E5%AD%A6 "数理统计学")中，**似然函数（**英语：likelihood function）是一种关于[统计模型](https://zh.wikipedia.org/wiki/%E7%BB%9F%E8%AE%A1%E6%A8%A1%E5%9E%8B "统计模型")中的[参数](https://zh.wikipedia.org/wiki/%E6%AF%8D%E6%95%B8 "参数")的[函数](https://zh.wikipedia.org/wiki/%E5%87%BD%E6%95%B0 "函数")，表示模型参数中的**似然性（英语：likelihood）。似然函数在[统计推断](https://zh.wikipedia.org/wiki/%E7%B5%B1%E8%A8%88%E6%8E%A8%E8%AB%96 "统计推断")中有重大作用，如在[最大似然估计](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1 "最大似然估计")和[费雪信息](https://zh.wikipedia.org/wiki/%E8%B4%B9%E9%9B%AA%E4%BF%A1%E6%81%AF "费雪信息")之中的应用等等。文字意义上，“似然性”与“或然性”或“[概率](https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87 "概率")”意思相近，都是指某种事件发生的可能性，但是在[统计学](https://zh.wikipedia.org/wiki/%E7%BB%9F%E8%AE%A1%E5%AD%A6 "统计学")中，“似然性”和“概率”（或然性）有明确的区分：概率，用于在已知一些参数的情况下，预测接下来在观测上所得到的结果；似然性，则是用于在已知某些观测所得到的结果时，对有关事物之性质的参数进行估值，也就是说已观察到某事件后，对相关参数进行猜测。

在这种意义上，似然函数可以理解为[条件概率](https://zh.wikipedia.org/wiki/%E6%9D%A1%E4%BB%B6%E6%A6%82%E7%8E%87 "条件概率")的逆反。在已知某个参数**B**时，事件**A**会发生的概率写作：
$$
P(A|B) = \frac{P(A|B)}{P(B)}
$$
利用 **贝叶斯定理**
$$
P(B|A) = \frac{P(A|B)P(B)}{P(A)}
$$
因此，我们可以反过来构造表示似然性的方法：已知有事件 **A** 发生，运用似然函数 $\mathbb L(\mathbf B|\mathbf A)$，我们估计或猜测参数**B**的不同值的可能性。形式上，似然函数也是一种条件概率函数，但我们关注的[变量](https://zh.wikipedia.org/wiki/%E5%8F%98%E9%87%8F "变量")改变了：
$$
b \mapsto P(A|B=b)
$$
注意到这里并不要求似然函数满足归一性：$\sum_{b\in\mathcal B} P(A|B=b)=1$。一个似然函数乘以一个正的常数之后仍然是似然函数。对所有 $\alpha>0$，都可以有似然函数：
$$
L(b|A) = \alpha P(A|B=b)
$$
###### 例子
考虑投掷硬币的实验。通常来说，已知掷出一枚“公平的硬币”（即正面朝上和反面朝上的概率相同）时，正面（Head）朝上的概率为 $p_H=0.5$，我们可以此推论得知投掷若干次后出现各种结果的可能性。比如说，连续投两次都是正面朝上的概率是 0.25。用条件概率表示，就是：
$$
p(\mathrm{HH}|p_{H}=0.5) = 0.5^2 = 0.25
$$
其中 $\mbox{H}$ 表示正面朝上。

在统计学中，我们更关心的是在*已知一系列投掷的结果时，关于单独投掷一次硬币时正面朝上的概率*（即$p_H$）为何。我们实际上是无法从一系列投掷的结果来逆推真实的$p_H$，但是我们可以推估$p_H$是某个值的可能性为何。举例来说，假设因为这可能不是一枚真正“公平的硬币”，所以我们不知道$p_H$是多少，也无法计算投掷三次硬币其中两次是正面的概率是多少。现在如果我们真的实际去掷了三次硬币，结果其中两次为正面，那我们是否能够依此次实验逆推出$p_H$的信息？如果无法逆推出真实的$p_H$，那我们有没有办法知道，譬如说$p_H=0.5$的可能性为何？$p_H=0.6$ 的可能性又为何？或甚至再更退一步，至少我们能不能知道$p_H=0.5$跟$p_H=0.6$哪一个*比较*有可能？

投掷一次硬币，正面朝上的概率用 $p_H$ 来代表，它就是我们这个例子的参数，而我们用事件 $\mbox A$ 来代表投掷三次硬币其中两次是正面这个事实。使用联合概率（英语：joint probability）计算可知
$$
P(\mbox A | p_H) = 3 \times p_{H}^2 \times (1-p_H)
$$
我们首先假设 $p_H=0.5$，则看到三次投掷中两次是正面的概率为 $P(\mbox A|p_H=0.5)=0.375$。再来如果假设$p_H=0.6$，则看到三次投掷中两次是正面的概率为$P(\mbox A|p_H=0.6)=0.432$。显然地，如果 $p_H=0.6$ 的话，我们看到两个正面的机会比较高。所以当我们投掷了三次硬币并且看到了两次正面，即使我们无法知道实际 $p_H$ 到底是多少，我们至少知道 $p_H$ 是 $0.6$ 的可能性比是 $0.5$的可能性还要高。我们可以合理猜测，$p_H$ 比较可能是 $0.6$ 而非 $0.5$。

这里我们就引进了似然性的概念：似然性代表某个参数为特定值的可能性。从上面例子得知在已观察到事件 $\mbox A$ 的情况下，关于事件 $\mbox A$ 的似然估计为
$$
L(p_H|\mbox A) = P(\mbox A | p_H)
$$
其中 $p_H$ 为我们所要确定的参数。所以当我们投掷硬币三次，其中两次是正面，则 $p_H=0.5$ 的似然性是$L(p_H=0.5|\mbox A)=P(\mbox A|p_H=0.5)=0.375$，而 $p_H=0.6$ 的似然性是$L(p_H=0.6|\mbox A)=P(\mbox A|p_H=0.6)=0.432$。注意，$L(p_H=0.5|\mbox A)=0.375$ 并不是说当已知 $\mbox A$ 发生了，则$p_H$ 为 $0.5$ 的概率是 $0.375$。似然性跟概率具有不同的意义。

若单独看0.375这个数字或0.432这个数字是没有意义的，因为似然性并不是概率，并不是一定介于0到1之间，而所有可能的 $p_H$ 的似然性加起来也不是1，所以单独得知 $L(p_H=0.5|\mbox A)=0.375$ 是没有意义的。似然性是用在把各种可能的 $p_H$ 值放在一起比较，来得知哪个 $p_H$ 值的可能性比较高。而似然函数（在这个例子中，即 $L(p_H|\mbox A)=3\times p_H^2\times(1-p_H)$，除了用来计算似然性外，则是用来了解当参数 $p_H$ 改变时，似然性怎么变化，用来寻找最大可能性的 $p_H$ 值会是多少。\

#### 最大似然估计
###### 最大似然估计的原理
给定一个概率分布 $D$ ，已知其[概率密度函数](https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87%E5%AF%86%E5%BA%A6%E5%87%BD%E6%95%B0 "概率密度函数")（连续分布）或[概率质量函数](https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87%E8%B4%A8%E9%87%8F%E5%87%BD%E6%95%B0 "概率质量函数")（离散分布）为 $f_D$ ，以及一个分布参数 $\theta$，我们可以从这个分布中抽出一个具有 $n$ 个值的采样 $X_1,X_2,\dots,X_n$，利用$f_D$计算出其[似然函数](https://zh.wikipedia.org/wiki/%E4%BC%BC%E7%84%B6%E5%87%BD%E6%95%B0 "似然函数")：
$$L(\theta|x_1,\dots,x_n)=f_{\theta}(x_1,\dots,x_n)$$
若 $D$ 是离散分布，$f_{\theta}$ 即是在参数为 $\theta$ 时观测到这一采样的概率；若其是连续分布，$f_{\theta}$ 则为 $X_1,X_2,\dots,X_n$ 联合分布的概率密度函数在观测值处的取值。一旦我们获得$X_1,X_2,\dots,X_n$，我们就能求得一个关于 $\theta$ 的估计。最大似然估计会寻找关于 $\theta$ 的最可能的值（即，在所有可能的 $\theta$ 取值中，寻找一个值使这个采样的“可能性”最大化）。从数学上来说，我们可以在 $\theta$ 的所有可能取值中寻找一个值使得似然[函数](https://zh.wikipedia.org/wiki/%E5%87%BD%E6%95%B0 "函数")取到最大值。这个使可能性最大的 $\hat \theta$ 值即称为 $\theta$ 的**最大似然估计**。由定义，最大似然估计是样本的函数。

###### 注意
- 这里的似然函数是指 $x_1,x_2,\dots,x_n$ 不变时，关于 $\theta$ 的一个函数。
- 最大似然估计不一定存在，也不一定唯一。

###### 推导
最大似然估计可以从 [KLD] KL 散度 相对熵推导而来。[相对熵](https://zh.wikipedia.org/wiki/%E7%9B%B8%E5%AF%B9%E7%86%B5 "相对熵")衡量了使用一个给定分布 $Q$ 来近似另一个分布 $P$ 时的信息损失，对于离散型随机变量，可以用以下公式：
$$
D_{KL}(P\|Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$
其中，$P$ 是真实分布，$Q$ 是近似分布。在最大似然估计的情景下，假设分布拥有一系列参数 $\theta$，我们希望通过样本得到参数的估计值 $\hat \theta$。我们可以利用[相对熵](https://zh.wikipedia.org/wiki/%E7%9B%B8%E5%AF%B9%E7%86%B5 "相对熵")来评判估计的好坏：
$$
D_{KL}(p_{\theta}(x)||p_{\hat \theta} (x)) = \sum_{x\in E}p_{\theta} (x) \log \frac{p_{\theta}(x)}{p_{\hat \theta}(x)}
$$
根据期望的定义，我们可以将上式改写为：
$$\begin{aligned}
D_{KL}(p_{\theta}(x)\| p_{\hat \theta}(x)) &= \mathbb E_{\theta} \left[\log \left(\frac{p_{\theta}(x)}{p_{\hat \theta}(x)} \right) \right]\\
&= \mathbb E_{\theta} [\log p_{\theta}(x)] - \mathbb E_{\theta} [\log p_{\hat \theta}(x)]
\end{aligned}$$
KL值越大，参数估计越坏，因此，需要通过改变估计参数$\hat \theta$ 的值来获得最小的值，所对应的参数极为最佳估计参数。即：
$$
\hat \theta_{\mathrm{best}} = \arg \min_{\hat \theta} D_{KL} (p_{\theta}(x)\| p_{\hat \theta} (x))
$$
假设有n![{\displaystyle n}](https://wikimedia.org/api/rest_v1/media/math/render/svg/a601995d55609f2d9f5e233e36fbe9ea26011b3b)个样本，根据[大数定理](https://zh.wikipedia.org/wiki/%E5%A4%A7%E6%95%B8%E6%B3%95%E5%89%87 "大数定律")，可以进行替换：
$$
\mathbb {E}_{\theta }[\log p_{\hat{\theta}}(x)]\rightsquigarrow \frac 1n \sum_{i=1}^{n} \log p_{\hat \theta}(x)
$$
即，可以通过下式评估：
$$
D_{KL}(p_{\theta}(x)\| p_{\hat \theta}(x)) = \mathbb E_{\theta} [\log p_{\theta} (x)] - \frac 1n \sum_{i=1}^n \log p_{\hat \theta}(x_i)
$$
对于一个已知的分布，其参数 $\theta$ 是确定的。因此，$\mathbb {E} _{\theta }[\log p_{\theta }(x)]$为常数。因此，我们可以通过最小化KL值获得最佳估计参数：
$$\begin{aligned}{\hat {\theta }}&=\arg \min _{\hat {\theta }}\mathbb {E} _{\theta }[\log p_{\theta }(X)]-{\frac {1}{n}}\sum _{i=1}^{n}\log p_{\hat {\theta }}(x_{i})\\&\Rightarrow \arg \min _{\hat {\theta }}-{\frac {1}{n}}\sum _{i=1}^{n}\log p_{\hat {\theta }}(x_{i})\\&\Rightarrow \arg \max _{\hat {\theta }}{\frac {1}{n}}\sum _{i=1}^{n}\log p_{\hat {\theta }}(x_{i})\\&\Rightarrow \arg \max _{\hat {\theta }}\sum _{i=1}^{n}\log p_{\hat {\theta }}(x_{i})\\&\Rightarrow \arg \max _{\hat {\theta }}\log \left[\prod _{i=1}^{n}p_{\hat {\theta }}(x_{i})\right]\\&\Rightarrow \arg \max _{\hat {\theta }}\prod _{i=1}^{n}p_{\hat {\theta }}(x_{i})\\\end{aligned}$$
因此，要得到最佳参数估计值，只需要最大化$\prod _{i=1}^{n}p_{\hat {\theta }}(x_{i})$，这就是最大似然函数。对于连续型随机变量，有相同的结论。

###### 例子
https://zh.wikipedia.org/wiki/最大似然估计