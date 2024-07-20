**KL散度**（**Kullback-Leibler divergence**，简称**KLD**），在消息系统中称为**相对熵**（relative entropy），在连续时间序列中称为随机性（randomness），在统计模型推断中称为消息增益（information gain）。也称消息散度（information divergence）。

**KL散度**是两个概率分布P和Q差别的非对称性的度量。 KL散度是用来度量使用基于Q的分布来编码服从P的分布的样本所需的额外的平均比特数。典型情况下，P表示数据的真实分布，Q表示数据的理论分布、估计的模型分布、或P的近似分布。

###### 定义
对于离散随机变量，其概率分布 $P$ 和 $Q$ 的 KL 散度可按下式定义为
$$
D_{KL}(P||Q) = - \sum_i P(i) \ln \frac{Q(i)}{P(i)}
$$
等价于
$$
D_{KL}(P||Q) = \sum_i P(i) \ln \frac{P(i)}{Q(i)}
$$
即按概率 $P$ 求得的 $P$ 和 $Q$ 的对数商的平均值。KL 散度仅当概率 $P$ 和 $Q$ 各自总和均为 1，且对任何 $i$ 皆满足 $Q(i)>0$ 及 $P(i)>0$ 时，才有定义。式中出现 $0\ln 0$ 的情况，其值按 $0$ 处理。

对于 连续随机变量，我们下次用到了再讨论
https://zh.wikipedia.org/wiki/相对熵

###### 特性
**相对熵** 的值为非负数：$D_{KL}(P||Q)\ge 0$
由 吉布斯不等式 可知，当且仅当 $P=Q$ 时 $D_{KL}(P||Q)$ 为零

KL 散度不是 度量或距离函数。因为KL散度具有不对称性：从分布 $P$ 到 $Q$ 的距离通常并不等于从 $Q$ 到 $P$ 的距离
$$
D_{KL} (P||Q) \ne D_{KL}(Q||P)
$$
###### KL 散度和其它量的关系
[自信息](https://zh.wikipedia.org/wiki/%E8%87%AA%E4%BF%A1%E6%81%AF "自信息")和KL散度
![{\displaystyle I(m)=D_{\mathrm {KL} }(\delta _{im}\|\{p_{i}\}),}](https://wikimedia.org/api/rest_v1/media/math/render/svg/b5259d3907dac00533fee6c90ccf30425dbaceeb)

  
[互信息](https://zh.wikipedia.org/wiki/%E4%BA%92%E4%BF%A1%E6%81%AF "互信息")和KL散度
![{\displaystyle {\begin{aligned}I(X;Y)&=D_{\mathrm {KL} }(P(X,Y)\|P(X)P(Y))\\&=\mathbb {E} _{X}\{D_{\mathrm {KL} }(P(Y|X)\|P(Y))\}\\&=\mathbb {E} _{Y}\{D_{\mathrm {KL} }(P(X|Y)\|P(X))\}\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/4f0dd25539b4734c56b6a1927ad80243023d026a)

  
[信息熵](https://zh.wikipedia.org/wiki/%E4%BF%A1%E6%81%AF%E7%86%B5 "信息熵")和KL散度
![{\displaystyle {\begin{aligned}H(X)&=\mathrm {(i)} \,\mathbb {E} _{x}\{I(x)\}\\&=\mathrm {(ii)} \log N-D_{\mathrm {KL} }(P(X)\|P_{U}(X))\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/e166092dd97aea2f0c5f669b441cb7fa6df32420)

  
[条件熵](https://zh.wikipedia.org/wiki/%E6%9D%A1%E4%BB%B6%E7%86%B5 "条件熵")和KL散度
![{\displaystyle {\begin{aligned}H(X|Y)&=\log N-D_{\mathrm {KL} }(P(X,Y)\|P_{U}(X)P(Y))\\&=\mathrm {(i)} \,\,\log N-D_{\mathrm {KL} }(P(X,Y)\|P(X)P(Y))-D_{\mathrm {KL} }(P(X)\|P_{U}(X))\\&=H(X)-I(X;Y)\\&=\mathrm {(ii)} \,\log N-\mathbb {E} _{Y}\{D_{\mathrm {KL} }(P(X|Y)\|P_{U}(X))\}\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/e148ead5812df5ef8b5482f300339f9f75e334e2)

  
[交叉熵](https://zh.wikipedia.org/wiki/%E4%BA%A4%E5%8F%89%E7%86%B5 "交叉熵")和KL散度
$$ \mathrm {H} (p,q)=\mathrm {E} _{p}[-\log q]=\mathrm {H} (p)+D_{\mathrm {KL} }(p\|q)$$
[[MLE]]