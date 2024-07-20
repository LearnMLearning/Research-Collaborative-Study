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
其中 $p_H$ 为我们所要确定的参数。所以当我们投掷硬币三次，其中两次是正面，则pH=0.5![{\displaystyle p_{H}=0.5}](https://wikimedia.org/api/rest_v1/media/math/render/svg/24b52f82140b64ef1559864293ef266ebb0500cd)的似然性是L(pH=0.5∣A)=P(A∣pH=0.5)=0.375![{\displaystyle L(p_{H}=0.5\mid {\mbox{A}})=P({\mbox{A}}\mid p_{H}=0.5)=0.375}](https://wikimedia.org/api/rest_v1/media/math/render/svg/673e50d143c223e28a239a455c780ccfe7bf333b)，而pH=0.6![{\displaystyle p_{H}=0.6}](https://wikimedia.org/api/rest_v1/media/math/render/svg/f9457df537bf2ee65cbea29efd87848395c1a167)的似然性是L(pH=0.6∣A)=P(A∣pH=0.6)=0.432![{\displaystyle L(p_{H}=0.6\mid {\mbox{A}})=P({\mbox{A}}\mid p_{H}=0.6)=0.432}](https://wikimedia.org/api/rest_v1/media/math/render/svg/e673c407c7d283d29f8dbed823fcbecef83d4f49)。注意，L(pH=0.5∣A)=0.375![{\displaystyle L(p_{H}=0.5\mid {\mbox{A}})=0.375}](https://wikimedia.org/api/rest_v1/media/math/render/svg/45dfba76d649bb6553b883daf7ca298cbada34a0)并不是说当已知A![{\displaystyle {\mbox{A}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/5ae904df569a1bc943ae9533602f29d39899c107)发生了，则pH![{\displaystyle p_{H}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/249c90c847d6a2931229be04965e2395f7b0b297)为0.5![{\displaystyle 0.5}](https://wikimedia.org/api/rest_v1/media/math/render/svg/c867fe7d5d53ce2c0790852289b794c6ed185f36)的概率是0.375![{\displaystyle 0.375}](https://wikimedia.org/api/rest_v1/media/math/render/svg/7b71aa684690d215fe9044c62feac308764698fc)。_似然性跟概率具有不同的意义。_

若单独看0.375![{\displaystyle 0.375}](https://wikimedia.org/api/rest_v1/media/math/render/svg/7b71aa684690d215fe9044c62feac308764698fc)这个数字或0.432![{\displaystyle 0.432}](https://wikimedia.org/api/rest_v1/media/math/render/svg/d33ef0024e03588e79f416da5790ae42d4a5158b)这个数字是没有意义的，因为似然性并不是概率，并不是一定介于0![{\displaystyle 0}](https://wikimedia.org/api/rest_v1/media/math/render/svg/2aae8864a3c1fec9585261791a809ddec1489950)到1![{\displaystyle 1}](https://wikimedia.org/api/rest_v1/media/math/render/svg/92d98b82a3778f043108d4e20960a9193df57cbf)之间，而所有可能的pH![{\displaystyle p_{H}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/249c90c847d6a2931229be04965e2395f7b0b297)的似然性加起来也不是1![{\displaystyle 1}](https://wikimedia.org/api/rest_v1/media/math/render/svg/92d98b82a3778f043108d4e20960a9193df57cbf)，所以单独得知L(pH=0.5∣A)=0.375![{\displaystyle L(p_{H}=0.5\mid {\mbox{A}})=0.375}](https://wikimedia.org/api/rest_v1/media/math/render/svg/45dfba76d649bb6553b883daf7ca298cbada34a0)是没有意义的。似然性是用在把各种可能的pH![{\displaystyle p_{H}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/249c90c847d6a2931229be04965e2395f7b0b297)值放在一起比较，来得知哪个pH![{\displaystyle p_{H}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/249c90c847d6a2931229be04965e2395f7b0b297)值的可能性比较高。而似然函数（在这个例子中，即L(pH∣A)=3×pH2×(1−pH)![{\displaystyle L(p_{H}\mid {\mbox{A}})=3\times p_{H}^{2}\times (1-p_{H})}](https://wikimedia.org/api/rest_v1/media/math/render/svg/a1fe809e59c1f77ccb5c2ba49969185c7c6a7c7a)），除了用来计算似然性外，则是用来了解当参数pH![{\displaystyle p_{H}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/249c90c847d6a2931229be04965e2395f7b0b297)改变时，似然性怎么变化，用来寻找最大可能性的pH![{\displaystyle p_{H}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/249c90c847d6a2931229be04965e2395f7b0b297)值会是多少。