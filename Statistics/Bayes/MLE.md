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

