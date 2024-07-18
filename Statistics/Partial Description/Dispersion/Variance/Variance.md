#### 数学期望
数学期望 $E[Y]$ 是函数 $Y=g(X)$ 根据概率 的 加权平均
$$
E[Y] = E[g(X)] = 
\begin{cases}
\sum_x g(x) p_X(x) & \text{if} \, X \text{ is discrete}\\
\int_{-\infty}^{+\infty} g(x) f_X(x) \, dx & \text{if } X \text{ is continuous}
\end{cases}
$$
**性质：**
1. $E[a]=a$，$a$ 是一个常数
2. $E[a\cdot g(X)] = a \cdot E[g(X)]$，$a$ 是一个常数
3. $E[g_1(X) + g_2(X)] = E[g_1(X)] + E[g_2(X)]$
**数学期望是一个线性操作**
###### 推论：
1. $\frac{d}{d\theta} E[g(X,\theta)] = E[\frac{d}{d\theta}g(X,\theta)]$
2. 如果 $g(X) = g_1(X) g_2(X)$，且 $X_1$ 与 $X_2$ 是 independent 那么 $E[g(X)] = E[g_1(X_1)g_2(X_2)] =E[g_1(X_1)]E[g_2(X_2)]$

#### Variance 方差
$$
\mathrm {Var} [X] = 
\begin{cases}
\sum_{x} (x-\mu_X)^2 p_X(x) & \mathrm{if} \, X \, \text{is discrete}\\
\int_{-\infty}^{\infty} (x-\mu_X)^2 f_X(x)\,dx & \text{if} \, X \, \text{is continuous}
\end{cases}
$$
是 PMF 和 PDF 图像的 second central moment
**性质：**
1. $\mathrm{Var}[a] = 0$，where $a$ is a constant
2. $\mathrm{Var}[a_1X+a_2]=a_1^2\mathrm{Var}[X]$，where $a_1,a_2$ are constants
3. $\mathrm{Var}[X_1+X_2] = \mathrm{Var}[X_1] + \mathrm{Var}[X_2]$，**如果** $X_1 \perp X_2$ (独立事件 方差计算)
4. $\mathrm{Var}[X] = E[X^2] - E[X]^2$ (简便计算)
#### Standard Deviation (std) 标准差
$$
\sigma_X = \sqrt{\mathrm{Var}[X]}
$$
#### Coefficient of Variation 方差系数
没有量纲，归一化，用百分比表示，$\mu_X$ 为 0 的时候没有意义
$$
\delta_X = \frac{\sigma_X}{|\mu_X|}
$$
[[Covariance]]
