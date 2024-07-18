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
