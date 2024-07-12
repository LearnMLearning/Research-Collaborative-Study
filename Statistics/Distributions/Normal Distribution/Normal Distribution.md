**定义：**
随机变量 $X$，两个参数 $\mu \in \mathbb R$，$\sigma>0$，是**正态分布**，记作 $X\sim \text{N}(\mu,\sigma^2)$，$\mu$ (Mean)，$\sigma$ 标准差 (std)

PDF:
$$
f_X(x;\mu,\sigma)=\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac 12\left(\frac{x-\mu}{\sigma} \right)^2} 
$$
**标准正态 Standard Normal**
当 $\mu = 0,\sigma = 1$ 时，函数成为 标准正态
PDF: 记作 $\phi(u)$
$$
\phi(u) = f_U(u) = \frac{1}{\sqrt{2\pi}}e^{-\frac 12 u^2}
$$
CDF: 记作 $\Phi(u)$
具体的数值，我们可以查表来获取
$$
\Phi(u) = \int_{-\infty}^u \frac{1}{\sqrt{2\pi}} e^{-\frac 12 s} \,ds = \mathbb P(U\le u)
$$
**非标准正态 转化 标准正态**
对于一个 普通的 正态分布 $X\sim \mathrm N(\mu,\sigma^2)$


[[Lognormal Distribution]]

[[Bivariate Normal Distribution]]
