两个随机变量 $X$ 和 $Y$ 的 Mean 分别为 $\mu_X$ 和 $\mu_Y$，他们的 Covariance 为：
$$
\mathrm{Cov}[X,Y] = 
\begin{cases}
\sum_x \sum_y (x-\mu_X) (y-\mu_Y) p_{XY} (x,y) & \text{for discrete}\\
\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} (x-\mu_X)(y-\mu_Y) f_{XY} (x,y) \, dx dy &\text{for continuous}
\end{cases}
$$
**性质：**
1. $\mathrm{Cov}[X,Y] = E[XY]-E[X]E[Y]$
	**证明**
	$$\begin{aligned}
	\mathrm{Cov} (X,Y) &= E[(X-E[X])(Y-E[Y])]\\
	&= E[XY-XE[Y]-YE[X] + E[X]E[Y]]\\
	&= E[XY] - E[XE[Y]] - E[YE[X]] + E[E[X]E[Y]]\\
	&= E[XY] - 2E[X]E[Y] + E[X]E[Y]\\
	&= E[XY]-E[X]E[Y]
	\end{aligned}$$ 
2. $\mathrm{Cov}[X,Y] = 0$，如果 $X \perp Y$ **定理**
3. $\mathrm{Cov}[X,Y] = \mathrm{Cov}[Y,X]$ 
4. $\mathrm{Cov}[aX,bY] = ab\mathrm{Cov}[X,Y]$，$a,b$ 是常数
5. $\mathrm{Cov}[\sum_{i=1}^n X_i,\sum_{j=1}^mY_j]=\sum_{i=1}^n \sum_{j=1}^m \mathrm{Cov}[X_i,Y_j]$
6. $\mathrm{Cov}[X,X] = \mathrm{Var}[X]$
7. $\mathrm{Var} [X+Y] = \mathrm{Var}[X] + \mathrm{Var}[Y] + 2\mathrm{Cov}[X,Y]$


[[Covariance Matrix]]
