在 统计学 和 概率论 中，**协方差矩阵** 是一个 方针，代表着任两列 **随机变量** 间的 **协方差**，是 **协方差** 的直接推广。

###### 定义
设 $(\Omega,\Sigma,P)$ 是概率空间，$X = \{x_i\}_{i=1}^m$ 与 $Y = \{y_i\}_{j=1}^n$ 是定义在 $\Omega$ 上的两列实数随机变量序列
若二者对应的期望分别为：
$$\begin{aligned}
E(x_i) &= \int_\Omega x_i \, d P = \mu_i\\
E(y_i) &= \int_\Omega y_j \, d P = \nu_j
\end{aligned}$$
则这两列随机变量间的**协方差矩阵**为一个 $m\times n$ 矩阵
$$
\mathbf {cov}(X,Y) := \begin{bmatrix}
\mathrm{cov}(x_1,y_1) & \dots & \mathrm{\mathrm{cov}(x_1,y_1)}\\ 
\vdots & \ddots & \vdots\\
\mathrm{cov}(x_m,y_1) & \dots & \mathrm{cov}(x_m,y_n)
\end{bmatrix} = 
\begin{bmatrix}
E(x_1y_1) - \mu_1\nu_1 & \dots & E(x_1y_n) - \mu_1 \nu_1 \\
\vdots & \ddots & \vdots \\
E(x_my_1) - \mu_m \nu_1 & \dots & E(x_my_n) - \mu_m \nu_n
\end{bmatrix}
$$
根据 测度积分 的线性性质，协方差矩阵还可以进一步化简为：
$$
\mathbf{cov} (X,Y) = [E(x_iy_j) - \mu_i\nu_j]_{m\times n}
$$
###### 矩阵表示法
以上定义所述的随机变量序列 $X$ 和 $Y$，也可以用行向量 $\mathbf X :=[x_i]_m$ 与 $\mathbf Y := [y_j]_n$ 表示，换句话说：
$$
\mathbf X:=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_m
\end{bmatrix},\mathbf Y:=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_n
\end{bmatrix}
$$
这样的话，对于 $m\times n$ 个定义在 $\Omega$ 上的随机变量 $a_{ij}$ 所组成的矩阵 $\mathbf A = [a_{ij}]_{m\times n}$，定义：
$$
E[\mathbf A] : = [E(a_{ij})]_{m\times n}
$$
那上小节定义的协方差矩阵就可以记为：
$$
\mathbf{cov}(\mathbf X, \mathbf Y) = E \left[(\mathbf X-E[\mathbf X])(\mathbf Y-E[\mathbf Y])^{\mathrm T} \right]
$$
所以协方差矩阵也可对 $\mathbf X$ 与 $\mathbf Y$ 来定义。

###### 术语与符号分歧
也有人把以下的 $\mathbf \Sigma_X$ 称为协方差矩阵：
$$
\mathbf \Sigma_X : = [\mathrm{cov}(x_i,x_j)_{m\times m}] = \mathrm{cov}(X,X)
$$
但这里沿用 威廉·费勒 的说法，把 $\mathbf \Sigma_X$ 称为 $X$ 的 **方差**，来跟 $\mathbf{cov}(X,Y)$。这是因为：
$$
\mathrm{cov} (x_i,x_i) =E[(x_i-\mu_i)^2] = \mathrm{var} (x_i)
$$
换句话说，$\mathrm \Sigma_X$ 的对角线由随机变量 $x_i$ 的方差所组成。据此，也有人把 $\mathbf {cov}(X,Y)$ 称为 **方差-协方差矩阵**。

更有人因为方差和离差的相关性，含混的将 $\mathbf {cov}(X,Y)$ 称为离差矩阵。

###### 性质
$\mathbf \Sigma = \mathbf {cov} (X,X)$ 有以下的基本性质：
1. $\mathbf \Sigma = E(\mathbf X \mathbf X^{\mathrm T}) - E(\mathbf X) [E(\mathbf X)]^{\mathrm T}$
2. $\mathbf \Sigma$是半正定的和对称的矩阵。
3. $\mathrm{var} (\mathbf a^{\mathrm T}\mathbf X) = \mathbf a^{\mathrm T} \mathrm{var} (\mathbf X) \mathbf a$
4. $\mathbf \Sigma \ge 0$
5. 