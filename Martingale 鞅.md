**鞅**
已知过去某一 时刻 $s$ 以及之前所有时刻的观测值，若某一时刻 $t$ 的观测值的 **条件期望** 等于过去某一时刻 $s$ 的观测值，则称这一随机过程是鞅。

**离散时间鞅**
$$
E(|X_n|) < \infty
$$
$$
E(X_{n+1} | X_1,\dots,X_n) = X_n, n \in \mathbb N
$$

**上鞅**满足
$$
E[X_{n+1}|X_1,\cdots,X_n] \le X_n, n \in \mathbb N
$$
$(M_k)_{k=0}^n$ 非负上鞅，对于 $\forall \lambda > 0$，
$$
P \left(\max_{1\le k \le n} M_k \ge \lambda \right) \le \frac{E[M_n]}{\lambda}
$$
