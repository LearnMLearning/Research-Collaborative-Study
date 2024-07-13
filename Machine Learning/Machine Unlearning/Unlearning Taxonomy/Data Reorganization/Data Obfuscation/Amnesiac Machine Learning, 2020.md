https://arxiv.org/abs/2010.10981


$E$ epochs, $B$ batches, $\Delta \theta_{e,b}$
$$
\theta_M = \theta_{initial} + \sum_{e=1}^E\sum_{b=1}^B \Delta \theta_{e,b}
$$
$sb \in SB$ (the list of sensitive data batches)
$$
\theta_{M'} = \theta_{initial} + \sum_{e=1}^E \sum_{b=1}^B \Delta \theta_{e,b} - \sum_{sb=1}^{SB} \Delta \theta_{sb} = \theta_M - \sum_{sb=1}^{SB} \Delta \theta_{sb}
$$
**潜在缺点** 是需要大量的存储空间来保存来自每个批处理的一组参数更新值。虽然这个成本可能相当大，特别是对于最先进的**大型模型**，但**大容量存储的低成本**意味着这个成本通常低于从头开始重新训练完整模型的成本。**关心这种存储成本的模型所有者**最好使用不需要这种**存储开销**的方法，比如 unlearning。

[[Model Inversion Attacks 2015]]

[[MIA (Membership Inference Attacks) 2017]]

