活动自动识别是一个活跃的研究课题，其目的是对人类活动进行自动识别。一个重要的挑战是有效地认识新的活动。在本文中，我们提出了一种有效的类增量学习方法，称为类增量随机森林 Class Incremental Random Forests (CIRF)，使现有的活动识别模型能够识别新的活动。我们设计了一种基于分离轴定理的分裂策略，插入内部节点，并采用基尼指数或信息增益对决策树的随机森林(RF)叶子进行分裂。使用这两种策略，在增量学习阶段都允许插入新节点和分裂叶子。我们在三个UCI公共活动数据集上评估了我们的方法，并与其他最先进的方法进行了比较。实验结果表明，所提出的增量学习方法收敛于 batch learning methods 批学习方法(RF和极端随机树)的性能。与其他先进的方法相比，它能够连续识别新的类数据，并且性能更好。
