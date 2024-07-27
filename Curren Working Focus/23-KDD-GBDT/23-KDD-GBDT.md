###### Challenges
1. **Differentiability**
	Gradient-based methods in DNN require computing the gradient of the deleted data while GBDT is not differentiable.
2. **Fine-tuning**
	DNN: not increase parameter size
	GBDT: more tree, more parameters, high inference costs
3. **Tree generation**
	RandF: independently
	GBDT: depend on the residual of the trees of all previous iterations

