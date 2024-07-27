###### Challenges
1. **Differentiability**
	Gradient-based methods in DNN require computing the gradient of the deleted data while GBDT is not differentiable.
2. **Fine-tuning**
	DNN: not increase parameter size
	GBDT: more tree, more parameters, high inference costs
3. **Tree generation**
	RandF: independently
	GBDT: depend on the residual of the trees of all previous iterations

###### Baseline
Retrain

#### Training
$D_{tr} = \{y_i,\mathbf x_i\}_{i=1}^N,N$ is the number of training instances.
For the $i^{th}$ instance: $\mathbf x_i$ is the feature vector
$y_i \in \{1,2,\dots,K\}$
$$
p_{i,k} = \mathbf {Pr}(y_i = k|\mathbf x_i)=\frac{e^{F_{i,k}(\mathbf x_i)}}{\sum_{s=1}^Ke^{F_{i,s}(\mathbf x_i)}},i = 1,2,\dots,N
$$
![[Pasted image 20240727090320.png]]
![[Pasted image 20240727090335.png]]
###### Robust LogitBoost
![[Pasted image 20240727090343.png]]
![[Pasted image 20240727090411.png]]

###### MART
![[Pasted image 20240727090420.png]]
![[Pasted image 20240727090435.png]]
###### Inference
$K\times M$ regression tree $f_{i,k}$
![[Pasted image 20240727090703.png]]
[[softmax]]
![[Pasted image 20240727090850.png]]

###### Discretize features into integers as a preprocessing
![[Pasted image 20240727091026.png]]
PingLi, Christopher J.C.Burges, and Qiang Wu. Mcrank: Learning to rank using multiple classification and gradient boosting. In Advances in Neural Information Processing Systems (NIPS), pages 897–904, Vancouver, Canada, 2007.

Ping Li and Weijie Zhao. Package for fast abc-boost. CoRR, abs/2207.08770, 2022.

#### Unlearning
###### Unlearning in GBDT
![[Pasted image 20240727091517.png]]
###### Unlearning in one tree
![[Pasted image 20240727091624.png]]

![[Pasted image 20240727091636.png]]
Thinking of HedgeCut
###### Time Complexity
![[Pasted image 20240727092247.png]]

#### Incremental Update
###### Split Gain

###### Derivatives

