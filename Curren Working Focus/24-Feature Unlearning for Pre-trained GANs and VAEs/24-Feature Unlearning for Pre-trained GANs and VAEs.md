[[VAE (Variational Autoencoders)]]

[[GAN]]

[[Feature Unlearning]]

无监督式，从GANs和VAEs中解决特征学习。
Unlearn Target: dataset 的一个subset $\times$  $\longrightarrow$  specific feature 特定的特征
如果直接拿掉图像，会导致剩余区域其他细节丢失。
![[Pasted image 20240720140400.png]]
###### 如何指定哪些特征要忘记？


识别与目标特征对应的潜在表征 $\mathbf z$

