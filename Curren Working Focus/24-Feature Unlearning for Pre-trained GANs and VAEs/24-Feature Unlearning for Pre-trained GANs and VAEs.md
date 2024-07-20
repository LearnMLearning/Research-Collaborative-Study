[[VAE (Variational Autoencoders)]]

[[GAN]]

[[Feature Unlearning]]

无监督式，从GANs和VAEs中解决特征学习。
Unlearn Target: dataset 的一个subset $\times$  $\longrightarrow$  specific feature 特定的特征
如果直接拿掉图像，会导致剩余区域其他细节丢失。
![[Pasted image 20240720140400.png]]
###### 如何指定哪些特征要忘记？
当学习特征时，目标特征是微妙的 subtle，如特定发型，眼镜，微妙性使得传统监督模型学习方法难以采用

1. 框选？粗暴 $\longrightarrow$ 丢失剩余区域信息
2. Pixel-wise？成本高昂，甚至有些情况我们在学习过程中无法访问训练数据。

这篇文章首次提出了在预训练的 GANs 和 VAEs 中去除目标特征的框架。
**步骤：**
1. **收集**随机生成的包含目标特征的**图像**
2. **识别**目标特征的**潜在表示** *latent representation*
3. 使用该表示对训练模型进行 *fine-tuning*

**Latent Space Analysis**
GANs & VAEs 在低维空间内很好地保存在数据的信息，称为潜在空间 *Latent Space*。人们提出了各种遍历潜在空间和提取代表视觉特征的潜在向量的技术。这在我现阶段 Review 的 VAE 的 Tutorial 里面就有详细涉及 (通过 重参数化 reparameterization 实现降维映射)。

**前提**： 
1. 假设我们已经掌握了 **识别并提取** 与目标特征对应的潜在表征 $\mathbf z$ 的方法 (不是这篇work的重点)
2. Radford, Metz, and Chintala (2015) 提到了一种很好的方法：
	**含特征的图像的 平均特征潜向量 $\tilde {\mathbf z}_1$** 和 **不含特征的图像的 平均特征潜向量 $\tilde{\mathbf z}_2$**
	**目标视觉特征向量：**
	$$
	\mathbf z_e = \tilde{\mathbf z}_1 -\tilde{\mathbf z}_2
	$$
###### 实验数据集准备

