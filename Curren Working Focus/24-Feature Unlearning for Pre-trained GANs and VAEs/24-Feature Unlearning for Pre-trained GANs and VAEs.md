在本文提出的框架中，获得**目标向量的表示-在潜在空间中表示目标特征**是至关重要的一步。

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
*1. 特征 是什么？*
1. 假设我们已经掌握了 **识别并提取** 与目标特征对应的潜在表征 $\mathbf z$ 的方法 (不是这篇work的重点)
2. Radford, Metz, and Chintala (2015) 提到了一种很好的方法：
	**含特征的图像的 平均特征潜向量 $\tilde {\mathbf z}_1$** 和 **不含特征的图像的 平均特征潜向量 $\tilde{\mathbf z}_2$**
	**目标视觉特征向量 (表示潜在空间中的目标特征)：**
	$$
	\mathbf z_e = \tilde{\mathbf z}_1 -\tilde{\mathbf z}_2
	$$
3. 假设一旦训练完成，由于各种原因，例如有限的存储或隐私问题，训练数据集不可访问。

*2. 特征 是否包含？*
White(2016) 表明，**投影**可以表示潜在向量与目标特征之间的相似性。将其潜在向量投影到目标向量。对于实验，我们将阈值 $t$ 设置为正、负样本在潜在空间中的**平均投影值**。
令 $\mathrm{sim}(\mathbf z, \mathbf z_e)\in\{0,1\}$ 表示二值分类结果，即
$$
\mathrm{sim}(\mathbf z,\mathbf z_e) = 
\begin{cases}
0, & \mathrm{if} \,\mathrm{proj}_{\mathbf z_e}(\mathbf z) < t,\\
1, & \mathrm{otherwise},
\end{cases}
$$
$$
\mathrm{proj_{\mathbf z_e}}(\mathbf z) = \frac{\mathbf z_e^{\mathrm T}\mathbf z}{\|\mathbf z_e\|}, t = \frac{1}{N}\sum_{i=1}^N[\mathrm{proj}_{\mathbf z_e}(\mathbf z_i)]
$$

###### 实验数据集准备
特征学习旨在从预训练的生成模型中删除特定特征：
	在 *CelebA* 数据集训练的生成模型中去除微笑特征后，模型永远不会生成微笑的人图像。

用远程监督原则principle of distance supervision (Mintz et al. 2009) 收集数据集
	“正” ：包含要删除特征
	”负“ ：不包含
该作者开发了用户界面
![[Pasted image 20240720143442.png]]
###### 框架
![[Pasted image 20240720143655.png]]

###### Unlearning Process
$g_\theta$ 作为需要被unlearned 的model，先从 pre-trained generator $f$ 初始化得到。

$\mathrm{sim}(\mathbf z,\mathbf z_e) = 0$: $g_{\theta}$ 应该给出 $f$ 相同的 output
	制定 *reconstruction* objective 重建目标以最小化输出变化
	$$
	\mathcal L_{\mathrm{recon}} (\theta) = (1-\mathrm{sim}(\mathbf z,\mathbf z_e))\|g_{\theta} (\mathbf z) - f(\mathbf z)\|_1
	$$
	$\mathbf z$ 是随机向量。因此，当 *latent vector* 不包含目标特征时，$g_{\theta}$ 试图模仿 $f$。

$\mathrm{sim}(\mathbf z,\mathbf z_e)=1$: 需要改变生成过程，实现 unlearn
	给定随机向量 $\mathbf z$，将该向量投影到目标向量 $\mathbf z_e$ 上，然后将原始随机向量投影到投影向量上
	$$
	\mathbf z-(\|\mathrm {proj}_{\mathbf z_e} (\mathbf z)\|-t)\mathbf z_e
	$$
	其中 $t$ 是预定义的阈值 (Recall: $t = \frac{1}{N}\sum_{i=1}^N[\mathrm{proj}_{\mathbf z_e}(\mathbf z_i)]$)。转换后的向量用作产生目标擦除输出的原始生成器的输入。然后
	使用修改后的输出来训练 $g$，使其具有以下的 unlearning 目标
	$$
	\mathcal L_{\mathrm{unlearn}}(\theta) = \mathrm{sim} (\mathbf z,\mathbf z_e) \|g_{\theta}(\mathbf z) - f(\mathbf z-(\mathrm{proj_{\mathbf z_e}(\mathbf z)-t})\mathbf z_e) \|_1
	$$
$^1$ **将多层潜在变量连接起来以应用Unlearn算法。**

![[Pasted image 20240720153929.png]]

强制 $g_{\theta}(\cdot)$ 产生与 $f(\cdot)$ 相似的输出，去除 target feature 。如果投影能够正确地测量 latent space 中target feature 的存在，同时解除其他特征的纠缠，则 $g_{\theta}$ 可以成功地忘记 latent space 中的 target feature。

众所周知，$L_2$ 和 $L_1$ loss 发生在图像生成 image generation 和 恢复任务 restorations tasks 的模糊效果中(Pathak et al. 2016;Zhang, Isola, and Efros 2016;Isola et al. 2017;Zhao et al. 2016)。先前的研究通过引入不同的技术来解决 **模糊效应** *blurry effects*，例如在训练过程中添加 感知 *perceptual* 或 对抗 *adversarial* loss (Johnson, Alahi, and Fei-Fei 2016;Zhao et al. 2016)。为了克服模糊效应，我们在目标函数中加入了**感知损失** (*[[Perceptual Loss]]*)。
$$
\mathcal L_{\mathrm{percep}}(\theta) = \mathrm{sim} (\mathbf z,\mathbf z_e) (1-\text{MS-SSIM} (g_{\theta}(\mathbf z),f(\mathbf z - (\mathrm{proj}_{\mathbf z_e}(\mathbf z) -t)\mathbf z_e)))
$$
其中 MS-SSIM 函数指的是多尺度结构相似度(Zhao et al. 2016)，它通过比较亮度、对比度和结构信息来衡量两幅图像之间的感知相似度。

最后合并
$$
\mathcal L(\theta) = \alpha (\mathcal L_{\mathrm{unlearn}}(\theta) + \mathcal L_{\mathrm{percep}}(\theta)) + \mathcal L_{\mathrm{recon}}(\theta)
$$
$\alpha$ 是调节 unlearning 和 reconstruction 误差平衡的超参数。


#### 实验
![[Pasted image 20240720162121.png]]

从每个数据集中选择两个特征，每个特征约占数据集的10%。非二值化特征 选取 前 10 % 作为正数据集。
**Training details**: Use Adam optimizer (Adaptive momentum)
###### Evaluating Metric
1. How well the **unlearning is done**
2. How good the **sample qualities** are 常用指标 (IS|high) (FID|low) Studio GAN
 
###### 用户研究
我们进行了一项用户研究，以评估我们的遗忘框架在更现实的情况下的有效性。本研究旨在考察未学习模型与原始模型在不同维度上的表现。我们招募了13名参与者，并要求他们选择包含“眼镜”的图片，因为这个功能很容易被用户识别。每个参与者使用图2所示的界面对500个样本进行注释。注释过程平均花费大约5分钟。然后，我们对每个参与者在FFHQ上训练的预训练StyleGAN进行学习。用户研究的详细信息和截图在章节中提供。
![[Pasted image 20240720163534.png]]

#### Thinking
###### Loss Function
![[Pasted image 20240720160842.png]]
###### More Datasets?

####  Adversarial Attack


[[VAE (Variational Autoencoders)]]

[[GAN]]

[[Feature Unlearning]]
