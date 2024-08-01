![[Pasted image 20240801221517.png]]
## 1 卷积神经网络的模型

![[Pasted image 20240801221536.png]]
#### 1.1 背景
![[Pasted image 20240801221550.png]]
![[Pasted image 20240801221601.png]]


#### 1.2 卷积
###### 1.2.1 数学卷积
在数学中，卷积 (convolution) 是定义在两个函数上的运算，表示用其中一个函数对另一个函数的形状进行的调整。这里考虑一维卷积。设 $f$ 和 $g$ 是两个可积的实值函数，则积分
$$
\int_{-\infty}^{\infty} f (\tau) g(t-\tau) \, d\tau
$$
定义了一个新的函数 $h(t)$，称为 $f$ 和 $g$ 的卷积，记作
$$
h(t) = (f \circledast g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t-\tau) \, \mathrm d\tau
$$
其中，符号 $\circledast$ 表示卷积运算。

根据定义可知，卷积满足交换律 $(f \circledast g)(t) = (g \circledast f)(t)$，即有
$$
(f \circledast g)(t) = \int_{-\infty}^{\infty} f(t-\tau) g(\tau) \, \mathrm d \tau
$$
**例 24.1** 以下是数学卷积的例子
$$
y(t) = (x \circledast w) (t) =\int_{-\infty}^{\infty} x(\tau) w(t-\tau) \, \mathrm d \tau
$$
其中，$x(\tau)$ 是任意给定函数，$w(t)$ 是高斯核函数。
$$
w(t) = \frac{1}{\sqrt{2\pi}\sigma} \exp \left(-\frac{t^2}{2\sigma^2} \right)
$$
卷积表示用高斯核函数 $w(t)$ 对给定函数 $x(\tau)$ 进行平滑得到的结果 (见图 24.1)
数学卷积也可以自然地扩展到二维和离散的情况。具体例子参见习题。
###### 1.2.2 二维卷积
卷积神经网络中的卷积与数学卷积并不相同，实际是数学的互相关 (cross correlation)
![[Pasted image 20240801221742.png]]
两个实值函数 $f$ 和 $g$ 的互相关是指
$$
(f*g) (t) = \int_{-\infty}^{+\infty} f(\tau) g(t+\tau) \, \mathrm d\tau
$$
式中记号 $*$ 表示互相关运算。互相关不满足交换律 $(f*g)(t)\ne (g*f)(t)$。可以将以上互相关自然地扩展到二维和离散的情况。
卷积神经网络中的卷积一般为二维线性互相关，用矩阵形式表示。本书称之为机器学习卷积。

**定义 24.1 (二维卷积)** 给定一个 $I\times J$ 输入矩阵 $\mathbf X = [x_{ij}]_{I\times J}$，一个 $M\times N$ 核矩阵 $\mathbf W = [w_{mn}]_{M\times N}$，满足 $M \ll I, N \ll J$。让核矩阵在输入矩阵上从左到右再从上到下按顺序滑动，在滑动的每一个位置，核矩阵与输入矩阵的一个子矩阵重叠。求核矩阵与每一个子矩阵的内积，产生一个 $K\times L$ 输出矩阵 $\mathbf Y = [y_{kl}]_{K\times L}$，称此运算为卷积 (convolution) 或二位卷积。写作
$$
\mathbf Y = \mathbf W * \mathbf X
$$
其中，$\mathbf Y = [y_{kl}]_{K\times L}$。
$$
y_{kl} = \sum_{m=1}^M \sum_{n=1}^N w_{m,n}x_{k+m-1,l+n-1}
$$
其中，$k=1,2,\cdots,K,l=1,2,\cdots,L,K=I-M+1,L=J-N+1$。

![[Pasted image 20240801222802.png]]
**例 24.2** 

![[Pasted image 20240801222827.png]]
![[Pasted image 20240801222846.png]]
![[Pasted image 20240801222907.png]]
###### 1.2.3 填充和步幅
![[Pasted image 20240801222950.png]]
![[Pasted image 20240801223003.png]]
![[Pasted image 20240801223017.png]]
![[Pasted image 20240801223029.png]]
![[Pasted image 20240801223041.png]]
![[Pasted image 20240801223052.png]]
![[Pasted image 20240801223103.png]]
###### 1.2.4 三维卷积
![[Pasted image 20240801223139.png]]
![[Pasted image 20240801223150.png]]
![[Pasted image 20240801223159.png]]
![[Pasted image 20240801223210.png]]
![[Pasted image 20240801223219.png]]
![[Pasted image 20240801223228.png]]
![[Pasted image 20240801223243.png]]


#### 1.3 汇聚
卷积神经网络还是用汇聚 (pooling) 运算。
###### 1.3.1 二维汇聚
![[Pasted image 20240801223533.png]]
![[Pasted image 20240801223548.png]]
![[Pasted image 20240801223556.png]]
![[Pasted image 20240801223604.png]]
![[Pasted image 20240801223616.png]]
###### 1.3.2 三维汇聚
![[Pasted image 20240801223703.png]]
![[Pasted image 20240801223714.png]]
![[Pasted image 20240801223723.png]]
#### 1.4 卷积神经网络
###### 1.4.1 模型定义

###### 1.4.2 模型例子


#### 1.5 卷积神经网络性质

###### 1.5.1 表示效率


###### 1.5.2 不变性


###### 1.5.3 感受野

## 2 卷积神经网络的学习算法
![[Pasted image 20240801224106.png]]

#### 2.1 卷积导数
![[Pasted image 20240801224153.png]]
![[Pasted image 20240801224206.png]]


#### 2.2 反向传播算法
![[Pasted image 20240801224223.png]]
###### 2.2.1 卷积层
![[Pasted image 20240801224250.png]]
![[Pasted image 20240801224310.png]]
###### 2.2.2 汇聚层
![[Pasted image 20240801224333.png]]
![[Pasted image 20240801224344.png]]
###### 2.2.3 算法
![[Pasted image 20240801224406.png]]
![[Pasted image 20240801224421.png]]
![[Pasted image 20240801224433.png]]



## 3 图像分类中的应用
![[Pasted image 20240801224901.png]]

#### 3.1 AlexNet
![[Pasted image 20240801224926.png]]
![[Pasted image 20240801224940.png]]


#### 3.2 残差网络 ResNet
###### 3.2.1 基本想法
![[Pasted image 20240801224956.png]]
![[Pasted image 20240801225008.png]]
###### 3.2.2 模型架构
![[Pasted image 20240801225041.png]]
![[Pasted image 20240801225050.png]]
![[Pasted image 20240801225103.png]]
![[Pasted image 20240801225115.png]]
![[Pasted image 20240801225131.png]]
![[Pasted image 20240801225140.png]]
###### 3.2.3 模型特点
![[Pasted image 20240801225202.png]]
![[Pasted image 20240801225212.png]]
![[Pasted image 20240801225224.png]]
###### 3.2.4 图像分类
![[Pasted image 20240801225245.png]]
![[Pasted image 20240801225255.png]]
![[Pasted image 20240801225304.png]]
![[Pasted image 20240801225314.png]]
![[Pasted image 20240801225324.png]]
![[Pasted image 20240801225342.png]]
![[Pasted image 20240801225352.png]]
![[Pasted image 20240801225400.png]]

