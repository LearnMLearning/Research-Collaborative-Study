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
以上是基本卷积的定义，还有多种扩展。注意式 $\mathbf Y = \mathbf W * \mathbf X$ 中的卷积符号是 $*$，$\mathbf X$ 和 $\mathbf W$ 的顺序是有意义的，本书将卷积核矩阵放在前面。卷积核又被称为滤波器 (filter)。

比较定义式 $h(t) = (f \circledast g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t-\tau) \, \mathrm d\tau$ 和式 $(f*g) (t) = \int_{-\infty}^{+\infty} f(\tau) g(t+\tau) \, \mathrm d\tau$ 可知数学的卷积和互相关并不等价。卷积神经网络采用互相关作为“卷积”，主要是为了处理方便。如果数学的卷积和互相关的和矩阵都是从数据中学到的，那么效果是一样的。本书中的卷积除特别声明外均指互相关。

**例 24.2** 给定输入矩阵 $\mathbf X$ 和核矩阵 $\mathbf W$：
$$
\mathbf X = 
\begin{bmatrix}
3 & 2 & 0 & 1\\
0 & 2 & 1 & 2\\
2 & 0 & 0 & 3\\
2 & 3 & 1 & 2
\end{bmatrix}, \mathbf W = 
\begin{bmatrix}
2 & 1 & 2\\
0 & 0 & 3\\
0 & 0 & 2
\end{bmatrix}
$$
求卷积 $\mathbf Y = \mathbf W * \mathbf X$。
**解** $\mathbf W$ 作用在 $\mathbf X$ 上，并不超出 $\mathbf X$ 的范围。按照式 $y_{kl} = \sum_{m=1}^M \sum_{n=1}^N w_{m,n}x_{k+m-1,l+n-1}$ ，计算
$$
y_{11} = \sum_{m=1}^3 \sum_{n=1}^3 w_{mn} x_{mn} = 11
$$
$$
y_{12} =\sum_{m=1}^3 \sum_{n=1}^3 w_{mn}x_{m,n+1} = 18
$$
同样可计算 $y_{21},y_{22}$，得到输出矩阵 $\mathbf Y$：
$$
\mathbf Y = \begin{bmatrix}
2 & 1 & 2\\
0 & 0 & 3\\
0 & 0 & 2
\end{bmatrix} * \begin{bmatrix}
3 & 2 & 0 & 1\\
0 & 2 & 1 & 2\\
2 & 0 & 0 & 3\\
2 & 3 & 1 & 2
\end{bmatrix} = \begin{bmatrix}11 & 18 \\ 6 & 22 \end{bmatrix}
$$
输入矩阵是 $4\times 4$ 矩阵，核矩阵是 $3\times 3$ 矩阵，输出矩阵是 $2\times 2$ 矩阵。图24.2 显示这个卷积计算的过程。
![[Pasted image 20240802150246.png]]

###### 1.2.3 填充和步幅
卷积运算的扩展可以通过增加填充和步幅实现。在输入矩阵的周边添加元素为 0 的行和列，使卷积核能更充分地作用于输入矩阵边缘的元素，这样的处理称为填充 (padding) 或 零填充 (zero padding)。 下面是含有填充的卷积运算的例子。

**例 24.3** 对 例 24.2 的输入矩阵进行填充，得到矩阵
$$
\hat {\mathbf X} = 
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0\\
0 & 3 & 2 & 0 & 1 & 0\\
0 & 0 & 2 & 1 & 2 & 0\\
0 & 2 & 0 & 0 & 3 & 0\\
0 & 2 & 3 & 1 & 2 & 0\\
0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$
核矩阵 $\mathbf W$ 不变，求卷积 $\mathbf Y = \mathbf W * \hat{\mathbf X}$。

**解** $\mathbf W$ 作用在 0 填充后的 $\mathbf X$ 上。按照式 $y_{kl} = \sum_{m=1}^M \sum_{n=1}^N w_{m,n}x_{k+m-1,l+n-1}$ 可以计算每一个卷积的值，得到输出矩阵 $\mathbf Y$：
$$
\mathbf Y = \begin{bmatrix}2 & 1 & 2 \\ 0 & 0 & 3 \\ 0 & 0 & 2\end{bmatrix} * \begin{bmatrix}0 & 0 & 0 & 0 & 0 & 0\\ 0 & 3 & 2 & 0 & 1 & 0 \\ 0 & 0 & 2 & 1 & 2 & 0\\ 0 & 2 & 0 & 0 & 3 & 0 \\ 0 & 2 & 3 & 1 & 2 & 0\\ 0 & 0 & 0 & 0 & 0 & 0  \end{bmatrix} = 
\begin{bmatrix}
10 & 2 & 7 & 0\\
13 & 11 & 18 & 1\\
10 & 6 & 22 & 4\\
11 & 7 & 12 & 3
\end{bmatrix}
$$
输入矩阵通过填充由 $4\times 4$ 变为 $6\times 6$ 的矩阵，核矩阵是 $3\times 3$ 矩阵，输出矩阵是 $4\times 4$ 矩阵。
图 24.3 显示这个卷积计算的一步。

![[Pasted image 20240802113717.png]]
在卷积计算中，卷积核每次向右或向下移动的列数或行数被称为步幅 (stride)。以上的卷积运算例中步幅均为 1。下面是步幅为 2 的卷积运算的例子。

**例 24.4** 给定输入矩阵 $\mathbf X$ 和核矩阵 $\mathbf W$：
$$
\mathbf X = 
\begin{bmatrix} 
3 & 2 & 0 & 1 & 0 & 2 & 1 \\
0 & 2 & 1 & 2 & 1 & 2 & 1\\
2 & 0 & 0 & 3 & 0 & 0 & 2\\
2 & 3 & 1 & 0 & 1 & 1 & 3\\
2 & 2 & 1 & 1 & 0 & 3 & 1\\
1 & 1 & 0 & 0 & 1 & 2 & 2\\
2 & 1 & 0 & 3 & 2 & 1 & 1
\end{bmatrix},
\mathbf W = 
\begin{bmatrix}
2 & 1 & 2 \\ 
0 & 0 & 3 \\ 
0 & 0 & 2 
\end{bmatrix}
$$
设卷积步幅为 2，求卷积 $\mathbf Y = \mathbf W * \mathbf X$。

**解** $\mathbf W$ 作用在 $\mathbf X$ 上，每次计算向右或向下移动两列或两行。按照式 $y_{kl} = \sum_{m=1}^M \sum_{n=1}^N w_{m,n}x_{k+m-1,l+n-1}$ 可以计算每一个卷积的值，得到输出矩阵 $\mathbf Y$：
$$
\mathbf Y  = \begin{bmatrix}
2 & 1 & 2 \\ 
0 & 0 & 3 \\ 
0 & 0 & 2 
\end{bmatrix} * \begin{bmatrix} 
3 & 2 & 0 & 1 & 0 & 2 & 1 \\
0 & 2 & 1 & 2 & 1 & 2 & 1\\
2 & 0 & 0 & 3 & 0 & 0 & 2\\
2 & 3 & 1 & 0 & 1 & 1 & 3\\
2 & 2 & 1 & 1 & 0 & 3 & 1\\
1 & 1 & 0 & 0 & 1 & 2 & 2\\
2 & 1 & 0 & 3 & 2 & 1 & 1
\end{bmatrix} = 
\begin{bmatrix}
11 & 4 & 11 \\
9 & 6 & 15\\
8 & 10 & 13
\end{bmatrix}
$$
输入矩阵是 $7\times 7$ 矩阵，核矩阵是 $3\times 3$ 矩阵，输出矩阵是 $3\times 3$ 矩阵。图 24.4 显示这个卷积计算的两步。

卷积运算依赖于卷积核的大小、填充的大小、步幅。这些是卷积运算的超参数。假设输入矩阵的大小是 $I\times J$，卷积核的大小是 $M \times N$，两个方向填充的大小是 $P$ 和 $Q$，步幅的大小是 $S$，则卷积的输出矩阵的大小 $I \times J$ 满足
$$
K \times L =\left \lfloor \frac{I + 2P - M}{S} + 1\right \rfloor\times \left \lfloor \frac{J+2Q-N}{S} + 1 \right \rfloor
$$
这里 $\lfloor a \rfloor$ 表示不超过 $a$ 的最大整数。填充 $P$ 和 $Q$ 的最大值分别是 $M-1$ 和 $N-1$，这时的填充称为全填充 (full padding)。

在图像处理中，卷积实现的是特征检测。最基本的情况是二维卷积，卷积的输入矩阵表示灰度图像，矩阵的一个元素对应图像的一个像素，代表像素上的灰度 (权重)，一个卷积核表示一个特征。卷积运算将卷积核在图像上进行滑动，在图像的每一个位置对一个特定的特征进行检测，输出一个检测值，参见图 24.2 ~ 图 24.4。当在某个位置的图像的特征和卷积核的特征一致时，检测值最大，这是因为卷积进行的是矩阵内积计算。注意在卷积神经网络中卷积核的权重是通过学习获得的，也就是说学习的是特征检测的能力。
![[Pasted image 20240801223041.png]]
卷积的输入和输出称为特征图 (feature map)。二维卷积的特征图一般是矩阵 (后面叙述特征图是张量的情况)。灰度图像的输入矩阵也可以看作是一种特殊的特征图。

**例 24.5** 给定输入矩阵 $\mathbf X$ 和 核矩阵 $\mathbf W$：
$$
\mathbf X = 
\begin{bmatrix}
0 & 0 & 0 & 0\\
0 & 2 & 0 & 0\\
0 & 2 & 0 & 0\\
0 & 2 & 2 & 2
\end{bmatrix}, \mathbf W = 
\begin{bmatrix}
2 & 0 & 0\\
2 & 0 & 0\\
2 & 2 & 2
\end{bmatrix}
$$
求卷积 $\mathbf Y = \mathbf W * \mathbf X$。
**解** 按照卷积公式计算可得：
$$
\mathbf Y = 
\begin{bmatrix}
2 & 0 & 0\\
2 & 0 & 0\\
2 & 2 & 2
\end{bmatrix} * 
\begin{bmatrix}
0 & 0 & 0 & 0\\
0 & 2 & 0 & 0\\
0 & 2 & 0 & 0\\
0 & 2 & 2 & 2
\end{bmatrix} = 
\begin{bmatrix}
4 & 8 \\
8 & 20
\end{bmatrix}
$$
图 24.5 显示卷积进行特征检测的情况。输入矩阵 $\mathbf X$ 表示一个 $4 \times 4$ 图片，取值为 $0$ 或 $2$，图片中有一个 $L$ 字。核矩阵 $\mathbf W$ 表示一个特征，取值也是 $0$ 或 $2$，也包含一个 $L$ 字。输出矩阵表示特征检测值，当卷积滑动到图片中的 $L$ 字型边时，检测值最大。
![[Pasted image 20240801223103.png]]
###### 1.2.4 三维卷积
三维卷积的输入和输出一般是由张量 (tensor) 表示的特征图 (注意矩阵表示的特征图可以看作是一张特征图，张量表示的特征图可以看作是多张特征图，本书都称为特征图)。这样的特征图有高度、宽度、深度。这里，将彩色图像数据也看作是一种特殊的特征图。

图像处理常使用彩色图像，由红、绿、蓝三个通道的数据组成。每一个通道的数据由一个矩阵表示，矩阵的每一个元素对应一个像素，代表颜色的深度。三个矩阵排列起来构成一个张量。三维卷积作用于这样的张量数据 (特征图)。彩色图像三个通道的矩阵的行数和列数是特征图的高度和宽度，也就是彩色图像看上去的高度和宽度，通道数是特征图的深度 (见图 24.6 左侧)。
![[Pasted image 20240801223150.png]]
通过卷积或汇聚运算也得到由张量表示的特征图。张量由多个大小相同的矩阵组成。矩阵的行数和列数是特征图的高度和宽度，矩阵的个数是特征图的深度 (见图 24.6 右侧)。三维卷积作用于这样的特征图。一个三维卷积的输出是一个矩阵。多个三维卷积的输出矩阵排列起来得到一个张量特征图。

下面，以彩色图像数据为例介绍三维卷积的计算方法。输入是三通道数据，用张量表示 $\mathbf X = (\mathbf X_{\mathrm R},\mathbf X_{\mathrm G}, \mathbf X_{\mathrm B})$。其中 $\mathbf X_{\mathrm R},\mathbf X_{\mathrm{R}},\mathbf X_{\mathrm B}$ 是三个通道的数据，各自用矩阵表示。卷积核也用张量表示 $\mathbf W = (\mathbf W_{\mathrm R},\mathbf W_{\mathrm G},\mathbf W_{\mathrm B})$，其中 $\mathbf W_{\mathrm R},\mathbf W_{\mathrm G},\mathbf W_{\mathrm B}$ 是三个通道的 (二维) 卷积核，也各自由矩阵表示。那么，三维卷积可以通过以下等价关系计算。
$$
\mathbf Y = \mathbf X * \mathbf W = \mathbf X_{\mathrm R} * \mathbf W_{\mathrm{R}} + \mathbf X_\mathrm G * \mathbf W_{\mathrm G} + \mathbf X_{\mathrm B} * \mathbf W_{\mathrm B}
$$
也就是说以上的三维卷积计算首先使用三个不同的二维卷积对三个通道的输入矩阵分别进行二维卷积计算，然后将得到的三个输出矩阵相加，最终得到一个三维卷积的输出矩阵。注意这时二维卷积核的个数和通道的个数相等。

**例 24.6** 输入张量由三个通道的矩阵组成 $\mathbf X = (\mathbf X_{\mathrm R},\mathbf X_{\mathrm G},\mathbf X_{\mathrm B})$，
$$
\mathbf X_{\mathrm R} = 
\begin{bmatrix}
3 & 2 & 0 & 1\\
0 & 2 & 1 & 2\\
2 & 0 & 0 & 3\\
2 & 3 & 1 & 2
\end{bmatrix},\mathbf X_{\mathrm G}=
\begin{bmatrix}
3 & 2 & 0 & 1\\
2 & 1 & 0 & 1\\
1 & 0 & 2 & 1\\
2 & 1 & 0 & 0
\end{bmatrix}, \mathbf X_{\mathrm B}=
\begin{bmatrix}
4 & 2 & 0 & 1\\
0 & 3 & 1 & 0\\
3 & 1 & 0 & 2\\
2 & 2 & 0 & 1
\end{bmatrix}
$$
卷积核张量由三个矩阵组成 $\mathbf W = (\mathbf W_{\mathrm R},\mathbf W_{\mathrm G},\mathbf W_{\mathrm B})$，
$$
\mathbf W_{\mathrm R} = 
\begin{bmatrix} 
2 & 1 & 2\\
0 & 0 & 3\\
0 & 0 & 2
\end{bmatrix}, \mathbf W_{\mathrm G} = 
\begin{bmatrix}
1 & 0 & 1\\
0 & 1 & 0\\
1 & 0 & 1
\end{bmatrix}, \mathbf W_{\mathrm B} =
\begin{bmatrix} 
1 & 0 & -1\\
1 & 0 & -1\\
1 & 0 & -1
\end{bmatrix}
$$
求在其上的三维卷积 $\mathbf Y$。

**解** 按照式 $\mathbf Y = \mathbf X * \mathbf W = \mathbf X_{\mathrm R} * \mathbf W_{\mathrm{R}} + \mathbf X_\mathrm G * \mathbf W_{\mathrm G} + \mathbf X_{\mathrm B} * \mathbf W_{\mathrm B}$ 计算，可得输出矩阵 $\mathbf Y$：
$$
\mathbf Y = 
\begin{bmatrix} 
2 & 1 & 2\\
0 & 0 & 3\\
0 & 0 & 2
\end{bmatrix} * 
\begin{bmatrix}
3 & 2 & 0 & 1\\
0 & 2 & 1 & 2\\
2 & 0 & 0 & 3\\
2 & 3 & 1 & 2
\end{bmatrix} + 
\begin{bmatrix}
1 & 0 & 1\\
0 & 1 & 0\\
1 & 0 & 1
\end{bmatrix} * 
\begin{bmatrix}
3 & 2 & 0 & 1\\
2 & 1 & 0 & 1\\
1 & 0 & 2 & 1\\
2 & 1 & 0 & 0
\end{bmatrix} + 
\begin{bmatrix} 
1 & 0 & -1\\
1 & 0 & -1\\
1 & 0 & -1
\end{bmatrix} * 
\begin{bmatrix}
4 & 2 & 0 & 1\\
0 & 3 & 1 & 0\\
3 & 1 & 0 & 2\\
2 & 2 & 0 & 1
\end{bmatrix}
$$
输出矩阵是一个 $2\times2$ 矩阵。图 24.7 示意三维卷积计算的一步。
![[Pasted image 20240801223243.png]]
#### 1.3 汇聚
卷积神经网络还是用汇聚 (pooling) 运算。
###### 1.3.1 二维汇聚
**定义 24.2 (二维汇聚)** 给定一个 $I\times J$ 输入矩阵 $\mathbf X = [x_{ij}]_{I\times J}$，一个虚设的 $M\times N$ 核矩阵，$M \ll I$，$N\ll J$。让核矩阵在输入矩阵上从左到右再从上到下滑动，将输入矩阵分成若干大小为 $M\times N$ 的子矩阵，这些子矩阵相互不重叠且完全覆盖整个输入矩阵。对每一个子矩阵求最大值或平均值，产生一个 $K\times L$ 输出矩阵 $\mathbf Y = [y_{kl}]_{K\times L}$，称此运算为汇聚或二位汇聚。对子矩阵取最大值的称为最大汇聚 (max pooling)，取平均值的称为平均汇聚 (mean pooling)，即有
$$
y_{kl} = \max_{\begin{aligned}m &\in \{1,2,\cdots,M\}\\ n &\in\{1,2,\cdots,N\} \end{aligned}} x_{k+m-1,l+n-1}
$$
或
$$
y_{kl} = \frac{1}{MN} \sum_{m=1}^M \sum_{n=1}^N x_{k+m-1,l+n-1}
$$
其中，$k=1,2,\cdots,K,l=1,2,\cdots,L$，$K$ 和 $L$ 满足
$$
K = \frac{I}{M} , L = \frac JN
$$
这里假设 $I$ 和 $J$ 分别可以被 $M$ 和 $N$ 整除。

以上是基本汇聚的定义，还有多种扩展。在汇聚运算中，核矩阵每次向右或向下移动的列数或行数也称为步幅。通常汇聚的步幅与核的大小相同。汇聚运算也可以进行填充，即在输入矩阵的周边添加元素为 $0$ 的行和列。汇聚运算依赖核的大小、填充的大小和步幅，也就是说，这些都是超参数。

汇聚也称为**下采样** (down sampling)，因为通过汇聚数据矩阵的大小变小。相反，使数据矩阵变大的运算称为**上采样** (upsampling)。

 比较式 $y_{kl} = \sum_{m=1}^M \sum_{n=1}^N w_{m,n}x_{k+m-1,l+n-1}$ 和式 $y_{kl} = \frac{1}{MN} \sum_{m=1}^M \sum_{n=1}^N x_{k+m-1,l+n-1}$ 容易看出，平均汇聚是卷积的一种特殊情况，其参数个数为 $0$。

**例 24.7** 给定输入矩阵 $\mathbf X$：
$$
\mathbf X = 
\begin{bmatrix} 
3 & 2 & 0 & 1\\
0 & 2 & 1 & 2\\
2 & 0 & 0 & 3\\
2 & 3 & 1 & 2
\end{bmatrix}
$$
核的大小为 $2\times 2$，步幅为 $2$。求 $\mathbf X$ 上的最大汇聚。

**解** 按照 $y_{kl} = \max_{\begin{aligned}m &\in \{1,2,\cdots,M\}\\ n &\in\{1,2,\cdots,N\} \end{aligned}} x_{k+m-1,l+n-1}$ 计算得到最大汇聚的输出矩阵 $\mathbf Y$：
$$
\mathbf Y = 
\begin{bmatrix}
3 & 2 \\
3 & 3
\end{bmatrix}
$$
图 24.8 示意最大汇聚的计算过程。
![[Pasted image 20240801223556.png]]


**例 24.8** 对于与例 24.7 相同的输入矩阵 $\mathbf X$，核的大小为 $2\times 2$，步幅为 $2$，求 $\mathbf X$ 上的平均汇聚。
**解** 按照式 $y_{kl} = \frac{1}{MN} \sum_{m=1}^M \sum_{n=1}^N x_{k+m-1,l+n-1}$ 计算得到平均汇聚的输出矩阵 $\mathbf Y$：
$$
\mathbf Y = 
\begin{bmatrix} 
1.75 & 1\\
1,75 & 1.5
\end{bmatrix}
$$
图 24.9 示意平均汇聚计算的一步。
![[Pasted image 20240802132649.png]]
在图像处理中，汇聚实现的是特征选择。最基本的情况是二维汇聚，输入是一个矩阵，矩阵的一个元素表示一个特征，代表特征的检测值。汇聚运算实际是将汇聚核在输入矩阵上进行滑动，从汇聚核覆盖的特征检测值中选择一个最大值或平均值，这样可以有效地进行特征抽取。输出是一个缩小的矩阵，也就是进行了下采样。
###### 1.3.2 三维汇聚
三维汇聚的输入和输出都是张量表示的特征图。汇聚对输入张量的各个矩阵分别进行汇聚计算，再将结果排列起来，产生输出张量。汇聚的输入特征图和输出特征图的深度相同，输出特征图比输入特征图有更小的高度和宽度。

**例 24.9** 对于例 24.6 的输入张量，核的大小为 $2\times 2$，步幅为 $2$，求 $\mathbf X$ 上的三维最大汇聚。
**解** 对各个矩阵分别按照式 $y_{kl} = \max_{\begin{aligned}m &\in \{1,2,\cdots,M\}\\ n &\in\{1,2,\cdots,N\} \end{aligned}} x_{k+m-1,l+n-1}$ 计算，可得输出张量 $\mathbf Y$：
$$\begin{aligned}
\mathbf Y &= \mathrm{pooling} \left(
\begin{bmatrix}
3 & 2 & 0 & 1 \\
0 & 2 & 1 & 2 \\
2 & 0 & 0 & 3 \\
2 & 3 & 1 & 2 
\end{bmatrix},
\begin{bmatrix}
3 & 2 & 0 & 1 \\
2 & 1 & 0 & 1 \\
1 & 0 & 2 & 1 \\
2 & 1 & 0 & 0 
\end{bmatrix},
\begin{bmatrix}
4 & 2 & 0 & 1 \\
0 & 3 & 1 & 0 \\
3 & 1 & 0 & 2 \\
2 & 2 & 0 & 1 
\end{bmatrix}
\right)\\
& = \left(
\begin{bmatrix}
3 & 2 \\
3 & 3
\end{bmatrix},
\begin{bmatrix}
3 & 1 \\
2 & 2 
\end{bmatrix},
\begin{bmatrix}
4 & 1 \\
3 & 2 
\end{bmatrix}
\right) 
\end{aligned}$$
输出特征图 (输出张量) 是一个 $2\times 2\times 3$ 张量。输入特征图和输出特征图的深度都是 $3$。输入特征图高度和宽度都是 $4$，而输出特征图的高度和宽度都是 $2$。图 24.10 示意三维最大汇聚计算的一步。

![[Pasted image 20240801223723.png]]
#### 1.4 卷积神经网络
###### 1.4.1 模型定义
卷积神经网络是包含卷积运算的一种特殊前馈神经网络。卷积神经网络一般由卷积层、汇聚层和全连接层构成。卷积神经网络架构如图 24.11 所示。
![[Pasted image 20240802134106.png]]
**定义 24.3 (卷积神经网络)**
卷积神经网络是具有以下特点的神经网络。输入是张量表示的数据，输出是标量，表示分类或回归的预测值。经过多个卷积层，有时中间经过汇聚层，最后经过全连接层。每层的输入是张量 (包括矩阵) 表示的特征图，输出也是张量 (包括矩阵) 表示的特征图。

卷积层进行基于卷积函数的仿射变换和基于激活函数的非线性变换。假设第 $l$ 层是卷积层，则第 $l$ 层的计算如下：
$$
\mathbf Z^{(l)} = \mathbf W^{(l)} * \mathbf X^{(l-1)} +\mathbf b^{(l)}\\
$$
$$
\mathbf X^{(l)} = a(\mathbf Z^{(l)})
$$
这里 $\mathbf X^{(l-1)}$ 是输入的 $I \times J \times K$ 张量，$\mathbf X^{(l)}$ 是输出的 $I'\times J' \times K'$ 张量，$\mathbf W^{(l)}$ 是卷积核的 $M\times N\times K\times K'$ 张量，$\mathbf b^{(l)}$ 是偏置的 $I' \times J' \times K'$ 张量，$\mathbf Z^{(l)}$ 是净输入的 $I' \times J' \times K'$ 张量，$a(\cdot)$ 是激活函数。可以认为式 $\mathbf Z^{(l)} = \mathbf W^{(l)} * \mathbf X^{(l-1)} +\mathbf b^{(l)}$ 和 $\mathbf X^{(1)} = a(\mathbf Z^{(l)})$ 表示的变换由一组函数决定，也就是第 $l$ 层的神经元，一个神经元对应输出张量 $\mathbf X^{(l)}$ 的一个元素。另一方面，输入张量 $\mathbf X^{(l-1)}$ 也就是第 $l-1$ 层的输出张量由第 $l-1$ 层的神经元决定。当 $\mathbf X^{(l-1)}$ 的元素到 $\mathbf X^{(l)}$ 的元素之间存在映射关系时，对应的神经元之间存在连接。

汇聚层进行汇聚运算。假设第 $l$ 层是汇聚层，则第 $l$ 层的计算如下：
$$
\mathbf X = \mathrm{pooling} \left(\mathbf X^{(l-1)} \right)
$$
这里 $\mathbf X^{(l-1)}$ 是输入的 $I \times J \times K$ 张量，$\mathbf X^{(l)}$ 是输出的 $I' \times J' \times K$ 张量，$\mathrm{pooling}(\cdot)$ 是汇聚运算。

可以认为式 $\mathbf X = \mathrm{pooling} \left(\mathbf X^{(l-1)} \right)$ 表示的是基于神经元的变换 (汇聚加恒等)。输入张量 $\mathbf X^{(l-1)}$ 由第 $l-1$ 层的神经元决定，输出张量 $\mathbf X^{(l)}$ 由第 $l$ 层的神经元决定。当 $\mathbf X^{(l-1)}$ 的元素到 $\mathbf X^{(l)}$ 的元素之间存在映射关系时，对应的神经元之间存在连接。

全连接的第 $l$ 层是前馈神经网络的一层，进行仿射变换和非线性变换。
$$
\mathbf z^{(l)} = \mathbf W^{(l)} \mathbf x^{(l-1)} + \mathbf b^{(l)}
$$
$$
\mathbf x^{(l)} = a \left(\mathbf z^{(l)} \right)
$$
这里 $\mathbf x^{(l-1)}$ 是 $N$ 维输入向量，是由张量展开得到的；$\mathbf x^{(l)}$ 是 $M$ 维输出向量；$\mathbf W^{(l)}$ 是 $M\times N$ 权重矩阵；$\mathbf b^{(l)}$ 是 $M$ 维偏置向量；$\mathbf z^{(l)}$ 是 $M$ 维净输入向量；$a(\cdot)$ 是激活函数。全连接的最后一层输出的是标量。

卷积神经网络中的所有参数，包括卷积核的权重和偏置、全连接的权重和偏置，都通过学习获得。

卷积神经网络也可以只有卷积层和全连接层，而没有汇聚层。步幅大于 $1$ 的卷积运算也可以起到下采样作用，以代替汇聚运算。为了达到更好的预测效果，设计上的原则通常是使用更小的卷积核 (如 $3 \times 3$ ) 和更深的结构，前段使用少量的卷积核，后端使用大量的卷积核。

可以将 $I \times J \times K$ 张量展开成 $K$ 个 $I\times J$ 矩阵，将 $I' \times J' \times K'$ 张量展开成 $K'$ 个 $I'\times J'$ 矩阵。卷积层计算也可以写作
$$
\mathbf Z_{\mathbf k'}^{(l)} = \sum_{k} \mathbf W_{k,k'}^{(l)} * \mathbf X_k ^{(l-1)} + \mathbf b_{k'}^{(l)}
$$
$$
\mathbf X_{k'}^{(l)} = a \left(\mathbf Z_{k'}^{(l)} \right)
$$
这里 $\mathbf X_k^{(l-1)}$ 是输入的第 $k$ 个 $I\times J$ 矩阵，$\mathbf X_{k'}^{(l)}$ 是输出的第 $k'$ 个 $I' \times J'$ 矩阵，$\mathbf W_{k,k'}^{(l)}$ 是二维卷积核的第 $k\times k'$ 个 $M \times N$ 矩阵，$\mathbf b_{k'}^{(l)}$ 是偏置的第 $k'$ 个 $I'\times J'$ 矩阵，$\mathbf Z_{k'}^{(l)}$ 是净输入的第 $k'$ 个 $I' \times J'$ 矩阵。图 24.12 显示卷积层的输入和输出张量 (特征图)。
![[Pasted image 20240802134347.png]]
每次对 $K$ 个 $I\times J$ 矩阵同时进行卷积运算得到 $1$ 个 $I'\times J'$ 矩阵，整体计算 $K'$ 次得到 $K'$ 个 $I' \times J'$ 矩阵，卷积核是 $K'$ 个 $M\times N \times K$ 张量。输入和输出张量的深度分别是 $K$ 和 $K'$。

可以将 $I\times J\times K$ 张量展开成 $K$ 个 $I \times J$ 矩阵，将 $I'\times J' \times K$ 张量展开成 $K$ 个 $I' \times J'$ 矩阵。汇聚层计算也可以写作
$$
\mathbf X_{k}^{(l)} = \mathrm{pooling} \left(\mathbf X_k^{(l-1)} \right)
$$
这里 $\mathbf X_{k}^{(l-1)}$ 是输入的是第 $k$ 个 $I\times J$ 矩阵分别进行，得到 $K$ 个 $I'\times J'$ 矩阵，汇聚核是 $K$ 个 $M\times N$ 矩阵。输入和输出张量的深度都是 $K$。

卷积神经网络的特点可以由每一层的输入和输出张量体现，所以习惯上用输入和输出张量表示其架构。
![[Pasted image 20240802134356.png]]
###### 1.4.2 模型例子
下面是一个简单的卷积神经网络的例子。这个 CNN 模型与 LeCun 提出的 LeNet 模型有相近的架构和规模。该模型在手写数字识别上达到很高的准确率，是卷积神经网络最基本的模型。整个网络由两个卷积层、两个汇聚层、两个全连接层、一个输出层组成 (图 24.14)。表 24.1 列出了卷积层、汇聚层、全连接层、输出层的超参数，输出特征图的大小，其中 $F$ 表示卷积核或汇聚核的大小，$S$ 表示步幅，$W$ 表示权重矩阵的大小，$B$ 表示偏置向量的长度。
![[Pasted image 20240802134411.png]]
![[Pasted image 20240802134427.png]]
#### 1.5 卷积神经网络性质

###### 1.5.1 表示效率
卷积神经网络的表示和学习效率比前馈神经网络高。首先层与层之间的连接是稀疏的，因为卷积代表的是稀疏连接，比全连接的数目大幅减少，如图 24.15 所示。
![[Pasted image 20240802150025.png]]
其次同一层的卷积的参数是共享的，卷积核在前一层的各个位置上滑动计算，在所有位置上具有相同的参数，这样就大幅减少了参数的数量。另外，每一层内的卷积运算可以并行处理，这样也可以加快学习和推理的速度。
###### 1.5.2 不变性
设 $f(\mathbf x)$ 是以 $\mathbf x$ 为输入的函数，$\tau(\mathbf x)$ 是对 $\mathbf x$ 的变换，如平移变换、旋转变换、缩放变换。如果满足一下关系，
$$
f(\mathbf x) = f(\tau (\mathbf x))
$$
则称函数 $f(\cdot)$ 对变换 $\tau (\cdot)$ 具有不变性。如果 $\tau(\cdot)$ 表示的是平移变换、旋转变换、缩放变换，则函数 $f(\cdot)$ 具有平移不变性、旋转不变性、缩放不变性。

卷积神经网络具有平移不变性，但不能严格保证；不具有旋转不变性、缩放不变性。这意味着图像识别中，图像中的物体平行移动位置也能被识别。在图像识别中，往往通过数据增强的方法提高卷积神经网络的旋转不变性和缩放不变性。

**例 24.10** 图 24.16 给出从两张图片中进行特征抽取的例子。两张图片中包含 $L$ 字，但位置发生了平移。通过卷积和汇聚运算，可以分别抽取出两张图片中的这个特征，卷积使用表示 $L$ 字的卷积核，汇聚使用最大汇聚。所以，这里的卷积和汇聚运算对特征抽取具有平移不变性。
![[Pasted image 20240802112328.png]]
下面给出三个不变性的严格定义。在平面上的点的坐标 $(x,y)$ 通过以下矩阵表示的变换变成新的坐标 $(x',y')$，则分别称变换为平移变换、旋转变换、缩放变换。
**(1) 平移变换**
$$
\begin{bmatrix}x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix}1 & 0 & t_{x} \\ 0 & 1 & t_y\\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix}x \\ y \\ 1 \end{bmatrix}
$$
其中，$t_x$ 和 $t_y$ 分别表示点在 $x$ 轴和 $y$ 轴方向平移的幅度。
**(2) 旋转变换**
$$
\begin{bmatrix}
x'\\
y'\\
1
\end{bmatrix} = 
\begin{bmatrix}
\cos \theta & -\sin \theta & 0 \\
\sin \theta & \cos \theta & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\ y \\1
\end{bmatrix}
$$
其中，$\theta$ 表示点围绕原点旋转的角度。
**(3) 缩放变换**
$$
\begin{bmatrix}
x'\\
y'\\
1
\end{bmatrix} = 
\begin{bmatrix}
s_x & 0 & 0\\
0 & s_y & 0\\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x\\
y\\
1
\end{bmatrix}
$$
其中，$s_x$ 和 $s_y$ 分别表示点在 $x$ 轴和 $y$ 轴方向缩放的尺度。
###### 1.5.3 感受野
卷积神经网络利用卷积实现了图像处理需要的特征的表示。前端端神经元表示的是局部的特征，如物体的轮廓；后端的神经元表示的是全局的特征，如物体的部件，可以更好地对图像数据进行预测。
卷积神经网络通过特殊的函数表示和学习实现了自己的感受野机制。卷积神经网络的感受是指神经元涵盖的输入矩阵的部分 (二维图像的区域)。图 24.17 显示的网络有两个卷积层，输入层是二维图像。第一层的绿色神经元的感受野是输入层的绿色区域，第二层的黄色神经元的感受野是输入层的整个区域。感受野是从神经元的输出到输入反向看过去得到的结果。卷积核加激活函数产生的感受野具有与生物视觉系统中的感受野相似的特点。
![[Pasted image 20240802145850.png]]
考虑卷积神经网络全部由卷积层组成的情况，神经元的感受野的大小有以下关系成立。证明留作习题。
$$
R^{(l)} = 1 + \sum_{j=1}^{l} \left(F^{(j)} - 1 \right) \prod_{i=0}^{j-1} S^{(i)}
$$
设输入矩阵和卷积核都呈正方形。$R^{(l)} \times R^{(l)}$ 表示第 $l$ 层的神经元的感受野的大小，$F^{(j)}\times F^{(j)}$ 表示第 $j$ 层的卷积核的大小，$S^{(i)}$ 表示第 $i$ 层卷积的步幅，设 $S^{(0)}=1$。
## 2 卷积神经网络的学习算法
卷积神经网络的学习算法也是反向传播算法，与前馈神经网络学习的反向传播算法相似，不同点在于正向和反向传播基于卷积函数。
#### 2.1 卷积导数
设有函数 $f(\mathbf Z)$，$\mathbf Z=\mathbf W * \mathbf X$，其中 $\mathbf X = [x_{ij}]_{I\times J}$ 是输入矩阵，$\mathbf W = [w_{mn}]_{M\times N}$ 是卷积核，$\mathbf Z = [z_{kl}]_{K\times L}$ 是净输入矩阵，则 $f(\mathbf Z)$ 对 $\mathbf W$ 的偏导数如下：
$$
\frac{\partial f(\mathbf Z)}{\partial w_{mn}} = \sum_{k=1}^K \sum_{l=1}^L\frac{\partial z_{kl}}{\partial w_{mn}} \frac{\partial f(\mathbf Z)}{\partial z_{kl}} = \sum_{k=1}^{K} \sum_{l=1}^L x_{k+m-1,l+n-1} \frac{\partial f(\mathbf Z)}{\partial z_{kl}}
$$
整体可以写作
$$
\frac{\partial f(\mathbf Z)}{\partial \mathbf W} = \frac{\partial f(\mathbf Z)}{\partial Z} * \mathbf X
$$
$f(\mathbf Z)$ 对 $\mathbf X$ 的偏导数如下：
$$
\frac{f(\mathbf Z)}{\partial x_{ij}} = \sum_{k=1}^{K} \sum_{l=1}^{L} \frac{z_{kl}}{\partial x_{ij}} \frac{\partial f(\mathbf Z)}{\partial z_{kl}} = \sum_{k=1}^{K} \sum_{l=1}^L w_{i-k+1,j-1+1} \frac{\partial f(\mathbf Z)}{\partial z_{kl}}
$$
整体可以写作
$$
\frac{\partial f(\mathbf Z)}{\partial \mathbf X} = \mathrm{rot180}\left(\frac{\partial f(\mathbf Z)}{\partial \mathbf Z} \right) * \mathbf W = \mathrm{rot180}(\mathbf W)*\frac{\partial f(\mathbf Z)}{\partial \mathbf Z}
$$其中，$\mathrm{rot 180}()$ 表示矩阵 180 度旋转，这里的卷积 $*$ 是对输入矩阵进行全填充后的卷积。
#### 2.2 反向传播算法
卷积神经网络和前馈神经网络一样，也是通过反向传播算法求出损失函数对各层参数的梯度，利用随机梯度下降法更新模型参数。对于每次迭代，首先通过正向传播从前往后传递信号，然后通过反向传播从后往前传递误差，最后求损失函数对每层的参数的梯度，对每层的参数进行更新。对于卷积神经网络，特殊的是卷积层和汇聚层的参数更新。
###### 2.2.1 卷积层
设第 $l$ 层为卷积层。由式 $\mathbf Z_{\mathbf k'}^{(l)} = \sum_{k} \mathbf W_{k,k'}^{(l)} * \mathbf X_k ^{(l-1)} + \mathbf b_{k'}^{(l)}$ 和式 $\mathbf X_{k'}^{(l)} = a \left(\mathbf Z_{k'}^{(l)} \right)$ 可知，第 $l$ 层的第 $k'$ 个净输入矩阵 $\mathbf Z_{k'}^{(l)}$ 为
$$
\mathbf Z_{k'}^{(l)} = \sum_{k=1}^{K} \mathbf W_{k,k'}^{(l)} * \mathbf X_{k}^{(l-1)} + \mathbf b_{k'}^{(l)}
$$
其中，$\mathbf X_{k}^{(l-1)}$ 是第 $l$ 层的第 $k$ 个输入矩阵，$\mathbf W_{k,k'}^{(l)}$ 是第 $l$ 层的第 $k\times k'$ 个卷积核矩阵，$\mathbf b_{k'}^{(l)}$ 是第 $l$ 层的第 $k'$ 个偏置矩阵。第 $k'$ 个输出矩阵 $\mathbf X_{k'}^{(l)}$ 为

$$
\mathbf X_{k'}^{(l)} = a\left(\mathbf Z_{k'}^{(l)} \right)
$$
由此可以进行从第 $l-1$ 层到第 $l$ 层的正向传播，$\mathbf X_{k}^{(l-1)}$ 从第 $l-1$ 层的神经元传递到第 $l$ 层的相连神经元，得到 $\mathbf X_{k'}^{(l)}$。以上计算可以扩展到第 $l$ 层的所有 $K'$ 个输出矩阵上，
再考虑第 $l$ 层的梯度更新。第 $l$ 层的第 $k$ 个输入矩阵是 $\mathbf X_{k}^{(l-1)}$。设第 $l$ 层的第 $k'$ 个误差矩阵 $\delta_{k'}^{(l)}$ 是
$$
\delta_{k'}^{(l)} = \frac{\partial L}{\partial \mathbf Z_{k'}^{(l)}}
$$
###### 2.2.2 汇聚层
设第 $l$ 层为汇聚层。由式 $\mathbf X_{k}^{(l)} = \mathrm{pooling} \left(\mathbf X_k^{(l-1)} \right)$ 可知，第 $l$ 层的第 $k$ 个输出矩阵 $\mathbf X_{k}^{(l)}$ 为
$$
\mathbf X_k^{(l)} = \mathbf Z_k^{(l)} = \mathrm{pooling} \left(\mathbf X_k^{(l-1)}\right)
$$
这里 $\mathbf X_k^{(l-1)}$ 是第 $l$ 层的第 $k$ 个输入矩阵。引入第 $l$ 层的第 $k$ 个净输入矩阵 $\mathbf Z_k^{(l)}$，净输入 $\mathbf Z_k^{(l)}$ 和输出 $\mathbf X_k^{(l)}$ 之间是恒等变换。由此可以进行从第 $l-1$ 层到第 $l$ 层的正向传播，$\mathbf X_{k}^{(l-1)}$ 从第 $l-1$ 层的神经元传递到第 $l$ 层的相连神经元，得到 $\mathbf X_k^{(l)}$。以上计算可以扩展到第 $l$ 层的所有 $K$ 个输出矩阵上。

汇聚层没有参数，所以在学习过程中没有参数更新。
再考虑从第 $l$ 层到第 $l-1$ 层的误差反向传播。设第 $l$ 层的第 $k$ 个误差矩阵是
$$
\delta_{k}^{(l)} = \frac{\partial L}{\partial \mathbf Z_{k}^{(l)}}
$$
第 $l-1$ 层的第 $k$ 个误差矩阵 $\delta_k^{(l-1)}$ 是
$$
\delta_k^{(l-1)} = \frac{\partial L}{\partial \mathbf Z_k^{(l-1)}}
$$
通过 $\delta_{k}^{(l)}$ 计算 $\delta_k^{(l-1)}$ 。由链式法则可得：
$$
\delta_k^{(l-1)} = \frac{\partial L}{\partial \mathbf Z_k^{(l-1)}} = \frac{\partial \mathbf X_k^{(l-1)}}{\partial \mathbf Z_k^{(l-1)}} \frac{\partial L}{\partial \mathbf X_k^{(l-1)}} = \frac{\partial \mathbf X_{k}^{(l-1)}}{\partial \mathbf Z_{k}^{(l-1)}} \frac{\partial \mathbf Z_k^{(l)}}{\partial \mathbf X_k^{(l-1)}} \frac{\partial L}{\partial \mathbf Z_k^{(l)}} = \frac{\partial a}{\partial \mathbf Z_k^{(l-1)}} \odot \mathrm{up\_sample} \left(\delta_k^{ (l)} \right)
$$
这里 $\odot$ 表示矩阵的逐元素积：$\mathrm{up\_sample} \left(\mathbf \delta_k^{(l)} \right)$ 是误差矩阵 $\mathbf \delta_k^{(l)}$ 的上采样，是汇聚（下采样）的反向运算。最大汇聚时，$\delta_k^{(l)}$ 从第 $l$ 层的神经元传递到第 $l-1$ 层的输出最大的相连神经元：平均汇聚时，$\delta_{k}^{(l)}$ 从第 $l$ 层的神经元平均分配到第 $l-1$ 层的相连神经元。以上计算可以扩展到第 $l-1$ 层的所有 $K$ 个误差矩阵上。 
###### 2.2.3 算法
算法 24.1 给
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

