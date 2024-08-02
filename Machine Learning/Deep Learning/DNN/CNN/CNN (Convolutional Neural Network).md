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

![[Pasted image 20240801222907.png]]
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
![[Pasted image 20240802112242.png]]

###### 1.5.2 不变性
![[Pasted image 20240802112258.png]]
![[Pasted image 20240802112310.png]]
![[Pasted image 20240802112328.png]]
![[Pasted image 20240802112344.png]]


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

