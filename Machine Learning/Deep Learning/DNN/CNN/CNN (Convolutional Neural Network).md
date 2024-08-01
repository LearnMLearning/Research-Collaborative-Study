![[Pasted image 20240801221517.png]]
## 1 卷积神经网络的模型

![[Pasted image 20240801221536.png]]
#### 1.1 背景
![[Pasted image 20240801221550.png]]
![[Pasted image 20240801221601.png]]


#### 1.2 卷积
###### 1.2.1 数学卷积
![[Pasted image 20240801221645.png]]
###### 1.2.2 二维卷积
卷积神经网络中的卷积与数学卷积并不相同，实际是数学的互相关 (cross correlation)
![[Pasted image 20240801221742.png]]
![[Pasted image 20240801222747.png]]
![[Pasted image 20240801222802.png]]
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
#### 2.1 卷积导数


#### 2.2 反向传播算法
## 3 图像分类中的应用
#### 3.1 AlexNet
#### 3.2 残差网络
