译文：在本研究中，我们提出了一种称为自适应决策森林 Adaptive Decision Forest (ADF)的增量机器学习框架，它产生一个决策森林来对新记录进行分类。基于我们的两个新定理，我们引入了一种称为iSAT的新分裂策略，它允许ADF对新记录进行分类，即使它们与以前未见过的类相关联。ADF能够识别和处理概念漂移;然而，它不会忘记以前获得的知识。此外，如果数据可以分批处理，ADF是能够处理大数据的。我们在9个公开可用的自然数据集和1个合成数据集上评估了ADF，并将ADF的性能与8种最先进的技术的性能进行了比较。我们还研究了ADF在一些具有挑战性的情况下的有效性。我们的实验结果，包括统计符号检验和Nemenyi检验分析，表明所提出的框架明显优于最先进的技术。

## 1. Introduction


## 2. Problem formulation and related work
#### 2.1 Problem formulation and assumptions

#### 2.2 Related work


## 3. Our proposed incremental learning framework: Adaptive decision forest (ADF)
#### 3.1 Basic concept of ADF

###### 3.1.1 Improved separating axis theorem (isat) splitting strategy
###### 3.1.2 Decision trees repairable strategy

###### 3.1.3 Handling concept drifts using three parallel forests

###### 3.1.4 Identification of sustainable concept drift (SCD)

###### 3.1.5 Analysis of impact of using a window of batches

#### 3.2 Incremental tree growing mechanism of ADF

#### 3.3 Complexity analysis of ADF

## 4. Experimental results and discussion

#### 4.1 Datasets

#### 4.2 Simulation of training and test batch datasets

#### 4.3 Experimental settings

#### 4.4 Detail experimental results and performance analysis

#### 4.5 Statistical analysis of the experimental results

## 5. Analysis of the effectiveness of ADF in some challenging situations

## 6. Conclusion and future work

