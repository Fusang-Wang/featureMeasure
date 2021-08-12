# Measuring feature expansion of image data augmentation（with generative model）

## Abstract

目标：

着眼于特征，定义与量化评价数据增强有效性，量化评价GAN对样本特征的扩征能力。
（How to measure feature expansion for image data augmentation？）

方法：

1.探讨能够度量图片语义的空间需要哪些性质？提出在特征空间内有效评价语义扩充的条件，揭示
GAN对样本特征扩充的物理本质（可解释性、物理原理），指明了隐空间，作为一种特殊的特征空间，
具有语义表征的能力（摄影几何vs分布拟合，传统增强方法与GAN增强方法的不用）

2.分析验证GAN-like方法隐空间的性质与语义扩充评价条件，提出对GAN-like网络建立公共隐空间的方法
（验证公共隐空间的性质），实现对不同模型特征扩征的比较

3.解决了OT距离的存在性，提出评价指标的解析解，确立公共隐空间中的特征扩充评价指标（测度
metrics），量化评价不同算法扩征有效性

结果与结论：

对不同类型的GAN网络测试了评价模型与评价指标，分析性能差异，并与传统的评价模型做对比。探讨了能够度量语义信息的特征空间需要满足的性质，有效评价语义扩充的条件。在公共特征空间中，提出了着眼于语义信息评价数据增强效果的评价指标。指导在数据增强问题上GAN的评价与选择。

## 1 Introduction

### 1.1 Related work

### 1.2 Central question

How to measure feature expansion for image data augmentation?

-- What are the criterions on feature space for evaluation of image data augmentation，and our solution.

问题的逻辑，problem formulation.

## 2 Feature space for evaluation of image data augmentation(如果2.1只做review 不划分小章节，只做完整的一章从驳论，立论，解决方案)

### 2.1 feature space of image data

* 隐空间，满足的性质1，性质2，性质3 -> 通过前人对隐空间性质的研究验证。

* 隐空间具有语义表征的能力：属性编辑等。
* 揭示GAN对样本特征扩充的物理本质：**流形学习角度解释**GANs如何补全特征+小实验。 

### 2.2 Network-based feature space

* 需要文献调查

### 2.1 Feature space of image data

* latent space of gan

  * 隐空间，满足的性质1，性质2，性质3 -> 通过前人对隐空间性质的研究验证。

  * 隐空间具有语义表征的能力：属性编辑等。

  * 揭示GAN对样本特征扩充的物理本质：**流形学习角度解释**GANs如何补全特征+小实验。 

* network-based feature space
* 

## Data augementation（G）与 F 之间的逻辑关系

### The nature of feature-based data augmentation

* high dimensional data -> a distribution on low dimensional non linear manifold
* genrative model, transfor uniform distribution to the original data distribution -> oversampling the learned manifold with interpolation and expolation.

### How to evaluate

* how to evaluate interpolation or expolations on the manifold? (no inclusion of tradidtional method that requires prior knowleadge)

  * require a feature space that flatten the manifold. (difficulty: manifold too abstract,

  * interpolation: constrained in a single class. For each class, linearity of different semantic features
  * expolation:each class forms a convex set in the $\mathbb{F}$ space, when new feature appears, it falls out of the current zone

### 2.3 Criterions on feature space

* Criterions on feature space for evaluation of image data augmentation: how to evaluation data augmentation based on feature space? 能够度量图片语义的空间需要哪些性质?
* we propose a new feature space to evaluate feature expansion of images data:  $\mathbb{F}$​ space



## 3 Toward $\mathbb{F}$ space

Based on the criterions proposed in section 2, we attempt to, in this section, find the $\mathbb{F}$ space through training a network that maps images to  $\mathbb{F}$ .

### 3.1 criterion 1

### 3.2 criterion 2

### 3.3 criterion 3

### 3.4 OT-based metrics in feature space $\mathbb{F}$​（metric design）

* 解决了OT距离的存在性
* 提出评价指标的解析解
* 确立公共隐空间中的特征扩充评价指标（测度metrics)--LASFAS
* 量化评价不同算法扩征有效性

### 3.5  feature score in LASFAS

将验证评价指标与FID,IS等指数的关联性，例如 SIC = FID + IS + FS，其中FS代表了对语义信息的评价



### （3.5） Example on GAN-like networks(??放在哪里)

提出对GAN-like网络建立公共隐空间的方法

* 对于两个预训练的生成器网络A，B

* 寻找映射f, 以方法一中提所出的性质为目标将A，B映射到**公共中间空间**W， W是一个特征空间。

* 在**公共中间空间**W中选择比较分布的距离。

  

## 4 Experiment

To test and evaluate the performance of LASFAS, we compared it with traditional evaluation metrics (FID and IS) on several data sets with different GANs. Moreover,  we have explained its relevance with FID and IS using... theorem.

### 4.0 验证语义合理性

通过可视化的方式验证 $\mathbb{F}$ 的合理性​



### 4.1 CelebA dataset

提出对GAN-like网络建立公共隐空间的方法

* 对于两个预训练的生成器网络A，B

* 寻找映射f, 以方法一中提所出的性质为目标将A，B映射到**公共中间空间**W， W是一个特征空间。

* 在**公共中间空间**W中选择比较分布的距离。

  ## 结果

* | GAN 方法  | 隐空间的映射   | 对隐空间向量的操作：feature |      |
  | --------- | -------------- | --------------------------- | ---- |
  | style-gan | LSFAS, IS，FID | LSFAS, IS，FID              |      |
  | wgan      | LSFAS, IS，FID | LSFAS, IS，FID              |      |
  | ...       | LSFAS, IS，FID | LSFAS, IS，FID              |      |

### 4.2 Minist 





 ## 5 Results

对不同类型的GAN网络测试了评价模型与评价指标，分析性能差异，并与传统的评价模型做对比



## 6 Conclusion

探讨了能够度量语义信息的特征空间需要满足的性质，有效评价语义扩充的条件。在公共特征空间中，提出了着眼于语义信息评价数据增强效果的评价指标。指导在数据增强问题上GAN的评价与选择。

