---
layout: post
title:  "A survey on Learning to Hash"
date:   2019-09-03 21:03:36 +0530
categories: Machine_Learning

---

哈希算法主要分为两大类：

\1. 局部敏感哈希（Locality sensitive hashing）

\2. 学习哈希（learning to hash)

根据采用的相似性保留方式(the similarity perserving manner)来制定目标函数(the similarity perserving manner)的不同，将哈希算法（learning to hash）分为如下几类:

\1. The pairwise similarity preserving class

\2. The multiwise similarity preserving class

\3. The implicit similarity preserving class

\4. The quantization class

选择相似性保留方式进行分类的主要原因是相似度保持是哈希的基本目标。需要注意的是，哈希函数或优化算法等其他因素对搜索性能也很重要。

**一、** **The pairwise similarity preserving** 

\1. **Similarity-distance product minimization (SDPM)**: the distance in the coding space is expected to be smaller if the similarity in the original space is larger.

\2. **Similarity-similarity product maximization (SSPM)**: the similarity in the coding space is expected to be larger if the similarity in the original space is larger.

\3. **Distance-distance product maximization (DDPM)**: the distance in the coding space is expected to be larger if the distance in the original space is larger.

\4. **Distance-similarity product minimization (DSPM)**: the similarity in the coding space is expected to be smaller if the distance in the original space is larger.

\5. **Similarity-similarity difference minimization (SSDM)**: the difference between the similarities is expected to be as small as possible.

\6. **Distance-distance difference minimization (DDDM)**: the difference between the distances is expected to be as small as possible.

\7. **Normalized similarity-similarity divergence minimization (NSSDM)**

**1. Similarity-distance product minimization (SDPM)**

a. Spectral Hashing (2008)

b. Linear discriminant analysis (LDA) hashing (2012)

c. Minimal loss hashing (2011)

**2. Similarity-similarity product maximization (SSPM)**

a. Semi-supervised hashing (2010)

**3. Distance-distance product maximization (DDPM)**

a. Topology preserving hashing (2013)

**5.** **Similarity-similarity difference minimization (SSDM)**

a. Supervised hashing with kernels (2012)

b. Binary hashing (2016)

**6. Distance-distance difference minimization (DDDM)**

a. Binary reconstructive embedding (2009)

**二、Multiwise similarity preserving** 

\1. Order preserving hashing （2013)

\2. Triplet loss hashing (2012)

 

**三、Implicit similarity preserving**

\1. Random maximum margin hashing (2011)

\2. Complementary projection hashing (2013)

\3. Spherical hashing (2012)

**四、Quantization**

\1. Hypercubic Quantization

\2. Cartesian Quantization

**1. Hypercubic Quantization** 

a. Iterative Quantization (2011)

b. Harmonious hashing  (2013)

**2. Cartesian Quantization**

a. Product Quantization