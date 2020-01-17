---
layout: post
title:  "如何在目标函数中生成一个低秩矩阵？"
date:   2019-04-25 21:03:36 +0530
categories: Machine_Learning

---

在目标函数中限制一个矩阵为低秩矩阵的方法，例如下面的式子：

![img](http://note.youdao.com/yws/public/resource/2dbbb023cd040e0eee4c5710f6377fc5/xmlnote/8A56439C28964840953EF934AD98F827/416)

对XW的低秩约束使得优化更加困难，因为低秩是众所周知的np难问题。作为一种替代方法，迹范数

被广泛地用于在以前的工作中鼓励低秩性(Li et al. 2016)。

![img](http://note.youdao.com/yws/public/resource/2dbbb023cd040e0eee4c5710f6377fc5/xmlnote/7FA23C04491F456B92FA3110CCBC3A3A/414)

然而，迹范数控制矩阵的单个值，但是单个值的变化并不总是导致秩的变化。由(Ding, Shao, and Fu 2017)的启发，提议使用低秩约束的显式形式如下：

![img](http://note.youdao.com/yws/public/resource/2dbbb023cd040e0eee4c5710f6377fc5/xmlnote/BAC459D73F6F4942AD82CE57FB98C055/430)

其中**：**

![img](http://note.youdao.com/yws/public/resource/2dbbb023cd040e0eee4c5710f6377fc5/xmlnote/1A9B2DD7CF7C4095B25AB884D43B00EF/435)

其中tr是矩阵的迹算子，V是对应于(XW)(XW)^T的(d - r)个最小奇异值的奇异向量。