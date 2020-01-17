---
layout: post
title:  "Deep Learning based Recommender Syst..."
date:   2019-04-16 21:03:36 +0530
categories: Machine_Learning

---

**推荐系统相关的知名会议**：

   NIPS、ICML、ICLR、KDD、WWW、SIGIR、WSDM、RecSys等等

**传统的推荐系统通常分成三类：**

1. **collaborative filtering：** 通过学习用户 - 项的历史交互，显式(如用户以前的评分)或隐式反馈(如浏览历史)，提出建议。
2. **content based:**  主要基于项目和用户辅助信息之间的比较。可以考虑各种各样的辅助信息，例如文本、图像和视频。
3. **hybrid recommender system:**  综合了两种或两种以上推荐策略的推荐系统。

**神经网络适用于推荐系统的原因（即神经网络推荐系统的优点）：**

1. **Nonlinear Transformation:** 作为许多传统推荐者的基础的线性假设过于简单，并将极大地限制其建模表达性。 与线性模型相反，深度神经网络能够利用非线性激活（如relu，sigmoid，tanh等）对数据中的非线性建模。已经确定神经网络能够以任意精度逼近任何连续函数 通过改变激活选择和组合。 此属性可以处理复杂的交互paerns并精确地反映用户的偏好。
2. **Representation Learning:** 深度神经网络能够有效地从输入数据中学习潜在的解释因素和有用的表示形式。通常，在实际应用程序中可以获得大量关于项目和用户的描述性信息。利用这些信息可以提高我们对商品和用户的了解，从而得到更好的推荐。       

**Advantages**: 

- 1. 它减少了手工特征设计的工作量。特征工程是一项劳动密集型的工作，深层神经网络能够在无监督或监督的方式下自动从原始数据中学习特征。          
  2. 它使推荐模型能够包含不同种类的内容信息，如文本、图像、音频甚至视频。深度学习网络在多媒体数据处理方面取得了突破，在多种来源的表示学习方面显示出潜力。

1. **Sequence Modelling:** 序列信号建模是挖掘用户行为和项目演化时间动态的一个重要课题。例如，下一项/购物篮预测和基于会话的推荐是典型的应用程序。因此，深度神经网络成为这种顺序模式挖掘任务的完美工具。
2. **Fiexibility:** 深度学习技术具有很高的灵活性，特别是随着许多流行的深度学习框架的出现，如 Tensorow, Keras, PyTorch等. 我们很容易将不同的神经结构组合起来制定强大的混合模型，或者用其他模块替换一个模块。 因此，我们可以轻松地建立混合和复合推荐模型，以同时捕捉不同的特征和因素。

**在推荐系统上使用神经网络的局限性（即反对使用神经网络的几个论点）：**

1. **Interpretability：**反对深度神经网络的一个常见论点是隐藏的权重和激活通常是不可解释的，限制了可解释性。然而，随着神经注意力模型的出现，这种担忧一般得到了缓解，并为深度神经模型铺平了世界，这些模型享有更高的可解释性。 虽然解释单个神经元仍然对神经模型（不仅在推荐系统中）构成挑战，但是现有的最先进模型已经能够在某种程度上解释，从而能够提供可解释的推荐。 
2. **Data Requirement：**深度学习被认为是数据饥渴，因为它需要足够的数据才能完全支持其丰富的参数化。 然而，与标记数据稀缺的其他领域（例如语言或视觉）相比，在推荐系统研究的背景下获取大量数据相对容易。 百万/亿比例的数据集不仅在工业中很常见，而且作为学术数据集发布。
3. **Extensive Hyperparameter Tuning：**深度学习需要广泛的超参数调优。然而，超参数调整不是深度学习的唯一问题，而是一般的机器学习（例如，传统的矩阵分解同样必须针对正规化因子和学习速率等进行调整）。当然，在某些情况下，深度学习可能会引入额外的超参数。 例如最近的一项工作，对传统度量学习算法的注意扩展只引入了一个超参数。

**基于深度学习的分类推荐模型，主要分为两类（当然也可以按特定的应用领域进行划分）：**

1. **Recommendation with Neural Building Blocks:**  按照不同的深度学习模型，可将基于神经网络的推荐系统模型分为不同的子类。使用不同的深度学习技术决定了推荐模型的适用性。 例如，MLP可以轻松地模拟用户和项目之间的非线性交互; CNN能够从异构数据源（例如文本和视觉信息）中提取本地和全局表示; RNN使推荐系统能够对内容信息的时间动态和连续演进进行建模。
2. **Recommendation with Deep Hybrid Models:** 一些基于深度学习的推荐模型使用多种深度学习技术。 深度神经网络的灵活性使得将几个神经构建块组合在一起以相互补充并形成更强大的混合模型成为可能。深度学习技术（如RNN、CNN、MLP等）有许多可能的组合，但并非所有技术都被利用。

![img](http://note.youdao.com/yws/public/resource/59bd177fe5faf6f78ed91365f57c6f27/xmlnote/DC7B7D78968B4A20876FC4D21308753B/78)

**未来的研究方向和有待解决的问题：**

1. **Joint Representation Learning from User and Item Content Information：**

一个有趣的前瞻性研究问题是如何设计最能利用其他数据模式的神经架构。 最近的一项工作可能为这种性质的模型铺平道路，即联合代表学习框架。 学习用户和项目的联合（可能是多模态表示）可能会成为推荐系统研究的下一个新兴趋势。 为此，深入学习这一方面将是如何以端到端的方式设计更好的归纳偏差（混合神经架构）。 例如，推理不同的模态（文本，图像，交互）数据以获得更好的推荐性能。

1. **Explainable Recommendation with Deep Learning ：**

一种常见的解释是深度神经网络是高度不可解释的。因此，提出可解释的建议似乎是一项艰巨的任务。同样地，假设大而复杂的神经模型只是依据任何真正的理解来拟合数据，这也是很自然的。这正是为什么这个方向既令人兴奋又至关重要的原因。可解释的深度学习主要有两种方法。第一个是对用户做出可解释的预测，让他们了解网络建议背后的因素（即为什么推荐这个项目/服务？）。第二个轨道主要侧重于从业者的解释能力，探测权重和激活以更多地了解模型。截至今天，注意力模型或多或少地缓解了神经模型的不可解释的问题。如果有的话，注意力模型反而导致更大范围的可解释性，因为注意权重不仅提供关于模型的内部工作的见解，而且还能够向用户提供可解释的结果。虽然这是研究“预先深度学习”的现有方向，但注意力模型不仅能够提高性能，而且具有更强的可解释性。这进一步激发了深度学习的推荐用途。值得注意的是，模型的可解释性和可解释性强烈依赖于应用领域和内容信息的使用，这既直观又自然。例如，主要使用评论作为可解释性的媒介（这些评论导致进行哪些预测）。可以考虑许多其他介质/形式，例如图像。为此，一个有希望的方向和下一步将是设计更好的注意机制，可能达到提供对话或生成解释的水平。鉴于模型已经能够强调决策的作用，我们认为这是下一个前沿。

1.  **Going Deeper for Recommendation**

在深度学习的机器推理方面，包括对自然语言或视觉输入的推理，近年来已有许多进展。我们认为机器阅读、推理、问答甚至视觉推理等任务对推荐系统领域有很大的影响。这些任务看起来完全是武断的，与推荐系统无关，因此显得过于呆滞。然而，推荐系统需要对单个(或多个)模式(审查、文本、图像、元数据)进行推理，这最终将需要从这些相关的领域中借用(和调整)技术，这是非常必要的。从根本上说，推荐和推理(如回答问题)是高度相关的，因为它们都是信息检索问题。为此，推荐系统的下一个前沿可能适应需要多步推理和推理的情况。 一个简单的例子可以推断用户的社交资料，购买等，推理多种形式推荐产品。 总而言之，我们可以期待推理架构开始在推荐系统研究中占据前景。

1.  **Machine Reasoning for Recommendation**

1.  **Cross Domain Recommendation with Deep Neural Networks**

如今，许多大公司向客户提供多样化的产品或服务。例如，Google为我们提供网络搜索，移动应用程序和新闻服务;我们可以从亚马逊购买书籍，电子产品和衣服。单域推荐系统只关注一个域而忽略了其他域上的用户兴趣，这也加剧了稀疏性和冷启动问题。跨域推荐系统利用从源域学到的知识来辅助目标域推荐，为这些问题提供了理想的解决方案。跨领域推荐中研究最广泛的主题之一是转移学习，旨在通过使用从其他领域转移的知识来改善一个领域的学习任务。深度学习非常适合转移学习，因为它学习了解决不同领域变异的高级抽象。现有的几项工作表明深度学习在捕捉不同领域的概括和差异以及在跨域平台上产生更好的建议方面的功效。因此，这是一个很有希望但很少被探索的领域，预计会有更多的研究。

1.  **Deep Multi-Task Learning for Recommendation**

多任务学习在许多深度学习任务中取得了成功，从计算机视觉到自然语言处理。在所回顾的研究中，一些作品也将多任务学习应用于深度神经框架中的推荐系统，并在单一任务学习方面取得了一些改进。应用基于深度神经网络的多任务学习的优点有三方面：（1）一次学习几个任务可以通过推广共享隐藏表示来防止过度命中; （2）辅助任务提供可解释的输出以解释推荐; （3）多任务提供隐式数据增强，以减轻稀疏性问题。多任务可以在传统的推荐系统中使用，而深度学习使它们能够以更严格的方式集成。除了引入辅助任务之外，我们还可以为每个特定任务生成跨域推荐的多任务学习，为每个域生成建议。

1.  **Scalability of Deep Neural Networks for Recommendation**

大数据时代不断增长的数据量给实际应用带来了挑战。因此，可扩展性对于推荐模型在实际系统中的有用性至关重要，时间复杂度也是选择模型的主要考虑因素。幸运的是，深度学习已经证明了这一点。

在大数据分析中非常有效和有前途，特别是随着GPU计算能力的提高。然而，通过探索以下问题，应该研究如何有效推荐更多的未来工作：（1）非固定和流数据的增量学习，如大量的传入用户

和物品; （2）高维张量和多媒体数据源的计算效率; （3）通过参数的指数增长来平衡模型的复杂性和可扩展性。该领域一个有前途的研究领域涉及知识蒸馏，有论文已经在学习小型/紧凑型模型中进行了探索

用于推荐系统的推断。关键的想法是培养一个较小的学生模型，从大型教师模型中吸收知识。鉴于推理时间对于百万/亿用户规模的实时应用至关重要，我们认为这是另一个有希望的方向，值得进一步调查。

1.  **The Field Needs Better, More Unified and Harder Evaluation**

每次提出一个新的模型时，预期该出版物将根据若干基线提供评价和比较。大多数论文的基线和数据集的选择似乎是任意的，作者通常对数据集/基线的选择有自由的控制权。这里有几个问题。首先，这会产生一个不一致的分数报告，每个作者报告他们自己的结果分类。直到今天，对于模型的一般排序似乎还存在共识(值得注意的是，我们承认存在“没有免费午餐”定理)。偶尔，我们会发现结果可能是相互矛盾的，而且相对位置的变化非常频繁。这使得新神经模型的相对基准极具挑战性。问题是我们如何解决这个问题?看看邻近的领域(计算机视觉或自然语言处理)，这确实令人困惑。为什么没有MNIST, ImageNet或班组的推荐系统?因此，我们认为应该提出一套标准化的评估数据集。我们还注意到，MovieLens等数据集通常被许多实践者用于评估他们的模型。然而，测试分割通常是随机的。第二个问题是没有对评估程序的控制。为此，我们敦促推荐系统社区遵循CV/NLP社区，建立一个隐藏/盲测试集，其中预测结果只能通过web接口(如Kaggle)提交。最后，第三个反复出现的问题是，在推荐系统的结果中没有对测试样本的难易程度进行控制。时间的流逝是最好的吗?我们如何知道测试样本是否过于琐碎或无法推断?如果没有设计合适的测试集，我们认为实际上很难估计和测量该领域的进展。为此，我们认为推荐系统领域有很多值得计算机视觉或NLP社区学习的地方。

**※ Multilayer Perceptron based Recommendation****：**

  **（一）****Neural Extension of Traditional Recommendation Methods:**

1. **Neural Collaborative Filtering**
2. **Deep Factorization Machine**

![img](http://note.youdao.com/yws/public/resource/59bd177fe5faf6f78ed91365f57c6f27/xmlnote/894342BE8E504B1DB318876BE917B74F/79)

**（二）** **Feature Representation Learning with MLP：**

1. **Wide & Deep Learning**
2. **Collaborative Metric Learning (CML)**

![img](http://note.youdao.com/yws/public/resource/59bd177fe5faf6f78ed91365f57c6f27/xmlnote/A700BBFAA27F4D2DABA651A40DACBEE7/119)

**（三）Recommendation with Deep Structured Semantic Model：**

1. **Deep Semantic Similarity based Personalized Recommendation(DSPR)**
2. **Multi-View Deep Neural Network (MV-DNN)**

**※ Autoencoder based Recommendation**

将自动编码器应用于推荐系统有两种通用方法：

1. 使用自动编码器学习瓶颈层的低维特征表示。
2. 直接在重建层填充交互矩阵的空白。

**（一）Autoencoder-based Collaborative Filtering：**

1.  **AutoRec:**

- - - - 1. item-based AutoRec (I-AutoRec)
        2. user-based AutoRec(U-AutoRec)

​      CFN是AutoRec的扩展（ CFN的进一步扩展还包含辅助信息，在每一层中都注入辅助信息），具有以下两个优点：

- - 1. 它采用了去噪技术，使CFN更加健壮。
    2. 它结合了诸如用户简况和项目描述之类的辅助信息，以减轻稀疏性和冷启动影响。

​    

1. **Collaborative Denoising Auto-Encoder (CDAE)（ranking prediction)**

![img](http://note.youdao.com/yws/public/resource/59bd177fe5faf6f78ed91365f57c6f27/xmlnote/652547293DCC44B8B90F6B17B2CBE6D0/175)

**（二）Feature Representation Learning with Autoencoder：**自动编码器是一种功能强大的特征表示学习方法。同样，它也可以用于推荐系统中，从用户/项目内容特征中学习特征表示。

**Collaborative Deep Learning (CDL)：**CDL是一种分层贝叶斯模型，它将堆叠去噪自编码器(SDAE)集成到概率矩阵分解中。

![img](http://note.youdao.com/yws/public/resource/59bd177fe5faf6f78ed91365f57c6f27/xmlnote/B82894647D0E487FB1D15D157546FC9F/217)

1. **Collaborative Deep Ranking (CDR)：**CDR是专门针对top-n推荐的成对框架设计的。
2. **Deep Collaborative Filtering Framework：**它是通过协同过滤模型统一深度学习方法的一般框架。 该框架能够利用深度特征学习技术轻松构建混合协作模型。

**※ Convolutional Neural Networks based Recommendation**

**（一）Feature Representation Learning with CNNs：**

CNNs可用于从图像、文本、音频、视频等多种来源学习特征表示。

1. **CNNs for Image Feature Extraction**
2. **CNNs for Text Feature Extraction**
3. **CNNs for Audio and Video Feature Extraction**

**（二）CNNs based Collaborative filtering:**

Directly applying CNNs to vanilla collaborative filtering is  viable.

**（三）Graph CNNs for Recommendation：**

图卷积网络是非欧几里得数据的强大工具，例如：社交网络，知识图，蛋白质相互作用网络等。 推荐			    区域中的交互也可以被视为这样的结构化数据集（二分图）。 因此，它也可以应用于推荐任务。

**※ Recurrent Neural Networks based Recommendation**

**（一）Session-based Recommendation without User Identifier：**

1. **GRU4Rec**

**（二）Sequential Recommendation with User Identifier**

1. **Recurrent Recommender Network (RRN)**

![img](http://note.youdao.com/yws/public/resource/59bd177fe5faf6f78ed91365f57c6f27/xmlnote/DB1214D195314A2582E7B3737250B7A7/271)