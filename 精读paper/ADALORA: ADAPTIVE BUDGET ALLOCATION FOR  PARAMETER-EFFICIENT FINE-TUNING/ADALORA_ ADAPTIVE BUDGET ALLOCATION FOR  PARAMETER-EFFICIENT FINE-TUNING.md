[Zhang 等 - 2023 - AdaLoRA Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.pdf](https://www.yuque.com/attachments/yuque/0/2025/pdf/29704292/1758693202385-4c738f03-dbe7-488f-8753-6f21ad8d7635.pdf)

[GitHub - QingruZhang/AdaLoRA: AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning (ICLR 2023).](https://github.com/QingruZhang/AdaLoRA)

# 摘要
在下游任务上对大规模预训练语言模型进行微调已成为自然语言处理领域的重要范式。然而，常规做法是微调预训练模型中的所有参数，当存在大量下游任务时，这种方法变得不可行。因此，研究者提出了多种微调方法，以参数高效的方式学习预训练权重的增量更新，例如低秩增量。这些方法通常在所有预训练权重矩阵上均匀分配增量更新的预算，而**<font style="color:#DF2A3F;">忽视了不同权重参数的重要性差异，导致微调性能次优</font>**。为弥合这一差距，我们提出AdaLoRA，它根据各**权重矩阵的重要性**得分自适应地分配参数预算。具体而言，AdaLoRA将增量更新参数化为奇异值分解形式。这种新颖的方法使我们能够**<font style="color:#DF2A3F;">有效剪枝</font>**不重要更新的奇异值，本质上即为减少其参数预算，同时避免了密集的精确SVD计算。我们在自然语言处理、问答和自然语言生成等多个预训练模型上进行了大量实验，以验证AdaLoRA的有效性。结果表明，AdaLoRA在多个基线方法上显著提升性能，尤其在<font style="color:#DF2A3F;">低预算</font>设置下优势更为明显。我们的代码已公开于[https://github.com/QingruZhang/AdaLoRA](https://github.com/QingruZhang/AdaLoRA)。

:::color5
**💨****拓展 奇异值分解 (SVD)**

给定一个矩阵![image](https://cdn.nlark.com/yuque/__latex/a29eb9ee2111d8b446d60a4ff740a5dc.svg)**，**我们可以做到**奇异值分解（SVD）：**![image](https://cdn.nlark.com/yuque/__latex/3d67fc625dd869caefb70787f3b85a4b.svg)

![image](https://cdn.nlark.com/yuque/__latex/0bac1bbb213180e5fc78a6d7d72d7582.svg)：**正交矩阵**（左奇异向量） ![image](https://cdn.nlark.com/yuque/__latex/dbcea6def86e27d8e2cab610b7b8d30a.svg)（单位矩阵）

![image](https://cdn.nlark.com/yuque/__latex/381e06e6e86d7f6d132a9889f6af2b91.svg)：正交矩阵（右奇异向量）

![image](https://cdn.nlark.com/yuque/__latex/244b5345b6849b66dfdbb135f0be4179.svg)：**对角矩阵**，对角线上非负实数就是奇异值（对角线上是非零元素，其余全是 0 的矩阵。）

矩阵 ![image](https://cdn.nlark.com/yuque/__latex/de951302f41d4707b9d80ca1af34dd0f.svg) 作用在向量空间时，会把一个单位圆/球拉伸、旋转、压缩。



**左奇异向量**（对应矩阵 ![image](https://cdn.nlark.com/yuque/__latex/8962133df4487c91759a74e97ee2a528.svg) 的特征向量）。

**右奇异向量**（对应矩阵 ![image](https://cdn.nlark.com/yuque/__latex/dd8ef5e6cdc4ee13855bc4cfbee8c3e4.svg) 的特征向量）。

:::

# 问题背景
鉴于存在大量下游任务，全微调要求每个任务都维护一个大型模型的独立副本，导致内存消耗极其昂贵。两条解决方案：1.一条研究路径专注于向PLMs中添加小型神经模块，并仅对每个任务的这些模块进行微调（adapter tuning）2.参数高效的方式对预训练权重的增量更新进行建模，而无需修改模型架构

:::color4
❌LORA的局限性：

LoRA 仍存在局限性，因为它为每个增量矩阵 ∆ 预设了相同的秩 r。忽视了在微调预训练模型时，各模块和层的权重矩阵的重要性存在显著差异这一事实。

:::

# 提出问题
**如何根据模块的重要性自适应分配参数预算，以提高参数高效微调的性能？**

我们不直接计算SVD，而是将∆参数化为∆ = P ΛQ以模拟SVD。我们在训练损失中额外添加了惩罚项。这种参数化方式避免了SVD的密集计算。将增量矩阵P ΛQ划分为三元组，每个三元组Gi包含第i个奇异值及其对应的奇异向量。分较低的三元组被赋予较低优先级，其奇异值被置零；重要性较高的三元组则保留用于微调。

# 方法method
**(i) 基于SVD的自适应方法，将增量矩阵以奇异值分解的形式表示；**

![image](https://cdn.nlark.com/yuque/__latex/55c980efbf97eba3a0b94be7ebc86737.svg)

其中![image](https://cdn.nlark.com/yuque/__latex/e455d965d8205a304645d9df4c018a03.svg)，![image](https://cdn.nlark.com/yuque/__latex/c6c8807b5ee413cc58ad993f898d2d3e.svg)为左、右奇异向量，![image](https://cdn.nlark.com/yuque/__latex/bef678da302e903bcc7297b67952ab20.svg)包含奇异值，![image](https://cdn.nlark.com/yuque/__latex/bef678da302e903bcc7297b67952ab20.svg) 初始化为零，而 ![image](https://cdn.nlark.com/yuque/__latex/ffd1905f6d4d60accedfa6b91be93ea9.svg) 和 ![image](https://cdn.nlark.com/yuque/__latex/4ef7132d0df72d9e3db76f6391960a3d.svg) 采用随机高斯初始化，保证其正交性，![image](https://cdn.nlark.com/yuque/__latex/8a4ac3d72df57c6d70b5c3bb5e5d2d94.svg)

![image](https://cdn.nlark.com/yuque/__latex/bef678da302e903bcc7297b67952ab20.svg) 在梯度下降的时候会剪枝调整，避免了复杂的![image](https://cdn.nlark.com/yuque/__latex/ee478355ffcf6c4a0f3088e78344838b.svg)运算。

:::color4
**⚠️****剪枝的过程**

双元组（LORA ![image](https://cdn.nlark.com/yuque/__latex/98e7cb7e81b3c4ef5f597ab97aa51fa1.svg)）剪枝会全部清0

但是三元组（ADALORA ![image](https://cdn.nlark.com/yuque/__latex/fbc72886658ead32f45756d20c900364.svg)）剪枝只会对奇异值![image](https://cdn.nlark.com/yuque/__latex/bef678da302e903bcc7297b67952ab20.svg)进行掩码，而奇异向量![image](https://cdn.nlark.com/yuque/__latex/c6c8807b5ee413cc58ad993f898d2d3e.svg)始终得以保留

:::

**<font style="color:rgb(0, 0, 0) !important;">正则化函数 </font>**![image](https://cdn.nlark.com/yuque/__latex/9ee807f9fdf0afcd205b47aca85f83b5.svg)

![image](https://cdn.nlark.com/yuque/__latex/c831b0db419bd9a15fa3f7d1564219e1.svg)

**损失函数**![image](https://cdn.nlark.com/yuque/__latex/9b4abab2d4d67001988fb0235d36fcd0.svg)

![image](https://cdn.nlark.com/yuque/__latex/78c8ea2459fd28c43fad3014e2caafc4.svg)

其中![image](https://cdn.nlark.com/yuque/__latex/4aa418d6f0b6fbada90489b4374752e5.svg)是正则化系数（超参数），**<font style="color:rgb(0, 0, 0) !important;">同时保证模型的 “任务性能” 和 “结构约束”</font>**

**<font style="color:rgb(0, 0, 0) !important;">其中</font>**![image](https://cdn.nlark.com/yuque/__latex/14458e7c1d8801c5e885259c81399358.svg)**<font style="color:rgb(0, 0, 0) !important;">的更新梯度</font>**![image](https://cdn.nlark.com/yuque/__latex/b9b1bc2b398aa48bce212f66e68ffd03.svg)**<font style="color:rgb(0, 0, 0) !important;">为</font>**

![image](https://cdn.nlark.com/yuque/__latex/d988cbd0621f541d35af3403c69ca997.svg)<font style="color:rgb(0, 0, 0) !important;">为学习率</font>

<font style="color:rgb(0, 0, 0) !important;"></font>

**(ii) 重要性感知的秩分配，基于我们新设计的重要性度量来****<font style="background-color:#CEF5F7;"></font>****修剪冗余的奇异值。**

<font style="color:rgba(0, 0, 0, 0.85);">🌟</font><font style="color:rgba(0, 0, 0, 0.85);">矩阵 </font>![image](https://cdn.nlark.com/yuque/__latex/358bf0ff9db7f3c35c2db5beb1cbccbf.svg)**<font style="color:rgb(0, 0, 0) !important;">基于重要性分数的</font>****<font style="color:#DF2A3F;">剪枝</font>****<font style="color:rgb(0, 0, 0) !important;">（Pruning）操作</font>**

![image](https://cdn.nlark.com/yuque/__latex/65302caa8fae24a1a942cad06c7babca.svg)

![image](https://cdn.nlark.com/yuque/__latex/9ebe6174fc71fab08df1b268d5a70e07.svg)

![image](https://cdn.nlark.com/yuque/__latex/41c3542ce0b2b44bc8b6037095c98e4a.svg)：<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">第 t 步时，“三元组”的</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">重要性分数 </font>****<font style="color:rgb(0, 0, 0);background-color:#CEF5F7;">🙋</font>****<font style="color:rgb(0, 0, 0);background-color:#CEF5F7;">如何来评价是否重要呢，见下</font>**

> <font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">若某个奇异值对应的重要性分数 </font>![image](https://cdn.nlark.com/yuque/__latex/72045b60e9b1ed98ae8409d7ea5cd7b9.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">处于所有分数 </font>![image](https://cdn.nlark.com/yuque/__latex/020cdb427131145c678c29474b032be1.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 的</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">前</font>**![image](https://cdn.nlark.com/yuque/__latex/67234df74d85adeb29e0096261e8c1ea.svg)**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">名</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">（即 “最重要的一部分”），则保留该奇异值</font>![image](https://cdn.nlark.com/yuque/__latex/193fc80893ed3e6ec5841938e8a381b5.svg)
>

1. <font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">确定</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">单参数</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">重要性：</font>

![image](https://cdn.nlark.com/yuque/__latex/be11625ac4d3ed0734361365137fd2a1.svg)

**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">指数移动平均，</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">减少 “小批量随机采样” 带来的波动</font>

<font style="color:#DF2A3F;background-color:rgb(252, 252, 252);">最终的参数重要性</font><font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">为：</font>![image](https://cdn.nlark.com/yuque/__latex/a6e110feb15018daedf31edd1914ec66.svg)

2. **<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">矩阵分解中的 “三元组结构”（由</font>**![image](https://cdn.nlark.com/yuque/__latex/a7a4569dd37afaca2a8d9d683a41520f.svg)**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">组成）的重要性</font>**

![image](https://cdn.nlark.com/yuque/__latex/1eded3fd85ef73ee40a293babcd41748.svg)

:::color1
**伪代码：**

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758644330738-3831f740-ac8e-445d-9c74-45603385180d.png)

<font style="color:rgb(0, 0, 0);background-color:#CEF5F7;">在每一步训练中，先</font>**<font style="color:rgb(0, 0, 0);background-color:#CEF5F7;">评估参数重要性</font>**<font style="color:rgb(0, 0, 0);background-color:#CEF5F7;">（敏感性、平滑、不确定性、综合重要性），再</font>**<font style="color:rgb(0, 0, 0);background-color:#CEF5F7;">更新参数</font>**<font style="color:rgb(0, 0, 0);background-color:#CEF5F7;">（常规梯度下降），同时</font>**<font style="color:rgb(0, 0, 0);background-color:#CEF5F7;">根据重要性和预算剪枝</font>**<font style="color:rgb(0, 0, 0);background-color:#CEF5F7;">（确保资源高效），最终得到微调后的模型参数。</font>

:::

**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">预算调度函数</font>**

![image](https://cdn.nlark.com/yuque/__latex/92a9508bd83acef9f3eeca467e3c7885.svg)

![image](https://cdn.nlark.com/yuque/__latex/67234df74d85adeb29e0096261e8c1ea.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">表示</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">第 t 步的 “资源预算”，所有增量矩阵的总秩（total rank）</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">，也就是 “所有奇异值的数量”。</font>

# <font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">实验EXPERIMENTS</font>
AdaLoRA对DeBERTaV3-base和BART-large进行微调。在自然语言理解、问答和自然语言生成任务提升，我们每隔 ∆T 步（例如 ∆T = 100）对奇异值进行剪枝。

**Baselines：****Full fine-tuning、Bitfit、Adapter tuning、LoRA**

### **自然语言理解**
使用DeBERTaV3-base，基于基准测试模型benchmark的**GLUE**，任务分为 基准包括两个单句分类任务、三个相似性与改述任务以及四个自然语言推理任务。

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758646534303-d84c948e-4038-47dd-baaf-dc32516bbfdb.png)

### 问答方面
使用SQuAD v1.1和 SQuAD模型 

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758646856244-c3414839-e153-4fb5-ab80-e1ec5091ba3f.png)

### <font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">自然语言生成</font>
使用**BART-large**模型进行微调。我们在两个数据集上评估模型性能：XSum和CNN/DailyMail。

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758681353522-4dbcf202-1e4e-487b-84b6-7347cc6f6784.png)

# 分析ANALYSIS
不同预算水平下的微调性能。我们将AdaLoRA与应用于所有权重矩阵的通用LoRA进行比较。

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758681621549-b9a49cb7-56c1-41b3-a167-9847a39c5225.png)

AdaLoRA与LoRA剪枝之间的比较

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758681700310-88c11c03-bbc8-4cbb-bf7e-e464e5652862.png)

**消融实验**

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758682221746-e542c86b-cf14-4b38-b638-40d9a5059c36.png)

<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">每个增量矩阵（不同类型的适配权重矩阵）在各层（Layer）的最终排名（The final rank）情况。颜色越深，排名数值越高。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758682371039-76c345c6-88a6-4e42-b3c7-d270ef495b4c.png)



