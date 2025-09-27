[GitHub - YuanheZ/LoRA-One: LoRA-One: One-Step Full Gradient Could Suffice for Fine-Tuning Large Language Models, Provably and Efficiently (ICML2025 Oral)](https://github.com/YuanheZ/LoRA-One)

# 摘要
本文以大型语言模型中的低秩适配（LoRA）（Hu 等，2022）为案例，探讨理论如何指导并提升实际算法。我们严格证明，<font style="color:#DF2A3F;">在梯度下降下，LoRA 适配器与一步全微调梯度的特定奇异子空间对齐（对齐：LoRA 的低秩更新子空间 ≈ 梯度矩阵的主奇异子空间）。</font>这一结果表明，通过使用一步全梯度正确初始化适配器，可立即实现子空间对齐——该结论适用于线性与非线性模型。基于我们的理论，我们提出了一种理论驱动的算法 <font style="color:#DF2A3F;">LoRA-One</font>，其线性收敛性（以及泛化性）得以构建，且理论表明引入预条件子有助于缓解病态问题的影响。此外，我们的理论揭示了 LoRA-One 与其他基于梯度对齐方法之间的关联，有助于澄清此类算法设计中的误解。在自然语言理解、数学推理和代码生成的多个基准测试中，LoRA-One 相较于 LoRA 及其变体取得了显著的实证提升。

# introduction
**提出两个问题****🙋**

> Q1：如何在理论上刻画LoRA的低秩动态及其相关子空间对齐？  
>

> Q2：我们的理论结果如何为LoRA的实践算法设计提供贡献？
>

1. **对齐与算法设计原则**

:::color4
**📖****Spectral initialization 谱初始化**

从子空间对齐的角度识别了最优化初始化

![image](https://cdn.nlark.com/yuque/__latex/3c295d9d7539044e1a3ec2920396d909.svg)

![image](https://cdn.nlark.com/yuque/__latex/e57e0b40d5a5433963bc5037be4a0539.svg)

![image](https://cdn.nlark.com/yuque/__latex/160c5d0415185884dadeab492687a30b.svg)

![image](https://cdn.nlark.com/yuque/__latex/d68857ec08af8d2e5b32181505bc843c.svg)

通过** Spectral-init**，我们可以保证 ![image](https://cdn.nlark.com/yuque/__latex/ae1822d14c738a4b84328d789e9b7155.svg)很小，也就是说初始化时就已经很接近目标矩阵 ![image](https://cdn.nlark.com/yuque/__latex/6cbe179fa77b5a6af5c84309f72dd808.svg)。

🌟LoRA 的 **Spectral-init 初始化方式 + 一次完整梯度下降**，就已经能得到一个很好的近似解，不需要大量迭代。



❓** Spectral-init和Adalora的区别:**

+ **AdaLoRA 是一种 参数化的 SVD 低秩表示，训练时逐渐调整 **![image](https://cdn.nlark.com/yuque/__latex/a99c3dabbb2eeee45801b2b7d343cc65.svg)** 的秩，以控制参数预算.**
+ **Spectral-init 直接用一次 SVD 提供“好”的初始化，让 **![image](https://cdn.nlark.com/yuque/__latex/cf92c49c56cbddc6ba0d75c41bcc1b9f.svg)**一开始就逼近 **![image](https://cdn.nlark.com/yuque/__latex/6cbe179fa77b5a6af5c84309f72dd808.svg)**.**

:::



![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758869405112-926778d7-4b2d-4b7f-b027-4dc8c66a32e7.png)

# 线性模型下LoRA的分析
1. **<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">矩阵 </font>**![image](https://cdn.nlark.com/yuque/__latex/5ae2213501485496bc2add1b2b357665.svg)**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 与 </font>**![image](https://cdn.nlark.com/yuque/__latex/88ed89cef7b81cb7f50bb9a7c5275657.svg)**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 在 “右奇异子空间” 的严格对齐性：</font>**

![image](https://cdn.nlark.com/yuque/__latex/c48b76a2c5d48d72d77d8b6619786bf6.svg)

<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">对任意迭代步 </font>![image](https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">，“</font>![image](https://cdn.nlark.com/yuque/__latex/88ed89cef7b81cb7f50bb9a7c5275657.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 的正交补子空间部分” 与 “</font>![image](https://cdn.nlark.com/yuque/__latex/5ae2213501485496bc2add1b2b357665.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 的前</font>![image](https://cdn.nlark.com/yuque/__latex/414ffa3b6e46749d8cc021379e95bd6f.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">主成分子空间部分” 的</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">乘积矩阵是零矩阵，完全正交，没有交集</font>**

<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">“子空间对齐” 的本质是：</font>![image](https://cdn.nlark.com/yuque/__latex/5ae2213501485496bc2add1b2b357665.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 的有效子空间（即 </font>![image](https://cdn.nlark.com/yuque/__latex/5ae2213501485496bc2add1b2b357665.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 能发挥作用的维度范围）</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">完全包含在 </font>**![image](https://cdn.nlark.com/yuque/__latex/88ed89cef7b81cb7f50bb9a7c5275657.svg)**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 的主成分子空间中</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">。</font>

<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"></font>

2. <font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> </font>![image](https://cdn.nlark.com/yuque/__latex/88ed89cef7b81cb7f50bb9a7c5275657.svg)**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">与 </font>**![image](https://cdn.nlark.com/yuque/__latex/5fef476e49ed4eb0618a158f18065ea5.svg)**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 在“左奇异子空间”上的 “近似对齐”：</font>**

![image](https://cdn.nlark.com/yuque/__latex/6493ff47aaaf5cd673db9b0fbf8f72a7.svg)

<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">经过 </font>![image](https://cdn.nlark.com/yuque/__latex/df22f6ff2907974e0c9e9403e7405cff.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 步梯度下降后，</font>![image](https://cdn.nlark.com/yuque/__latex/5fef476e49ed4eb0618a158f18065ea5.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 中属于 “</font>![image](https://cdn.nlark.com/yuque/__latex/88ed89cef7b81cb7f50bb9a7c5275657.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 的核心左子空间（前 </font>![image](https://cdn.nlark.com/yuque/__latex/414ffa3b6e46749d8cc021379e95bd6f.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 左奇异子空间）” 的部分，与 </font>![image](https://cdn.nlark.com/yuque/__latex/88ed89cef7b81cb7f50bb9a7c5275657.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 中 “非核心左子空间（正交补子空间）” 的部分，</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">几乎没有重叠</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">（因为算子范数被限制在 </font>![image](https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg)<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);"> 这么小的量级）。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758874787961-a5365eaf-562a-450a-947d-a379272a12b0.png)

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758877268922-edd421a2-3cf8-41b2-a69f-c5da5422bc4a.png)

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758976782377-4d4278aa-33ca-44c9-a9e4-5c1f65aa107d.png)

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758877480787-8d20ef42-4a36-4546-97ea-9a65b536935f.png)

**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">LoRA-One 在 “秩不足” 和 “秩过足” 场景下，均能实现更优的风险下降与收敛性能</font>**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">；</font>

# <font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">实验Experiments</font>
1. One-Step单步梯度在真实微调任务中的能力

**baseline**：LoRA、LoRA+、P-LoRA、PiSSA、LoRA-GA、LoRA-Pro，主方法：秩为8的LoRA-One

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758903363932-94e6372e-15b6-46d4-bb18-82915d1cb593.png)

2. 自然语言生成

使用**LLaMA 2-7B**进行微调，**benchmarks**：GSM8K-D、GSM8K-CoT、MMLU、HumanEval

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758903831455-bce078fe-d264-44c4-832e-2fc601f0273f.png)

3. 数学推理

模型LLaMA 2-7B对数据集MetaMathQA（395K）进行4个周期的微调

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758904020066-f51f02a1-88da-4360-a2ed-444008892cfa.png)





**<font style="color:rgb(0, 0, 0);background-color:rgb(252, 252, 252);">  
</font>**

