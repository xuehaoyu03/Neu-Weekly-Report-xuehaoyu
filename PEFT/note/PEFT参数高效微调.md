[chatbot_prefix_tuning.ipynb](https://www.yuque.com/attachments/yuque/0/2025/ipynb/29704292/1758447120989-e8e330eb-67fd-41e3-86a1-df041e1fff8e.ipynb)

## 一、BitFit
**BitFit (Bias-Term Fine-Tuning)** 是一种大模型参数高效微调（PEFT, Parameter-Efficient Fine-Tuning）方法。  
它的核心思想是：**在下游任务中，仅训练模型的偏置参数（bias terms），而冻结所有其他权重参数**。

线性层和注意力层都有偏置项 ![image](https://cdn.nlark.com/yuque/__latex/d29c2e5f4926e5b0e9a95305650f6e54.svg)，BitFit 就只更新这些 ![image](https://cdn.nlark.com/yuque/__latex/d29c2e5f4926e5b0e9a95305650f6e54.svg)，而保持大部分参数（比如权重矩阵 ![image](https://cdn.nlark.com/yuque/__latex/a36915ecf0b5605493f5aeaf1480a9ac.svg)）不变。

需要训练的参数极少（通常占总参数量的 **0.1% ~ 0.5%**）。

```python
# bitfit
# 选择模型参数里面的所有bias部分

num_param = 0
for name, param in model.named_parameters():
    if "bias" not in name:
        param.requires_grad = False
    else:
        num_param += param.numel()

args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
# 模型训练
model = model.cuda()
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)
```

### 二、prompt-tuning
Prompt-Tuning的思想：冻结主模型全部参数，在训练数据前加入一小段Prompt，只训练Prompt的表示层，即一个Embedding模块。其中，Prompt又存在两种形式，一种是<font style="background-color:#FBDE28;">hardprompt</font>，一种是soft prompt。

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758435926557-4c8db597-f1a0-47a2-be05-0b4495962e49.png)

```python
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit

# Soft Prompt
# config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10)
# config
# Hard Prompt
config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM, # 自回归语言建模
    prompt_tuning_init=PromptTuningInit.TEXT, # 用文本初始化虚拟 token，即将文本编码成 token embedding 作为前缀的初始值。
    prompt_tuning_init_text="下面是一段人与机器人的对话。", # 初始化前缀的文本内容
    num_virtual_tokens=len(tokenizer("下面是一段人与机器人的对话。")["input_ids"]), # 初始化前缀的文本内容
    tokenizer_name_or_path="Langboat/bloom-1b4-zh" # 初始化前缀的文本内容
)


```

### 三、P-tuning
P-Tuning的思想：在Prompt-Tuning的基础上，对Prompt部分进行进一步的编码计算，加速收敛，具体来说，PEFT中支持两种编码方式，一种是<font style="background-color:#74B602;">LSTM</font>，一种是<font style="background-color:#74B602;">MLP</font>。与Prompt-Tuning不同的是，Prompt的形式只有SoftPrompt。

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758436086058-942e4ba2-0dc7-4cd0-9c16-0cbf6e352538.png)

```python
from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType

config = PromptEncoderConfig(
    task_type=TaskType.CAUSAL_LM, # 自回归语言建模，预测下一个 token，适合 GPT 类模型。
    num_virtual_tokens=10, # 前缀微调中“虚拟 token”的数量。
    encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
    encoder_dropout=0.1,
    encoder_num_layers=5, # 前缀 encoder 的隐藏层数量（MLP 层数）。
    encoder_hidden_size=1024 # 前缀 encoder 每层的隐藏层维度。
)

```

### Prefix-tuning
**但提示不只是加在输入 embedding 上，在每一层的 Self-Attention 里，额外引入一段“前缀键值对” **![image](https://cdn.nlark.com/yuque/__latex/63a34ad1e37192701eca1bb10bf55f4a.svg)

![image](https://cdn.nlark.com/yuque/__latex/689c062b89a1828ba8ee7fbe323cb338.svg)

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758436441700-b95dacd0-ed91-43b4-a04b-ff575c29af45.png)

```python
from peft import PrefixTuningConfig, get_peft_model, TaskType
config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM, # 自回归语言建模
    num_virtual_tokens=10, # 前缀长度，即虚拟 token 的数量
    prefix_projection=True # 是否对前缀向量进行 线性投影 
)

```

### LORA
![image](https://cdn.nlark.com/yuque/__latex/e603529418dc97fe27c2f35828caf24f.svg)

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758437182785-67c51e8d-ce23-4c39-a416-0ca32f3eddbb.png)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练的 GPT-2 模型和分词器
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
print(model)


from peft import get_peft_model, LoraConfig, TaskType

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
    inference_mode=False,          # 推理模式关闭，以进行训练
    r=8,                           # 低秩值 r
    lora_alpha=32,                 # LoRA 的缩放因子
    lora_dropout=0.1,              # Dropout 概率
)

# 将 LoRA 应用到模型中
model = get_peft_model(model, lora_config)
print(model)

# 查看 LoRA 模块
model.print_trainable_parameters()
```

开始微调（具体参考下面的代码）

[test.ipynb](https://www.yuque.com/attachments/yuque/0/2025/ipynb/29704292/1758439997653-5f4e1d0d-cace-42fc-993f-38e1beb9c44a.ipynb)

### IA3
A3的思想：抑制和放大内部激活，通过可学习的向量对激活值进行抑制或放大。具体来说，会对K、V、FFN三部分的值进行调整，训练过程中同样冻结原始模型的权重，只更新可学习的部分向量部分。训练完成后，与Lora类似，也可以将学习部分的参数与原始权重合并，没有额外推理开销。学习率：3e-3

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758440115939-46dcc201-f35d-49c2-8160-ae9532e610ca.png)

```python
from peft import IA3Config, TaskType, get_peft_model

config = IA3Config(task_type=TaskType.CAUSAL_LM)
```

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758442985735-bfe194a8-a367-4b8f-87e6-6aed6778fe79.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758443400647-a1104b18-87d3-48d0-b293-c80e8c6d5ac9.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758443468037-548b5ad2-a3b7-4f59-8f2c-cf5d383eaf6b.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758443536724-d19ce7b3-6622-4f9c-ac6f-ef1faeb4c54a.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758443869718-fa1b5659-6c1e-4bf8-bd15-e06a66bab141.png)

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758444008383-129c71da-358c-42a4-b4a1-37c924513770.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1758446935915-a646de5c-98b6-4606-bf9d-81e422f3d4ac.png)

