### 中|[En](README.md)

# FreeLM: Fine-Tuning-Free Language Model

## 下载 Checkpoint
您可以从这个[链接](https://pan.baidu.com/s/1sI1CgWmunxtvpO_bfesJrA?pwd=m8wr)下载我们已经训练好的模型文件. 提取码: m8wr


## 快速执行
1. 加载模型 和 Tokenizer
```python
import torch
from transformers import GPT2Tokenizer
from modeling import FreeLMModel

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = %Checkpoint Dir Path%
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
model = FreeLMModel.from_pretrained(checkpoint_path).to(device)
```

2. 生成
```python
input_text = 'Hello'
model.eval()
output = model.generate(tokenizer.encode(input_text, return_tensors="pt", pad_token_id=tokenizer.pad_token_id).to(device),
                        max_length=20,
                        min_length=5,
                        no_repeat_ngram_size=2,
                        early_stopping=True,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        return_dict_in_generate=True,,
                        top_k=50,
                        top_p=0.9)
output_text = tokenizer.decode(output['sequences'][0], skip_special_tokens=True)
print(output_text)
```

3. 理解 
```python
test_text = [
    "i can tell you that there's no other reason why anyone should bother remembering it. [sep] Here, the movie review is emotionally negative. [cls]",
    "i can tell you that there's no other reason why anyone should bother remembering it. [sep] Here, the movie review is emotionally positive. [cls]",
]

input_batch = {k: v.to(device) for k, v in tokenizer(test_text, return_tensors="pt", return_length=True).items()}
output_batch = model(**input_batch, train_type='mtc')
probs = torch.softmax(output['logits'].reshape((output['logits'].shape[0], output['logits'].shape[-1])), dim=-1)
print(f"The probability that text 1 is correct = {probs[0][1]}")
print(f"The probability that text 2 is correct = {probs[1][1]}")
```

# 引用

如何我们的工作对您产生了帮助，请引用FreeLM的论文：
```
@article{freelm,
  author       = {Xiang Li and Xin Jiang and Xuying Meng and Aixin Sun and Yequan Wang},
  title        = {FreeLM: Fine-Tuning-Free Language Model},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.01616}
}
```