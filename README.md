### [中](README_zh.md)|En

# FreeLM: Fine-Tuning-Free Language Model


## Download Checkpoint
You can download the model files we have trained from this [link](https://pan.baidu.com/s/1sI1CgWmunxtvpO_bfesJrA?pwd=m8wr). This is the extraction code: m8wr


## Quick Run
1. Load Model & Tokenizer
```python
import torch
from transformers import GPT2Tokenizer
from modeling import FreeLMModel

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = %Checkpoint Dir Path%
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
model = FreeLMModel.from_pretrained(checkpoint_path).to(device)
```

2. Generation
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

3. Understanding 
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

## Citation

If you find our work useful, please consider citing FreeLM:
```
@article{freelm,
  author       = {Xiang Li and Xin Jiang and Xuying Meng and Aixin Sun and Yequan Wang},
  title        = {FreeLM: Fine-Tuning-Free Language Model},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.01616}
}
```