import torch
from transformers import BertTokenizer

# 加载整个模型
model = torch.load('model.pth')

# 设置为评估模式
model.eval()

# 输入你的数据
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer("i am happy 3", padding=True, truncation=True, return_tensors="pt")
inputs1 = tokenizer("i am sad 3", padding=True, truncation=True, return_tensors="pt")

test_output = model(**inputs)
test_output1 = model(**inputs1)
print(test_output, test_output1)
