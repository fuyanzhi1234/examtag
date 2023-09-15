import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

# 1. 数据预处理
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

texts = ["i am happy 1111", "i am sad 2222", "i am happy 3", "i am sad 4", "i am happy 5", 
         "i am sad 6", "i am happy 7", "i am sad 8", "i am happy 9", "i am sad 10"]  # 题目
# 知识点标签
labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10

dataset = TextDataset(texts, labels, tokenizer, MAX_LEN)

# 划分训练集和验证集
train_data, val_data = train_test_split(dataset, test_size=0.2)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# 2. 构建模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 3. 设置优化器和学习率
optimizer = AdamW(model.parameters(), lr=2e-5)

# 4. 训练模型
for epoch in range(EPOCHS):
    for data in train_loader:
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        labels = data['labels']

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 5. 验证模型
model.eval()
for data in val_loader:
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    labels = data['labels']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

# 保存整个模型
torch.save(model, 'model.pth')

# 6. 预测
test_text = ["i am happy 11"]
test_encoding = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
test_output = model(**test_encoding)
print(test_output)