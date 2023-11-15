import os
import paddle
import paddlenlp
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

# 自定义数据集
from paddlenlp.datasets import load_dataset
import functools
import numpy as np

from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding

import time
import paddle.nn.functional as F
from metric import MultiLabelReport
from eval import evaluate
from common import clean_text, read_excel_data, readVocab
import matplotlib.pyplot as plt




label_vocab = readVocab('raw_data/20231103klmap.json')
num_classes = len(label_vocab) + 1

# load_dataset()创建数据集
train_ds = load_dataset(read_excel_data, is_test=False, lazy=False, labelnum = num_classes) 
test_ds = load_dataset(read_excel_data, is_test=True, lazy=False, labelnum = num_classes)

# lazy=False，数据集返回为MapDataset类型
print("数据类型:", type(train_ds))

# labels为One-hot标签
print("训练集样例:", train_ds[0])
print("测试集样例:", test_ds[0])
# 数据类型: <class 'paddlenlp.datasets.dataset.MapDataset'>
# 训练集样例: {'text': '2013年11月28日原、被告离婚时自愿达成协议，婚生子张某乙由被告李某某抚养，本院以（2013）宝渭法民初字第01848号民事调解书对该协议内容予以了确认，该协议具有法律效力，对原、被告双方均有约束力。', 'labels': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
# 测试集样例: {'text': '综上，原告现要求变更女儿李乙抚养关系的请求，本院应予支持。', 'labels': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

# 加载中文ERNIE 3.0预训练模型和分词器
model_name = "ernie-3.0-medium-zh"

bsize  = 64
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=num_classes)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 数据预处理函数，利用分词器将文本转化为整数序列
def preprocess_function(examples, tokenizer, max_seq_length):
    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    result["labels"] = examples["labels"]
    return result

trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=128)
train_ds = train_ds.map(trans_func)
test_ds = test_ds.map(trans_func)

# collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠
collate_fn = DataCollatorWithPadding(tokenizer)

# 定义BatchSampler，选择批大小和是否随机乱序，进行DataLoader
train_batch_sampler = BatchSampler(train_ds, batch_size=bsize, shuffle=True)
test_batch_sampler = BatchSampler(test_ds, batch_size=bsize, shuffle=False)
train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
test_data_loader = DataLoader(dataset=test_ds, batch_sampler=test_batch_sampler, collate_fn=collate_fn)

# Adam优化器、交叉熵损失函数、自定义MultiLabelReport评价指标
optimizer = paddle.optimizer.AdamW(learning_rate=1e-4, parameters=model.parameters())
criterion = paddle.nn.BCEWithLogitsLoss()
metric = MultiLabelReport()
# 在之前的模型上继续训练
# model.set_dict(paddle.load('savedmodel/model_state.pdparams'))

epochs = 100 # 训练轮次
ckpt_dir = "ernie_ckpt" #训练过程中保存模型参数的文件夹

global_step = 0 #迭代次数
tic_train = time.time()
best_f1_score = 0

# 画图
fig = plt.figure()
# plt.ylim(0, 100)
plt_acc = fig.add_subplot()
x = np.arange(1, epochs + 1, 1)
y = np.zeros(epochs)
y[0] = 100

# 绘制初始曲线
linet_acc, = plt_acc.plot(x, y)
plt_acc.set_title("plt_acc") # 设置标题

for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']

        # 计算模型输出、损失函数值、分类概率值、准确率、f1分数
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        probs = F.sigmoid(logits)
        metric.update(probs, labels)
        auc, f1_score, precison, recall = metric.accumulate()
        print("global step %d, epoch: %d, batch: %d, loss: %.5f, auc: %.5f, f1 score: %.5f, precison: %.5f, recall: %.5f"
                % (global_step, epoch, step, loss, auc, f1_score, precison, recall))


        # 每迭代10次，打印损失函数值、准确率、f1分数、计算速度
        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, auc: %.5f, f1 score: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, auc, f1_score,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        
        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        # 每迭代20次，评估当前训练的模型、保存当前最佳模型参数和分词器的词表等
        if global_step % 20 == 0:
            save_dir = ckpt_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            eval_f1_score = evaluate(model, criterion, metric, test_data_loader, label_vocab, if_return_results=False)
            if eval_f1_score > best_f1_score:
                best_f1_score = eval_f1_score
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
            # 画图
            y[epoch - 1] = eval_f1_score * 100
            linet_acc.set_ydata(y)
            # 重新绘制图形
            plt.draw()
            plt.pause(0.1)
plt.show()           