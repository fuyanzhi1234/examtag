import os
import paddle
import paddlenlp
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.datasets import load_dataset
import re
from metric import MultiLabelReport
from eval import evaluate
from paddle.io import DataLoader, BatchSampler
import functools
from paddlenlp.data import DataCollatorWithPadding
import pandas as pd

def clean_text(text):
    text = text.replace("\r", "").replace("\n", "")
    text = re.sub(r"\\n\n", ".", text)
    return text

# 定义读取数据集函数
def read_custom_data(is_test=False, is_one_hot=True):

    file_num = 6 if is_test else 48
    filepath = 'raw_data/test/' if is_test else 'raw_data/train/'

    for i in range(file_num):
        f = open('{}labeled_{}.txt'.format(filepath, i))
        while True:
            line = f.readline()
            if not line:
                break
            data = line.strip().split('\t')
            # 标签用One-hot表示
            if is_one_hot:
                labels = [float(1) if str(i) in data[1].split(',') else float(0) for i in range(20)]
            else:
                labels = [int(d) for d in data[1].split(',')]
            yield {"text": clean_text(data[0]), "labels": labels}
        f.close()

# 定义读取数据集函数
def read_excel_data(labelnum, is_test=False):
    if is_test:
        filepath = 'raw_data/testdata.csv'
    else:
        filepath = 'raw_data/traindata.csv'

    data = pd.read_csv(filepath)
    # 取出知识点列
    knowledge = data.loc[:, 'knowledge']
    # 取出题目列
    exams = data.loc[:, 'exam']
    
    # 遍历exams
    total = len(exams)
    for index in range(total):
        examClean = clean_text(exams[index])
        if len(examClean) <= 10:
            continue
        # 将labels转换为One-hot表示
        labels = [float(1) if i == knowledge[index] else float(0) for i in range(labelnum)]

        print({"text": clean_text(examClean), "labels": labels})
        yield {"text": clean_text(examClean), "labels": labels}



# 加载中文ERNIE 3.0预训练模型和分词器
model_name = "ernie-3.0-medium-zh"
num_classes = 22
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=num_classes)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Adam优化器、交叉熵损失函数、自定义MultiLabelReport评价指标
optimizer = paddle.optimizer.AdamW(learning_rate=1e-4, parameters=model.parameters())
criterion = paddle.nn.BCEWithLogitsLoss()
metric = MultiLabelReport()
# 模型在测试集中表现
model.set_dict(paddle.load('ernie_ckpt/model_state.pdparams'))

# 也可以选择加载预先训练好的模型参数结果查看模型训练结果
# model.set_dict(paddle.load('ernie_ckpt_trained/model_state.pdparams'))

label_vocab = {
    0:"2A：实数大小比较",
    1:"29：实数与数轴",
    2:"算术平方根的非负性",
    3:"立方根的性质",
    4:"22：算术平方根",
    5:"整式的加减运算",
    6:"立方根的表示方法",
    7:"2C：实数的运算",
    8:"21：平方根的定义",
    9:"24：立方根的定义",
    10:"算术平方根",
    11:"23：非负数的性质：算术平方根",
    12:"26：无理数的定义",
    13:"25：计算器—数的开方",
    14:"实数的分类",
    15:"开平方",
    16:"28：实数的性质",
    17:"估算无理数大小",
    18:"27：实数的定义",
    19:"开立方",
    20:"算术平方根的定义",
    21:"2B：估算无理数的大小"
}

# load_dataset()创建数据集
test_ds = load_dataset(read_excel_data, is_test=True, lazy=False, labelnum = 22)
print("测试集样例:", test_ds[0])

# 数据预处理函数，利用分词器将文本转化为整数序列
def preprocess_function(examples, tokenizer, max_seq_length):
    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    result["labels"] = examples["labels"]
    return result

trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=128)
test_ds = test_ds.map(trans_func)

# collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠
collate_fn = DataCollatorWithPadding(tokenizer)

test_batch_sampler = BatchSampler(test_ds, batch_size=64, shuffle=False)

test_data_loader = DataLoader(dataset=test_ds, batch_sampler=test_batch_sampler, collate_fn=collate_fn)


print("ERNIE 3.0 在法律文本多标签分类test集表现", end= " ")





results = evaluate(model, criterion, metric, test_data_loader, label_vocab)

test_ds = load_dataset(read_excel_data, is_test=True, is_one_hot=False, lazy=False, labelnum = 22)
res_dir = "./results"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
with open(os.path.join(res_dir, "multi_label.tsv"), 'w', encoding="utf8") as f:
    f.write("text\tprediction\n")
    for i, pred in enumerate(results):
        f.write(test_ds[i]['text']+"\t"+pred+"\n")