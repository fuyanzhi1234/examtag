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



# 加载中文ERNIE 3.0预训练模型和分词器
model_name = "ernie-3.0-medium-zh"
num_classes = 20
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
    0: "婚后有子女",
    1: "限制行为能力子女抚养",
    2: "有夫妻共同财产",
    3: "支付抚养费",
    4: "不动产分割",
    5: "婚后分居",
    6: "二次起诉离婚",
    7: "按月给付抚养费",
    8: "准予离婚",
    9: "有夫妻共同债务",
    10: "婚前个人财产",
    11: "法定离婚",
    12: "不履行家庭义务",
    13: "存在非婚生子",
    14: "适当帮助",
    15: "不履行离婚协议",
    16: "损害赔偿",
    17: "感情不和分居满二年",
    18: "子女随非抚养权人生活",
    19: "婚后个人财产"
}

# load_dataset()创建数据集
test_ds = load_dataset(read_custom_data, is_test=True, lazy=False)
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

test_ds = load_dataset(read_custom_data, is_test=True, is_one_hot=False, lazy=False)
res_dir = "./results"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
with open(os.path.join(res_dir, "multi_label.tsv"), 'w', encoding="utf8") as f:
    f.write("text\tprediction\n")
    for i, pred in enumerate(results):
        f.write(test_ds[i]['text']+"\t"+pred+"\n")