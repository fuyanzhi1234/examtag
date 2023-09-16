import paddle
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
import paddle.nn.functional as F
from datetime import datetime
import re

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

def clean_text(text):
    # 替换所有的数字、字母、换行符和空格
    text = re.sub(r"[a-zA-Z0-9\n\s]", "", text)
    return text

# 预测任务
def predict(sentence, model, tokenizer, label_vocab):
    # 对输入文本进行分词和编码
    inputs = tokenizer(sentence)

    # 将编码后的文本转换为模型所需的输入格式
    input_ids = paddle.to_tensor([inputs['input_ids']])
    print("input_ids : {}".format(input_ids))
    token_type_ids = paddle.to_tensor([inputs['token_type_ids']])
    print("token_type_ids : {}".format(token_type_ids))

    # 将输入数据传递给模型并获取输出结果
    logits =  model(input_ids, token_type_ids)

    # 对输出结果进行解码和后处理
    probs = F.sigmoid(logits)
    probs = probs.tolist()
    result_id = []
    result = []
    for prob in probs:
        # for c, pred in enumerate(prob):
        #     if pred > 0.2:
        #         result_id.append(c)
        #         result.append("Label:" + str(c) + "," + label_vocab[c] + "," + str(round(pred, 3)))
        maxpred = 0
        final_c  = 0
        for c, pred in enumerate(prob):
            if pred > maxpred:
                final_c = c
                maxpred = pred
        result_id.append(final_c)
        result.append("Label:" + str(final_c) + "," + label_vocab[final_c] + "," + str(round(maxpred, 3)))
                

    print("预测结果：", result_id)
    print("预测结果：", result)
    return result

# 加载中文ERNIE 3.0预训练模型和分词器
num_classes = len(label_vocab)
model = AutoModelForSequenceClassification.from_pretrained('./ernie_ckpt', num_classes=num_classes)
tokenizer = AutoTokenizer.from_pretrained('./ernie_ckpt')
# 输入你的句子
# sentence = "2014年8月25日，被告吴某甲向本院提起同居关系子女抚养纠纷诉讼，本院于2014年9月22日作出（2014）温苍龙民初字第572号民事判决书，判决余某与吴某甲所生育的儿子吴某乙由吴某甲抚养至独立生活，余某每年支付抚养费6000元并享有每月探望儿子吴某乙二次的权利。"
while True:
    sentence = input("Please enter a sentence: ")
    sentence = clean_text(sentence)
    print("Start predict:", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    predict(sentence, model, tokenizer, label_vocab)    
    print("End predict:", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
