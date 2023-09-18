import paddle
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
import paddle.nn.functional as F
from datetime import datetime
import re

label_vocab = { 
    0:"22：算术平方根",
    1:"21：平方根的定义",
    2:"I6：几何体的展开图",
    3:"MJ：圆与圆的位置关系",
    4:"G8：反比例函数与一次函数的交点问题"
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
model.eval()
# 输入你的句子
# sentence = "2014年8月25日，被告吴某甲向本院提起同居关系子女抚养纠纷诉讼，本院于2014年9月22日作出（2014）温苍龙民初字第572号民事判决书，判决余某与吴某甲所生育的儿子吴某乙由吴某甲抚养至独立生活，余某每年支付抚养费6000元并享有每月探望儿子吴某乙二次的权利。"
while True:
    sentence = input("Please enter a sentence: ")
    sentence = clean_text(sentence)
    print("Start predict:", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    predict(sentence, model, tokenizer, label_vocab)    
    print("End predict:", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
