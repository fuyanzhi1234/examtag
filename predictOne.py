import paddle
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
import paddle.nn.functional as F
from datetime import datetime
import re

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

def clean_text(text):
    text = text.replace("\r", "").replace("\n", "")
    text = re.sub(r"\\n\n", ".", text)
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
        for c, pred in enumerate(prob):
            if pred > 0.2:
                result_id.append(c)
                result.append("Label:" + str(c) + "," + label_vocab[c] + "," + str(round(pred, 3)))

    print("预测结果：", result_id)
    print("预测结果：", result)
    return result

# 加载中文ERNIE 3.0预训练模型和分词器
num_classes = 20
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
