import paddle
import numpy as np
import paddle.nn.functional as F

# 构建验证集evaluate函数
@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, label_vocab, if_return_results=True):

    model.eval()
    metric.reset()
    losses = []
    results = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        probs = F.sigmoid(logits)
        losses.append(loss.numpy())
        metric.update(probs, labels)
        if if_return_results:
            probs = probs.tolist()
            for prob in probs:
                result = []
                for c, pred in enumerate(prob):
                    if pred > 0.5:
                        result.append(label_vocab[c])
                results.append(','.join(result))

    auc, f1_score, precison, recall = metric.accumulate()
    print("eval loss: %.5f, auc: %.5f, f1 score: %.5f, precison: %.5f, recall: %.5f" %
          (np.mean(losses), auc, f1_score, precison, recall))
    model.train()
    metric.reset()
    if if_return_results:
        return results
    else:
        return f1_score