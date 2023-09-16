from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# 假设有以下预测概率和真实标签
predictions = [0.2, 0.8, 0.9, 0.4]
labels = [0, 1, 0, 0]
predictions = [0.02890483, 0.03263339, 0.04271802, 0.05511478, 0.02434784,
       0.06324981, 0.02828374, 0.03784585, 0.03570158, 0.1016314 ,
       0.06055702, 0.16169488, 0.15920013, 0.1430632 , 0.04336781,
       0.05037033, 0.12465554, 0.05005378, 0.15908386, 0.04128664,
       0.04127666, 0.11402857]

labels = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
       0., 0., 0., 0., 0.]
# 创建准确率度量对象
accuracy = roc_auc_score(y_score=predictions, y_true=labels, average='micro')


print("准确率：", accuracy)
