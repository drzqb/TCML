"""
    中文文本分类
    CountVectorizer作为特征
"""

import os
import jieba
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import time

warnings.filterwarnings('ignore')


def cut_words(file_path):
    """
    对文本进行切词
    :param file_path: txt文本路径
    :return: 用空格分词的字符串
    """
    text_with_spaces = ''
    text = open(file_path, 'r', encoding='gb18030').read()
    textcut = jieba.cut(text)
    for word in textcut:
        text_with_spaces += word + ' '
    return text_with_spaces


def loadfile(file_dir, label):
    """
    将路径下的所有文件加载
    :param file_dir: 保存txt文件目录
    :param label: 文档标签
    :return: 分词后的文档列表和标签
    """
    file_list = os.listdir(file_dir)
    words_list = []
    labels_list = []
    for file in file_list:
        file_path = file_dir + '/' + file
        words_list.append(cut_words(file_path))
        labels_list.append(label)
    return words_list, labels_list


# 训练数据
train_words_list1, train_labels1 = loadfile('data/train/女性', '女性')
train_words_list2, train_labels2 = loadfile('data/train/体育', '体育')
train_words_list3, train_labels3 = loadfile('data/train/文学', '文学')
train_words_list4, train_labels4 = loadfile('data/train/校园', '校园')

train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4

# 测试数据
test_words_list1, test_labels1 = loadfile('data/test/女性', '女性')
test_words_list2, test_labels2 = loadfile('data/test/体育', '体育')
test_words_list3, test_labels3 = loadfile('data/test/文学', '文学')
test_words_list4, test_labels4 = loadfile('data/test/校园', '校园')

test_words_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4

stop_words = open('data/stop/stopword.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig')  # 列表头部\ufeff处理
stop_words = stop_words.split('\n')  # 根据分隔符分隔

# 计算单词权重
count = CountVectorizer(stop_words=stop_words, max_df=0.5)

train_features = count.fit_transform(train_words_list)

# 上面fit过了，这里transform
test_features = count.transform(test_words_list)


def get_text_classification(estimator, X, y, X_test, y_test):
    '''
    estimator: 分类器，必选参数
            X: 特征训练数据，必选参数
            y: 标签训练数据，必选参数
       X_test: 特征测试数据，必选参数
        y_tes: 标签测试数据，必选参数
       return: 返回值
           y_pred_model: 预测值
             classifier: 分类器名字
                  score: 准确率
                      t: 消耗的时间
                  matrix: 混淆矩阵
                  report: 分类评价函数

    '''
    start = time.time()

    print('\n>>>算法正在启动，请稍候...')
    model = estimator

    print('\n>>>算法正在进行训练，请稍候...')
    model.fit(X, y)
    print(model)

    print('\n>>>算法正在进行预测，请稍候...')
    y_pred_model = model.predict(X_test)
    print(y_pred_model)

    print('\n>>>算法正在进行性能评估，请稍候...')
    score = metrics.accuracy_score(y_test, y_pred_model)
    matrix = metrics.confusion_matrix(y_test, y_pred_model)
    report = metrics.classification_report(y_test, y_pred_model)

    print('>>>准确率\n', score)
    print('\n>>>混淆矩阵\n', matrix)
    print('\n>>>召回率\n', report)
    print('>>>算法程序已经结束...')

    end = time.time()
    t = end - start
    print('\n>>>算法消耗时间为：', t, '秒\n')
    classifier = str(model).split('(')[0]

    return y_pred_model, classifier, score, t, matrix, report


estimator_list, score_list, time_list = [], [], []

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost
import lightgbm
from sklearn import svm

# 0.02194380760192871 秒
knc = KNeighborsClassifier()
result = get_text_classification(knc, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 0.30817556381225586 秒
dtc = DecisionTreeClassifier()
result = get_text_classification(dtc, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 90.83265948295593 秒
mlp = MLPClassifier()
result = get_text_classification(mlp, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 0.009973287582397461 秒
bnb = BernoulliNB()
result = get_text_classification(bnb, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 1.1938064098358154 秒
gnb = GaussianNB()
result = get_text_classification(gnb, train_features.toarray(), train_labels, test_features.toarray(), test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 0.008979082107543945 秒
mnb = MultinomialNB()
result = get_text_classification(mnb, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 0.7517776489257812 秒
lr = LogisticRegression()
result = get_text_classification(lr, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 4.816923379898071 秒
gbdt = GradientBoostingClassifier()
result = get_text_classification(gbdt, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 1.5089595317840576 秒
rf = RandomForestClassifier()
result = get_text_classification(rf, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 0.329087495803833 秒
ab = AdaBoostClassifier()
result = get_text_classification(ab, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 1.4580988883972168 秒
xgb = xgboost.XGBClassifier()
result = get_text_classification(xgb, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 0.7360310554504395 秒
lgbm = lightgbm.LGBMClassifier()
result = get_text_classification(lgbm, train_features.toarray(), train_labels, test_features.toarray(), test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 2.302870273590088 秒
svc = svm.SVC()
result = get_text_classification(svc, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

print("model" + " " * 16 + " acc" + " " * 3 + "time")

for k in range(len(score_list)):
    modelname = estimator_list[k]
    lmn = len(modelname)

    if lmn >= 20:
        nr = modelname[:20]
    else:
        nr = modelname + " " * (20 - lmn)

    print(nr, " %.2f" % (score_list[k]), " %.6f" % (time_list[k]))

# model                 acc   time
# KNeighborsClassifier  0.61  0.022938
# DecisionTreeClassifi  0.77  0.312136
# MLPClassifier         0.86  99.901545
# BernoulliNB           0.77  0.010981
# GaussianNB            0.89  1.328537
# MultinomialNB         0.92  0.008978
# LogisticRegression    0.90  0.848904
# GradientBoostingClas  0.81  4.816923
# RandomForestClassifi  0.88  1.509989
# AdaBoostClassifier    0.57  0.330146
# XGBClassifier         0.81  1.500957
# LGBMClassifier        0.80  0.825791
# SVC                   0.85  2.392600
