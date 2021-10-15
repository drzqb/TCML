"""
    中文文本分类
    TFIDF作为特征
"""

import os
import jieba
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
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
train_words_list1, train_labels1 = loadfile('text classification/train/女性', '女性')
train_words_list2, train_labels2 = loadfile('text classification/train/体育', '体育')
train_words_list3, train_labels3 = loadfile('text classification/train/文学', '文学')
train_words_list4, train_labels4 = loadfile('text classification/train/校园', '校园')

train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4

# 测试数据
test_words_list1, test_labels1 = loadfile('text classification/test/女性', '女性')
test_words_list2, test_labels2 = loadfile('text classification/test/体育', '体育')
test_words_list3, test_labels3 = loadfile('text classification/test/文学', '文学')
test_words_list4, test_labels4 = loadfile('text classification/test/校园', '校园')

test_words_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4

stop_words = open('text classification/stop/stopword.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig')  # 列表头部\ufeff处理
stop_words = stop_words.split('\n')  # 根据分隔符分隔

# 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)

train_features = tf.fit_transform(train_words_list)

# 上面fit过了，这里transform
test_features = tf.transform(test_words_list)


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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost
import lightgbm
from sklearn import svm

# 0.022909879684448242 秒
knc = KNeighborsClassifier()
result = get_text_classification(knc, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 0.3570723533630371 秒
dtc = DecisionTreeClassifier()
result = get_text_classification(dtc, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 87.20000290870667 秒
mlp = MLPClassifier()
result = get_text_classification(mlp, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 0.008976221084594727 秒
bnb = BernoulliNB()
result = get_text_classification(bnb, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 1.091081142425537 秒
gnb = GaussianNB()
result = get_text_classification(gnb, train_features.toarray(), train_labels, test_features.toarray(), test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 0.007979631423950195 秒
mnb = MultinomialNB()
result = get_text_classification(mnb, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 0.8129186630249023 秒
lr = LogisticRegression()
result = get_text_classification(lr, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 1.4670474529266357 秒
rf = RandomForestClassifier()
result = get_text_classification(rf, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 0.4388270378112793 秒
ab = AdaBoostClassifier()
result = get_text_classification(ab, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 1.9542262554168701 秒
xgb = xgboost.XGBClassifier()
result = get_text_classification(xgb, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 0.4667503833770752 秒
lgbm = lightgbm.LGBMClassifier()
result = get_text_classification(lgbm, train_features, train_labels, test_features, test_labels)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# 2.6062283515930176 秒
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
# KNeighborsClassifier  0.89  0.019946
# DecisionTreeClassifi  0.71  0.361142
# MLPClassifier         0.91  87.738459
# BernoulliNB           0.77  0.008976
# GaussianNB            0.88  1.225822
# MultinomialNB         0.84  0.007950
# LogisticRegression    0.92  0.847191
# RandomForestClassifi  0.84  1.464180
# AdaBoostClassifier    0.62  0.426850
# XGBClassifier         0.83  1.866001
# LGBMClassifier        0.81  0.538560
# SVC                   0.93  2.603263